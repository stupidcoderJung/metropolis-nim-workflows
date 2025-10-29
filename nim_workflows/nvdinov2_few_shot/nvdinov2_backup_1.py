# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import io
import time
import uuid
import json
import typing as T
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm


class NVDINOv2:
    def __init__(self, api_key: str, base_url: str = "https://ai.api.nvidia.com/v1/cv/nvidia/nv-dinov2"):
        """Initialize with NV-DINOv2 url and API key"""
        self.base_url = base_url
        self.api_key = api_key
        self.header_auth = f"Bearer {self.api_key}"

        # 기본 타임아웃/재시도 정책
        self.request_timeout = 60      # 초
        self.max_retries = 2           # 실패 시 재시도 횟수(총 3회 시도)
        self.retry_backoff = 1.0       # 재시도 간 대기(초), 지수 백오프 사용

    # ---------- 내부 유틸 ----------

    def _extract_embedding(self, resp_json: T.Any) -> T.Optional[T.List[float]]:
        """
        응답에서 embedding 벡터를 안전하게 추출.
        다양한 포맷을 방어적으로 처리.
        """
        if not isinstance(resp_json, dict):
            return None

        # (A) 기대 포맷: {"metadata": [{"embedding": [...]}], ...}
        if "metadata" in resp_json and isinstance(resp_json["metadata"], list):
            md0 = resp_json["metadata"][0] if resp_json["metadata"] else None
            if isinstance(md0, dict) and "embedding" in md0:
                return md0["embedding"]

        # (B) 평탄 포맷: {"embedding": [...]}
        if "embedding" in resp_json and isinstance(resp_json["embedding"], (list, tuple)):
            return list(resp_json["embedding"])

        # (C) data/outputs/results 등의 래핑 포맷들(추정)
        # 예: {"data":[{"embedding":[...]}]}
        for k in ("data", "outputs", "results"):
            if k in resp_json and isinstance(resp_json[k], list) and resp_json[k]:
                first = resp_json[k][0]
                if isinstance(first, dict) and "embedding" in first:
                    return first["embedding"]

        # (D) error 포맷: {"error": "..."} 등
        # 여기서 None을 반환하면 상위에서 실패로 취급
        return None

    def _upload_asset(self, image_path_or_pil, description: str = "input image") -> uuid.UUID:
        """
        NVCF 자산 업로드. 문자열 경로나 PIL.Image.Image 모두 허용.
        업로드는 JPEG로 통일.
        """
        assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"

        headers = {
            "Authorization": self.header_auth,
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        s3_headers = {
            "x-amz-meta-nvcf-asset-description": description,
            "content-type": "image/jpeg",
        }

        payload = {"contentType": "image/jpeg", "description": description}

        # 1) 업로드 슬롯 발급
        r = requests.post(assets_url, headers=headers, json=payload, timeout=self.request_timeout)
        r.raise_for_status()
        j = r.json()
        asset_url = j["uploadUrl"]
        asset_id = j["assetId"]

        # 2) 이미지를 RGB JPEG로 변환하여 바이트 업로드
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        elif isinstance(image_path_or_pil, Image.Image):
            image = image_path_or_pil.convert("RGB")
        else:
            raise TypeError("image_path_or_pil must be a str (path) or PIL.Image.Image")

        buf = io.BytesIO()
        image.save(buf, format="JPEG")

        r = requests.put(asset_url, data=buf.getvalue(), headers=s3_headers, timeout=max(self.request_timeout, 300))
        r.raise_for_status()
        return uuid.UUID(asset_id)

    def _post_infer(self, asset_id: uuid.UUID) -> dict:
        """
        추론 요청. 필요 시 재시도/백오프.
        """
        payload = {"messages": []}
        asset_list = f"{asset_id}"
        headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
            "NVCF-INPUT-ASSET-REFERENCES": asset_list,
            "NVCF-FUNCTION-ASSET-IDS": asset_list,
            "Authorization": self.header_auth,
        }

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(self.base_url, headers=headers, json=payload, timeout=self.request_timeout)
                # 4xx/5xx면 예외
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * (2 ** attempt))
                else:
                    # 재시도 소진
                    raise last_err

    def _embed_once(self, image_path_or_pil) -> dict:
        """
        한 장 처리: 업로드 → 추론 → JSON 반환
        (원시 JSON을 반환하여 상위에서 meta/embedding 선택)
        """
        asset_id = self._upload_asset(image_path_or_pil)
        resp_json = self._post_infer(asset_id)
        return resp_json

    # ---------- 퍼블릭 API ----------

    def __call__(self, image_paths_or_pils, workers: int = 8, return_meta: bool = False):
        """
        이미지 경로(str) 또는 PIL.Image(Image)의 리스트를 받아 임베딩 수행.
        - return_meta=True  → 원시 JSON들의 리스트를 반환
        - return_meta=False → 임베딩 벡터(list[float])들의 리스트를 반환
        """
        # 단일 입력도 허용
        if isinstance(image_paths_or_pils, (str, Image.Image)):
            image_paths_or_pils = [image_paths_or_pils]

        inputs = list(image_paths_or_pils)

        responses = []
        futures = []

        print("submitting requests")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for x in tqdm(inputs):
                futures.append(ex.submit(self._embed_once, x))

            print("collecting responses")
            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    responses.append(fut.result())
                except Exception as e:
                    # 실패 건도 리스트에 기록해서 길이 대응 유지(혹은 스킵 가능)
                    responses.append({"error": str(e)})

        # 순서를 입력 순서로 맞추고 싶다면 as_completed 대신 enumerate 기반으로 수집하세요.
        # 여기서는 안정성을 위해 실패 건을 에러 객체 형태로 남김.

        if return_meta:
            return responses

        # 임베딩만 추출
        embeddings = []
        failed = 0
        for r in responses:
            emb = self._extract_embedding(r)
            if emb is None:
                failed += 1
                # 실패 건은 None 대신 예외를 내거나, 스킵하거나, 플레이스홀더를 넣을 수 있음
                # 여기선 스킵 대신 명시적으로 None을 넣고 후단에서 필터링할 수 있게 함
                embeddings.append(None)
            else:
                embeddings.append(emb)

        if failed:
            print(f"[WARN] {failed} item(s) failed to produce embeddings.")
        return embeddings