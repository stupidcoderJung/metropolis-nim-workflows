# NVDINOv2 NIM으로 Few Shot 분류하기

![few shot 데모](readme_assets/few_shot.gif)

## 소개

이 예제는 NVDINOv2 NIM을 사용해 벡터 데이터베이스와 결합함으로써 Few Shot 분류를 구현하는 방법을 소개합니다.

NVDINOv2 모델은 범용 비전 임베딩 모델입니다. NVDINOv2 NIM API를 사용하면 이미지 집합에 대해 빠르게 임베딩을 생성할 수 있습니다. 이렇게 생성한 임베딩을 벡터 데이터베이스에 저장하고 클러스터링 및 검색에 활용하면, 모델 학습이나 로컬 GPU 없이도 Few Shot 분류 파이프라인을 구축할 수 있습니다. 저장소에는 파이프라인 구현 과정을 안내하는 주피터 노트북과 Few Shot 분류를 손쉽게 실험할 수 있는 Gradio 데모가 포함되어 있습니다.

![few shot 아키텍처 다이어그램](readme_assets/few_shot_arch_diagram.png)

## 설정

**참고**: 이 예제의 노트북과 데모는 Milvus-Lite의 [Windows 미지원](https://github.com/milvus-io/milvus/issues/34854)으로 인해 Windows에서 직접 실행되지 않습니다. Windows 사용자는 WSL 환경에서 노트북과 데모를 실행해 주세요. Mac과 Linux는 지원됩니다.

### 리포지토리 클론
```
git clone https://github.com/NVIDIA/metropolis-nim-workflows
cd metropolis-nim-workflows/nim_workflows/nvdinov2_few_shot
```

### 가상환경 생성 및 활성화 (선택 사항)

Python 의존성을 설치하기 전에 가상환경을 구성하는 것이 좋습니다. 가상환경 생성 방법은 [Python 문서](https://docs.python.org/3/tutorial/venv.html)를 참고하세요.

```
python3 -m venv venv 
source venv/bin/activate
```

의존성 설치
```
python3 -m pip install -r requirements.txt
```

## 워크숍 노트북 (선택 사항)

NVDINOv2 사용법을 튜토리얼 형태로 살펴보고 싶다면 워크숍 노트북을 실행하세요. 노트북에서는 NVDINOv2로 임베딩을 생성하고, 벡터 데이터베이스를 이용해 Few Shot 분류 파이프라인을 구축하는 과정을 단계별로 안내합니다. 그렇지 않다면 다음 섹션의 Few Shot 분류 데모를 바로 실행해도 됩니다.

주피터 노트북 실행
```
python3 -m notebook 
```

위 명령을 실행하면 주피터 노트북 웹 인터페이스가 열립니다. 저장소에서 ```nvdinov2_nim_workshop.ipynb``` 노트북을 찾아 튜토리얼을 진행할 수 있습니다.

## Few Shot 분류 데모

Few Shot 분류 데모는 학습 없이도 Few Shot 분류 모델을 구성할 수 있도록 지원하는 Gradio UI입니다. UI에서 분류할 클래스를 지정하고, 소수의 예시 이미지를 업로드한 뒤 새로운 이미지를 분류할 수 있습니다.

![Few Shot 이미지](readme_assets/few_shot_still.png)

Few Shot 분류 데모를 실행하려면 `main.py` 스크립트를 필요한 인자와 함께 직접 실행하세요.

```
usage: main.py [-h] [--gradio_port GRADIO_PORT] api_key {nvclip,nvdinov2}

NVDINOv2 Few Shot Classification

positional arguments:
  api_key               NVIDIA NIM API Key
  {nvclip,nvdinov2}     Embedding model to use for few shot classification.

options:
  -h, --help            show this help message and exit
  --gradio_port GRADIO_PORT
                        Port to run Gradio UI
```

예시:

```
python3 main.py nvapi-*** nvdinov2
```

데모는 NVCLIP을 임베딩 생성기로 사용해 실행할 수도 있습니다.
```
python3 main.py nvapi-*** nvclip
```

필수 인자는 NIM API 키와 사용할 모델 선택뿐입니다. 스크립트를 실행하면 Gradio UI를 ```http://localhost:7860```에서 확인할 수 있습니다. NVDINOv2를 사용할 경우, 추가하거나 추론하는 각 샘플 이미지는 1개의 NIM API 크레딧을 소비합니다.

