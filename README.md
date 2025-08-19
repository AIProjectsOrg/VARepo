# VARepo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jWLeVjXbc6F99JJNhKAvoJ5dIl5ua4Ch?usp=sharing)

[![](https://img.shields.io/badge/Windows-11-0078D6?style=flat-square&logo=Windows)](https://www.microsoft.com/en-us/windows/)
[![](https://img.shields.io/badge/Cuda-11.8-6B8E23?style=flat-square&logo=Nvidia)](https://developer.nvidia.com/cuda-11.6-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)
[![](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=Python)](https://www.python.org/)
[![](https://img.shields.io/badge/PyTorch-000000?style=flat-square&logo=PyTorch)](https://pytorch.org/)

## 📚 Module Overview

TODO

## 1. Dataset

1.1 Smoking CCTV Detection [[Dataset link]](https://universe.roboflow.com/smoking-gqlqh/smoking-cctv-detection-x4fjr)

- The original dataset was released by the Roboflow user SmokingCigaretteCCTV and is available at [[link]](https://universe.roboflow.com/smokingcigarettecctv/smoking-cctv-detection).
- This dataset is a **rare dataset** that focuses on **realistic CCTV scenarios**, rather than the open-angle or close-up perspectives commonly seen in most existing datasets.
- However, the **original annotations** were **incomplete**, where i) person instances were not labelled, and ii) many existing class instances were missing or incorrectly labelled.
- We **re-annotated** the dataset to better align it with the scope of our study.
- Specifically, the revised dataset contains 207 images annotated in YOLOv8 format, with **three object classes**: 1) cigarette, 2) person, and 3) smoke.
```python
from roboflow import Roboflow
rf = Roboflow(api_key="rvc5pEYx6sd3cZ8EBcDW")
project = rf.workspace("smoking-gqlqh").project("smoking-cctv-detection-x4fjr")
version = project.version(4)
dataset = version.download("yolov8")
```

1.2 Smoking Person Detection [[Dataset link]](https://universe.roboflow.com/smoking-gqlqh/smoking-person-detection-2-ijwga)

1.3 Synthetic Dataset

1.1 labeled dataset
- synthetic dataset

## 2. Training

Model training
model generation in ONNX


3. NX Meta Server
