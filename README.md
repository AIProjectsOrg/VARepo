# VARepo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1adLKs0VtXp50P37aBb1ZN4qpXhlA2vYH?usp=sharing)

[![](https://img.shields.io/badge/Windows-11-0078D6?style=flat-square&logo=Windows)](https://www.microsoft.com/en-us/windows/)
[![](https://img.shields.io/badge/Cuda-11.8-6B8E23?style=flat-square&logo=Nvidia)](https://developer.nvidia.com/cuda-11.6-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)
[![](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=Python)](https://www.python.org/)
[![](https://img.shields.io/badge/PyTorch-000000?style=flat-square&logo=PyTorch)](https://pytorch.org/)

## 📚 Module Overview

TODO

## 1. Dataset

1.1 Smoking CCTV Detection
- The original dataset was released by the Roboflow user SmokingCigaretteCCTV and is available at [[Original Dataset]](https://universe.roboflow.com/smokingcigarettecctv/smoking-cctv-detection).
- This dataset is a **rare dataset** that focuses on **realistic CCTV scenarios**, rather than the open-angle or close-up perspectives commonly seen in most existing datasets.
- However, the **original annotations** were **incomplete**, where i) person instances were not labelled, and ii) many existing class instances were missing or incorrectly labelled.
- We **re-annotated** the dataset to better align it with the scope of our study.
- Specifically, the revised dataset contains **207 images** annotated in YOLOv8 format, with **three object classes**: 1) cigarette, 2) person, and 3) smoke.
- The images are split into training, validation, and test sets using a standard 7:2:1 ratio, resulting in **145 training images**, **41 validation images**, and **21 test images**. 
- Our re-annotated dataset can be accessed via the [[Dataset link]](https://universe.roboflow.com/smoking-gqlqh/smoking-cctv-detection-x4fjr), or through the Python code below.

```python
from roboflow import Roboflow
rf = Roboflow(api_key="rvc5pEYx6sd3cZ8EBcDW")
project = rf.workspace("smoking-gqlqh").project("smoking-cctv-detection-x4fjr")
version = project.version(4)
dataset = version.download("yolov8")
```

1.2 Smoking Person Detection
- We propose using Smoking-Person-Detection Dataset as a complementary training dataset.
- The original dataset consists of 2,789 images annotated in YOLOv8 format, with three labelled classes: 1) cigarette, 2) person, and 3) smoke.
- However, the **smoke image**s in this dataset are significantly **out-of-distribution** compared to those in the Smoking-CCTV-Detection dataset.
- Specifically, these images are close-up views of smoke, which differ greatly from the surveillance or CCTV angles central to our target application.
- Therefore, all images containing the smoke label were removed, resulting in a **filtered dataset** of **1,960 training images** (we discard validation and test set, since this dataset is meant for complementary training dataset).
- Our re-annotated dataset can be accessed via the [[Dataset link]](https://universe.roboflow.com/smoking-gqlqh/smoking-person-detection-2-ijwga), or through the Python code below. 

```python
from roboflow import Roboflow
rf = Roboflow(api_key="rvc5pEYx6sd3cZ8EBcDW")
project = rf.workspace("smoking-gqlqh").project("smoking-person-detection-2-ijwga")
version = project.version(1)
dataset = version.download("yolov8")
```

1.3 Synthetic Data Generation Pipeline
- This project applies the **Copy-Paste augmentation** technique to generate synthetic data, inspired by [this paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf).
- This method enriches training diversity by **cropping objects from source images** and **pasting them onto new background scenes**, enabling the creation of complex and varied samples without additional data collection. 
- In our implementation, all annotated objects from each source image are selected and pasted onto a target image that has been resized to the same resolution.
- ⚠️ We **STRONGLY** recommend generating your synthetic data based on your intended background and CCTV angles, so that the model can better adapt to your specific application use case.
- 💡 But if your dataset is big enough, then feel free to use more diverse background images.
- Notably, our synthetic dataset generation contributes to one of the **Top 3 Winning Teams** in the [IJCNN 2025 Drone vs Bird Detection Challenge](https://github.com/yjwong1999/IJCNN2025-DvB).

```bash
# The synthetic data generation code
python3 synthetic_data_generation.py --image_dir "/path/to/your/yolo/images" --label_dir "/path/to/your/yolo/txt/labels" --bg_dir "/path/to/your/background/images"
```

- You can get our compiled dataset below, which includes **1.1 Smoking CCTV Detection**, **1.2 Smoking Person Detection**, and our sample **synthetic dataset** we generated using Paris or Not Paris Dataset as background images.
- Please ONLY use the train set of the following dataset, since **1.1 Smoking CCTV Detection** is the main validation/test scenario.

```python
# Compiled Dataset
# 1.1 Smoking CCTV Detection + 1.2 Smoking Person Detection + 1.3 Synthetic Dataset Generation

from roboflow import Roboflow
rf = Roboflow(api_key="rvc5pEYx6sd3cZ8EBcDW")
project = rf.workspace("smoking-gqlqh").project("compiled-smoking-dataset-wkc8l")
version = project.version(1)
dataset = version.download("yolov8")
```

## 2. Training

Model training
model generation in ONNX


3. NX Meta Server
