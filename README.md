# Smoke Detection in Surveillance Footage via Video Analytics (VA)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1adLKs0VtXp50P37aBb1ZN4qpXhlA2vYH?usp=sharing)

[![](https://img.shields.io/badge/Windows-11-0078D6?style=flat-square&logo=Windows)](https://www.microsoft.com/en-us/windows/)
[![](https://img.shields.io/badge/Cuda-11.8-6B8E23?style=flat-square&logo=Nvidia)](https://developer.nvidia.com/cuda-11.6-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)
[![](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=Python)](https://www.python.org/)
[![](https://img.shields.io/badge/PyTorch-000000?style=flat-square&logo=PyTorch)](https://pytorch.org/)

## 📚 Module Overview

## 📋 Table of content
 1. [Dataset](https://github.com/AIProjectsOrg/VARepo?tab=readme-ov-file#1-dataset)
 2. [Training](https://github.com/AIProjectsOrg/VARepo?tab=readme-ov-file#2-training)
 3. [NX Meta Server](https://github.com/AIProjectsOrg/VARepo?tab=readme-ov-file#3-nx-meta-server)

## 1. Dataset

### 1.1 Smoking CCTV Detection
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

### 1.2 Smoking Person Detection
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

### 1.3 Synthetic Data Generation Pipeline
- This project applies the **Copy-Paste augmentation** technique to generate synthetic data, inspired by [this paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf).
- This method enriches training diversity by **cropping objects from source images** and **pasting them onto new background scenes**, enabling the creation of complex and varied samples without additional data collection. 
- In our implementation, all annotated objects from each source image are selected and pasted onto a target image that has been resized to the same resolution.
- ⚠️ We **STRONGLY** recommend generating your synthetic data based on your intended background and CCTV angles, so that the model can better adapt to your specific application use case.
- 💡 But if your dataset is big enough, then feel free to use more diverse background images.
- Notably, our synthetic dataset generation contributes to one of the **Top 3 Winning Teams** in the [IJCNN 2025 Drone vs Bird Detection Challenge](https://d197for5662m48.cloudfront.net/documents/publicationstatus/254531/preprint_pdf/0e4495c13bbf87af25543ab78189b3f0.pdf).

```bash
# The synthetic data generation code
# The generated data will be saved into the original image_dir and label_dir
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

The training code is provided in Google Colab link [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1adLKs0VtXp50P37aBb1ZN4qpXhlA2vYH?usp=sharing).

### 2.1 Model Selection and Training

We use the well-established `YOLOv8` model from the `Ultralytics` module as our object detection model. Specifically, the specifications of our model training are defined below. The remaining hyperparameters were set to their default values. 
1. Model: YOLOv8n
2. Epochs: 50
3. Training Batch Size: 16
4. Image Size: 640 x 640

💡 Improvement Suggestions:
- We suggest increasing the image size to 960 × 960 if your hardware supports higher input resolution. Empirical results indicate that 960 × 960 performs better, but our hardware only supports up to 640 × 640.
- Using YOLOv8s can improve performance. However, if you must choose between increasing the image resolution to 960 × 960 or upgrading the model from YOLOv8n to YOLOv8s, we recommend prioritizing image resolution. If your hardware can still support it afterward, then consider switching to YOLOv8s.
- ⚠️ Collecting additional data is more effective than naively scaling up your model size. We recommend gathering a small dataset from your actual deployment environment and training your model with it—alongside our provided dataset. This approach typically yields better results than simply increasing model size.

### 2.2 Balanced Sampling

The dataset used in this project is highly imbalanced, with a notable shortage of smoke instances, as illustrated in the figure below. Figure (a) is the Dataset 1.1 Smoking CCTV Detection, Figure (b) is 1.1 Smoking CCTV Detection + 1.2 Smoking Person Detection.

<img src="assets/Class Imbalanced.png" alt="Class Distribution" width="600"/>

We adopted balanced sampling using [Weighted Dataloader](https://y-t-g.github.io/tutorials/yolo-class-balancing/)
- Instead of modifying loss functions or undersampling majority classes, the author creates a custom YOLOWeightedDataset class that:
- Counts instances per class and computes inverse frequency weights.
- Aggregates label weights per image using functions like np.mean or np.sum.
- Calculates sampling probabilities to ensure minority classes appear more frequently in training batches.
- Overrides the `__getitem__` method to sample images based on these probabilities

Notably, we used `balanced sampling` in our [Winning Solution for ICIP 2025 Competition](https://d197for5662m48.cloudfront.net/documents/publicationstatus/270057/preprint_pdf/3a6eaf4b76a0286b76755e0d2a091fd3.pdf).

## 3. NX Meta Server

There is not much to elaborate on regarding the NX Meta Server deployment. Please refer to our report for further details.

### 3.1 ONNX Version

We found that the following settings allow the `ultralytics` ONNX export to work on our Windows machine. However, your configuration may differ depending on your environment, so adjustments might be necessary
1. Format: onnx
2. Opset: 12
3. Simplify: True
4. Dynamic: False
5. Batch: 1
6. Image Size: 640

### 3.2 Tracking Small Objects (Cigarettes)

- It is difficult to track the cigarette in CCTV footage because the object’s bounding box is small.
- Consequently, the Intersection over Union (IoU) between the previous and current appearances is close to zero, and occasionally exactly zero.
- We proposed a `non-parametric` way to solve this problem, which can also be `plug-and-play` regardless of the tracking algorithm.
- Just `enlarge` the bounding box of the cigarette(s) before passing it to the tracking algorithm. This greatly improves tracking performance in our tests. Afterwards, reduce the box size again for visualization.

<img src="assets/Challenges in Small Object Tracking.png" alt="Challenges in Small Object Tracking" width="600"/>

# Acknowledgement
1. [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
2. [Balanced Sampling using Weighted Dataloader](https://y-t-g.github.io/tutorials/yolo-class-balancing/)
