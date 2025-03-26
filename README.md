# Skin Disease Expert

This repository provides the **code and trained models** associated with the research study titled:

**"Identifying and Analyzing Consistently Misclassified Dermatoscopic Images by AI Models and Expert Dermatologists"**

## Overview

The integration of artificial intelligence (AI), particularly convolutional neural networks (CNNs), into dermatological diagnosis has demonstrated substantial potential in clinical practice. However, AI algorithms have not yet achieved widespread adoption due to persistent challenges and limitations.

### Objective
This study explores a critical yet underexplored issue: the consistent misclassification of specific dermatoscopic images by both AI models and expert dermatologists. We aim to investigate whether these consistent errors are attributable to the intrinsic complexity of the images or other factors, such as image quality.

### Key Findings
- Using comprehensive experimentation with multiple CNN architectures and cross-validation, we identified images that were consistently misclassified across different models.
- Statistical analysis confirmed that the common misclassifications were highly unlikely to be random.
- Expert dermatologists independently evaluated these problematic images and demonstrated a significantly lower diagnostic accuracy (31%) compared to 70% on control images correctly classified by CNNs.
- Image quality was identified as a potential contributing factor to these errors.
- Incorporating patient metadata and utilizing a meta-classifier improved classification performance, achieving:
  - **Balanced Accuracy**: 0.9003
  - **F1-score**: 0.8843

### Contribution
All data, code, and trained models associated with this study are made publicly available to promote transparency, reproducibility, and further research in this domain.

---

## Table of Contents
- [Dataset](#dataset)
- [Model Weights](#model-weights)
- [Usage](#usage)
---

## Dataset

The dataset used for training and evaluation consists of dermatoscopic images labeled according to various skin conditions is the publicy available ISIC 2019 dataset.

---

## Model Weights

Pre-trained model weights are available for download from the following Google Drive folder:

[Download Model Weights](https://drive.google.com/drive/folders/134wJNTck6hO1Jv_V3tEIAWNb5MqDUm8V)

Please place the downloaded weights in the appropriate directory as indicated in the project structure.

## Usage

This repository contains several Jupyter notebooks demonstrating different aspects of the workflow:

- **ImagePreprocessing.ipynb**: Techniques for identify the blurred images.

- **Training K Fold.ipynb**: Training models using K-Fold cross-validation to identify "difficult" images on the all dataset images.

- **Predict Image.ipynb**: Making predictions on new images using trained models.

- **Analyze Diagnosis.ipynb**: Analyzing the diagnosis made by human experts.

- **CreateDataframe.ipynb**: Creating dataframes from model predictions for meta-classification and to find "difficult" images.

- **FindDifficultImages.ipynb**: Identifying images that are challenging for the models.

- **MetaClassification.ipynb**: Combining predictions from multiple models for improved accuracy.
