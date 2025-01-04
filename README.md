# Deep Learning Application for Change Detection

This project provides a deep learning-based solution for detecting changes in images taken from the same region at different times. The application includes a model developed using the U-Net architecture, optimized for use in ONNX format during training.

## Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Model Architecture and Customizations](#model-architecture-and-customizations)
- [Training and Evaluation](#training-and-evaluation)
- [Setup and Usage](#setup-and-usage)
- [References](#references)

---

## About the Project

Change detection is a method used in remote sensing to analyze spatial and temporal changes in a specific region. In this project, a deep learning model built on the U-Net architecture was utilized to perform change detection tasks.

## Dataset

The dataset used for the task consists of three main folders:
- **A**: 5042 JPEG images (256x256 pixels RGB) from the first time period.
- **B**: 5042 JPEG images (256x256 pixels RGB) from a later time period.
- **label**: 833 `.npz` files containing binary masks highlighting the changes between the two time periods.

The dataset was processed using the `preprocess.py` script to make it suitable for training.

### Processed Dataset

- **images_A**: 833 JPG images from the first time period.
- **images_B**: Corresponding images from the second time period.
- **masks**: Binary masks in JPG format highlighting changes, consisting of 833 images.

## Model Architecture and Customizations

The model used is based on U-Net architecture with the following features:

- **Input**: 256x256x6 (combined input for temporal analysis).
- **Output**: 256x256x1 (binary change map).
- **Hyperparameters**:
  - **batch_size**: 16
  - **num_epochs**: 50
  - **learning_rate**: 0.001
  - **optimizer**: Adam
  - **loss**: Combination of Binary Cross-Entropy (BCE) and Dice Loss.

## Training and Evaluation

The model was trained for 50 epochs on Google Colab. During training, metrics like Dice Loss and Binary Cross-Entropy Loss were used.

### Evaluation Metrics

- **Binary Cross-Entropy Loss**
- **Dice Loss**
- **Jaccard Index (IoU)**

## Setup and Usage

1. **Requirements**:
   - Python 3.8+
   - PyTorch
   - ONNX Runtime

2. **Project Directory Structure**:
   ```
   .
   ├── preprocess.py
   ├── train_Unet.py
   ├── convert_to_onnx.py
   ├── evaluate.py
   ├── main.py
   └── data/
       ├── images_A/
       ├── images_B/
       └── masks/
   ```

3. **Training**:
   ```bash
   python train_Unet.py
   ```

4. **Convert to ONNX Format**:
   ```bash
   python convert_to_onnx.py
   ```

5. **Evaluation**:
   - For the original model:
     ```bash
     python evaluate.py
     ```
   - For the ONNX model:
     ```bash
     python main.py
     ```


## References

1. Cheng, Guangliang, et al. "Change detection methods for remote sensing in the last decade: A comprehensive review." Remote Sensing, 2024.
2. Bai, T., et al. "Deep learning for change detection in remote sensing: a review." Geo-Spatial Information Science, 2022.
3. Perelus, Eleonora Jonasova. "A review of deep-learning methods for change detection in multispectral remote sensing images." Remote Sensing, 2023.
