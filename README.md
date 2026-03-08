
# Hand Gesture Digit Recognition using MobileNetV2

## Project Overview

This project implements a **deep learning system for recognizing hand gesture digits (0–9)** using transfer learning with the **MobileNetV2 architecture**. The model classifies images of hand gestures representing numeric digits in sign language.

The system was developed as part of a deep learning research project focusing on **image classification, transfer learning, and experimental model improvements**.

The project includes the full machine learning pipeline:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Transfer learning using MobileNetV2
* Model training and fine-tuning
* Model evaluation and visualization
* Demo inference for predicting digits from new images

The final trained model achieves **~90% validation accuracy** on the hand gesture dataset.

---

# Team Members

* *Add your name here*
* *Add teammate names if applicable*

---

# Repository Structure

```
HandGestureRecognition/
│
├── README.md
│
├── src/
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── inference_demo.py
│
├── data/
│   ├── sample_images/
│   └── dataset_download_instructions.txt
│
├── models/
│   └── sign_language_model.h5
│
├── notebooks/
│   └── gesture_recognition_colab.ipynb
│
├── requirements.txt
│
└── results/
    ├── accuracy_plot.png
    ├── loss_plot.png
    └── confusion_matrix.png
```

### Folder Description

**src/**
Contains the core source code for training, evaluation, and inference.

**data/**
Contains sample images or instructions for downloading the dataset.

**models/**
Stores trained model weights and checkpoints.

**notebooks/**
Contains the main Colab notebook used during experimentation.

**results/**
Stores generated plots such as accuracy curves and confusion matrices.

---

# Dataset

The dataset used in this project is the **Sign Language Digit Dataset** from Kaggle.

It contains images representing hand gestures corresponding to digits **0–9**.

Each class represents a different digit in sign language.

Dataset characteristics:

* 10 classes (digits 0–9)
* RGB images
* Images resized to **128 × 128**
* Balanced dataset

### Download Dataset

You can download the dataset from Kaggle:

```
https://www.kaggle.com/datasets/kabilan03/sign-language-dataset
```

Or using Kaggle API:

```
kaggle datasets download kabilan03/sign-language-dataset
```

Place the dataset inside the `data/` folder.

---

# Environment Setup

This project was developed using **Python 3.10** and **TensorFlow 2.x**.

## Install dependencies

Create a virtual environment (recommended):

```
python -m venv venv
```

Activate environment:

Linux / Mac

```
source venv/bin/activate
```

Windows

```
venv\Scripts\activate
```

Install required packages:

```
pip install -r requirements.txt
```

---

# Requirements

Main libraries used in this project:

* TensorFlow
* NumPy
* Matplotlib
* Scikit-learn
* Pillow

Example `requirements.txt`:

```
tensorflow==2.15.0
numpy==1.24.4
matplotlib==3.7.2
scikit-learn==1.3.0
pillow==10.0.0
gradio==4.0.0
```

You can generate this automatically using:

```
pip freeze > requirements.txt
```

---

# Model Architecture

The model uses **transfer learning** with the **MobileNetV2** architecture.

Pipeline:

```
Input Image (128x128 RGB)
        ↓
Data Augmentation
        ↓
MobileNetV2 (pretrained on ImageNet)
        ↓
Global Average Pooling
        ↓
Dropout Layer
        ↓
Dense Softmax Layer (10 classes)
```

Transfer learning allows the model to reuse visual features learned from large datasets.

---

# Training the Model

To train the model run:

```
python src/train_model.py
```

Training steps:

1. Load dataset
2. Resize images
3. Normalize pixel values
4. Apply data augmentation
5. Train MobileNetV2 classifier
6. Save trained model

The model is saved in:

```
models/sign_language_model.h5
```

---

# Model Evaluation

The trained model is evaluated using several metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

To evaluate the model:

```
python src/evaluate_model.py
```

The evaluation generates:

* Accuracy curves
* Loss curves
* Confusion matrix
* Classification report

---

# Running Inference (Prediction Demo)

You can test the model on new images using:

```
python src/inference_demo.py
```

The system will:

1. Load a test image
2. Preprocess the image
3. Run the trained model
4. Output the predicted digit and confidence score

Example output:

```
Predicted Digit: 3
Confidence: 91.4%
```

---

# Experimental Improvements

Several experimental improvements were explored:

### Data Augmentation

* Random flip
* Random rotation
* Random zoom

These techniques improve model generalization.

### Transfer Learning

The MobileNetV2 base model was pretrained on **ImageNet**.

### Fine-Tuning

Additional experiments unfreezed deeper layers of the network to adapt high-level features to the gesture dataset.

---

# Results

Final model performance:

| Metric              | Value |
| ------------------- | ----- |
| Validation Accuracy | ~90%  |
| Precision           | ~0.91 |
| Recall              | ~0.90 |
| F1 Score            | ~0.90 |

The confusion matrix shows that most predictions fall along the diagonal, indicating correct classification for most gestures.

---

# Demo Interface (Optional)

A simple interactive interface can be launched using **Gradio** to allow users to upload images and receive predicted digits.

Run:

```
python src/inference_demo.py
```

The interface allows:

* image upload
* real-time prediction
* probability visualization

---

# References

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. (2018).
**MobileNetV2: Inverted Residuals and Linear Bottlenecks.**
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

Goodfellow, I., Bengio, Y., & Courville, A. (2016).
**Deep Learning. MIT Press.**

Shorten, C., & Khoshgoftaar, T. (2019).
**A survey on image data augmentation for deep learning.**

---

# Future Improvements

Potential improvements include:

* using larger datasets
* testing other architectures (EfficientNet, ResNet)
* implementing real-time webcam gesture recognition
* deploying the model on mobile devices using TensorFlow Lite

