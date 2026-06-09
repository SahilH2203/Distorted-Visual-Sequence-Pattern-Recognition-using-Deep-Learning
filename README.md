# Distorted Visual Sequence Pattern Recognition using Deep Learning

## Overview

This project focuses on recognizing and reconstructing text sequences from heavily distorted grayscale images using deep learning. The images contain alphanumeric character sequences that are affected by various visual degradations such as noise, occlusion, overlap, blur, deformation, and irregular character placement. Traditional OCR systems often struggle under these conditions because they rely on clear character boundaries and clean visual patterns.

The objective of this project is to develop an end-to-end deep learning framework capable of accurately extracting the underlying character sequence directly from distorted images without requiring explicit character segmentation.

---

## Project Objective

The primary goal is to build a robust Optical Character Recognition (OCR) system that can:

* Identify uppercase letters and digits from noisy grayscale images.
* Handle overlapping and partially occluded characters.
* Learn sequential relationships between characters.
* Generalize to unseen distortions and visual artifacts.
* Minimize Character Error Rate (CER), the primary evaluation metric.

The task can be viewed as a combination of image recognition and sequence prediction, where the model must convert an input image into an ordered sequence of characters.

---

## Methodology

### 1. Data Preparation

The dataset consists of grayscale images paired with corresponding text labels. Each label represents an alphanumeric sequence, typically six characters long.

To prepare the data for training:

* Images are loaded in grayscale format.
* Character labels are converted into numerical representations using a custom vocabulary.
* Labels are encoded for Connectionist Temporal Classification (CTC) training.
* Data is split into training and validation subsets for performance monitoring.

---

### 2. Image Enhancement and Preprocessing

Since the images contain significant visual corruption, preprocessing is applied to improve text visibility while preserving important stroke information.

The preprocessing pipeline includes:

* **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to improve local contrast.
* Intensity normalization for stable neural network training.
* Conversion into normalized tensors suitable for PyTorch models.

The preprocessing strategy is intentionally conservative to avoid removing subtle character details that may be important for recognition.

---

### 3. Data Augmentation

To improve robustness and reduce overfitting, a comprehensive augmentation pipeline is applied during training.

The augmentations include:

* Random rotation
* Translation and scaling
* Gaussian noise injection
* Motion blur
* Gaussian blur
* Random brightness and contrast adjustment
* Random occlusion using cutout/coarse dropout
* CLAHE-based contrast variations

These augmentations simulate the types of distortions present in the dataset and encourage the model to learn invariant and generalized visual representations.

---

### 4. CRNN Architecture

The core recognition system is based on a **Convolutional Recurrent Neural Network (CRNN)** architecture.

The model combines the strengths of convolutional networks for visual feature extraction and recurrent networks for sequence modeling.

#### CNN Feature Extractor

A deep convolutional backbone is used to:

* Extract discriminative visual features.
* Suppress background noise.
* Capture character-level patterns.

Progressive pooling operations reduce image height while preserving horizontal information. The final feature map is transformed into a sequence representation where each column corresponds to a temporal step.

#### Bidirectional LSTM Sequence Model

The extracted feature sequence is processed using a two-layer Bidirectional LSTM.

The BiLSTM:

* Models dependencies between neighboring characters.
* Uses both past and future context.
* Helps resolve ambiguities caused by distortion or occlusion.

#### Character Classification Layer

A fully connected layer maps sequence features into character probabilities over:

* 26 uppercase letters
* 10 digits
* 1 CTC blank token

This produces a probability distribution for every sequence position.

---

### 5. Connectionist Temporal Classification (CTC)

Character segmentation is not explicitly available in the dataset. Instead, the model is trained using **CTC Loss**.

CTC enables the network to:

* Learn alignments automatically.
* Handle variable spacing between characters.
* Predict complete sequences without character-level annotations.

During inference, CTC decoding removes repeated predictions and blank tokens to reconstruct the final text sequence.

---

## Architecture Diagram

```text
                    INPUT IMAGE
                  (1 × 100 × 200)
                           │
                           ▼
            ┌──────────────────────────┐
            │     Preprocessing        │
            │ • CLAHE Enhancement      │
            │ • Normalization          │
            └──────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────┐
            │     Data Augmentation    │
            │ • Rotation               │
            │ • Translation            │
            │ • Scaling                │
            │ • Noise Injection        │
            │ • Blur                   │
            │ • Cutout / Occlusion     │
            └──────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │       CNN Feature Extractor      │
        │                                  │
        │ Conv(1→64)  + BN + ReLU          │
        │ MaxPool                          │
        │ Conv(64→128) + BN + ReLU         │
        │ MaxPool                          │
        │ Conv(128→256) + BN + ReLU        │
        │ Height Pooling                   │
        │ Conv(256→512) + BN + ReLU        │
        │ Height Pooling                   │
        │ Conv(512→512) + BN + ReLU        │
        │ AdaptiveAvgPool(H=1)            │
        └──────────────────────────────────┘
                           │
                           ▼
              Feature Map Output
                 (512 × 1 × 50)
                           │
                           ▼
                Sequence Conversion
                 (50 × 512 Features)
                           │
                           ▼
        ┌──────────────────────────────────┐
        │      Bidirectional LSTM          │
        │                                  │
        │ Layers      : 2                  │
        │ Hidden Size : 256                │
        │ Direction   : Bidirectional      │
        └──────────────────────────────────┘
                           │
                           ▼
                Sequence Features
                    (50 × 512)
                           │
                           ▼
        ┌──────────────────────────────────┐
        │     Fully Connected Layer        │
        │                                  │
        │ 512 → 37 Classes                 │
        │                                  │
        │ Classes:                         │
        │ • A–Z (26)                       │
        │ • 0–9 (10)                       │
        │ • CTC Blank (1)                  │
        └──────────────────────────────────┘
                           │
                           ▼
                Character Probabilities
                     (50 × 37)
                           │
                           ▼
                  CTC Decoding
          (Collapse Repetitions & Blanks)
                           │
                           ▼
               Predicted Text Sequence

                    Example:
                 "XQ8NE2"
```


### 6. Training Strategy

The model is trained using:

* AdamW optimizer
* Weight decay regularization
* Mixed precision training (AMP)
* Gradient clipping
* Learning rate scheduling based on validation performance

Training performance is monitored using validation loss and Character Error Rate.

The best-performing model checkpoint is automatically saved based on the lowest validation CER.

---

### 7. Evaluation

Model performance is evaluated using **Character Error Rate (CER)**, which measures the number of insertions, deletions, and substitutions required to transform a predicted sequence into the correct target sequence.

A lower CER indicates better recognition performance.

In addition to CER, sequence-level accuracy is also monitored to measure the percentage of completely correct predictions.

---

## Architecture Summary

Input Image

↓

Image Preprocessing & Augmentation

↓

CNN Feature Extractor

↓

Sequence Transformation

↓

2-Layer Bidirectional LSTM

↓

Linear Classification Layer

↓

CTC Decoding

↓

Predicted Character Sequence

---

## Key Features

* End-to-end sequence recognition pipeline.
* Robust handling of noisy and distorted text images.
* Character segmentation-free training using CTC.
* Strong generalization through extensive augmentation.
* CNN + BiLSTM architecture optimized for OCR sequence learning.
* Validation based on Character Error Rate for reliable model selection.

---

## Outcome

The final system successfully learns to reconstruct alphanumeric sequences from highly distorted images by combining convolutional feature extraction, recurrent sequence modeling, and CTC-based decoding. The approach provides a practical and effective solution for challenging OCR scenarios involving noise, overlap, blur, deformation, and partial occlusion.
