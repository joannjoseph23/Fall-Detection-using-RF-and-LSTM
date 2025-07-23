
# Fall Detection System for Elderly

## Overview
This project aims to detect falls in elderly individuals using wearable sensor data from the WEDA dataset. It utilizes advanced preprocessing, feature engineering, and a Long Short-Term Memory (LSTM) neural network to classify fall events accurately.

## Features

### ✅ Dataset
- **WEDA Dataset**: Preprocessed wearable sensor data containing fall and non-fall activities.

### ✅ Feature Engineering
- Extracted time and frequency domain features from raw signals.
- Applied multiple feature selection techniques (Extra Trees, RFE, L2-SVC).
- Combined top-ranked features for optimal model input.

### ✅ Model Architecture
- Built a deep learning model using **LSTM** in Keras.
- Balanced class weights to address class imbalance.
- Used dropout and L2 regularization to reduce overfitting.

### ✅ Evaluation
- Visualized training/validation loss over epochs.
- Evaluated model with confusion matrix and classification report.
- Achieved promising accuracy and recall on test data.

### ✅ Code Modules
- `featuresExtraction_WEDA.py`: Feature extraction from raw sensor CSVs.
- `main.py`: Full pipeline from loading data to LSTM training and evaluation.

## How to Run
1. Set up project directory with WEDA CSV files.
2. Run `featuresExtraction.py` to preprocess and extract features.
3. Execute `main.py` to train the LSTM model and evaluate performance.

## Repository
[Fall Detection GitHub Repo](https://github.com/joannjoseph23/Fall-Detection-using-RF-and-LSTM)

---

## Project Description (5 Key Points)

1. **LSTM-Based Classification**: Implements a sequential deep learning model for fall event detection from sensor data.
2. **Robust Feature Selection**: Uses Extra Trees, RFE, and L2-SVC to identify the most informative features.
3. **Class Imbalance Handling**: Calculates class weights to ensure balanced training and better fall recall.
4. **Modular Pipeline**: Clean separation between data processing, feature engineering, and model training code.
5. **Visual Feedback**: Includes loss plots and normalized confusion matrix to interpret training and predictions.

