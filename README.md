# Animal Image Recognition

A computer vision project that classifies animal images into five categories using a Convolutional Neural Network (CNN) built with PyTorch. The project also includes traditional machine learning models for comparative evaluation.

## Project Overview

This project implements an end-to-end image classification workflow, including dataset preparation, model training, evaluation, and image prediction through a graphical user interface.

The system recognizes:

- Butterfly
- Cat
- Dog
- Elephant
- Horse

## Features

- Image dataset preparation and organization
- CNN-based image classification using PyTorch
- Comparison with KNN and SVM models
- Model training and evaluation
- Graphical user interface for image selection
- Predicted class display
- Saving prediction results to a text file

## Technologies Used

- Python
- PyTorch
- Scikit-learn
- NumPy
- Pillow
- Tkinter
- Matplotlib
- Jupyter Notebook

## Machine Learning Models

### Convolutional Neural Network (CNN)
Used as the main deep learning model for image classification.

### K-Nearest Neighbors (KNN)
Implemented as a traditional machine learning baseline.

### Support Vector Machine (SVM)
Used to compare traditional classification performance with the CNN approach.

## Project Structure

```text
AnimalRecognitionProject/
├── dataset/
├── models/
├── results/
│   └── predictions.txt
├── ui/
│   ├── app.py
│   └── classifier.py
├── train_cnn.py
├── train_knn_svm.py
├── prepare_dataset.py
├── test.py
├── test_model.py
├── gpu_test.py
├── requirements.txt
└── README.md
