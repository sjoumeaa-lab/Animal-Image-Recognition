# Animal Recognition System

## Project Description
This project is an Animal Recognition System developed using Python and PyTorch.

The system uses a Convolutional Neural Network (CNN) to classify animal images into five categories:

- Butterfly
- Cat
- Dog
- Elephant
- Horse

## Features

- Upload an image through a graphical user interface
- Display the selected image
- Predict the animal class
- Save prediction results to a text file

## Technologies Used

- Python
- PyTorch
- Tkinter
- Pillow

## Project Structure

AnimalRecognitionProject/
├── dataset/
├── models/
│   └── cnn_model.pth
├── results/
│   └── predictions.txt
├── ui/
│   ├── app.py
│   └── classifier.py
├── train_cnn.py
├── train_knn_svm.py
├── prepare_dataset.py
└── README.md

## How to Run

1. Open a terminal in the project folder.

2. Run:

python ui\app.py

3. Click "Select Image".

4. Choose an image.

5. The system will display the image and predict the animal.

## Output

Prediction results are stored in:

results/predictions.txt