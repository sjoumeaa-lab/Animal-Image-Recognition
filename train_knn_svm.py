import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib

dataset_path = "dataset"

X = []
y = []

labels = {
    "cat": 0,
    "dog": 1,
    "horse": 2,
    "elephant": 3,
    "butterfly": 4
}

print("Loading images...")

for animal in labels:
    folder_path = os.path.join(dataset_path, animal)

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        img = cv2.resize(img, (32, 32))
        img = img.flatten()

        X.append(img)
        y.append(labels[animal])

X = np.array(X)
y = np.array(y)
X = X[:3000]
y = y[:3000]

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training KNN...")

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

knn_predictions = knn.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)

print("KNN Accuracy:", knn_accuracy)

print("Training SVM...")

svm = LinearSVC(max_iter=5000)
svm.fit(X_train, y_train)

svm_predictions = svm.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)

print("SVM Accuracy:", svm_accuracy)

os.makedirs("models", exist_ok=True)

joblib.dump(knn, "models/knn_model.pkl")
joblib.dump(svm, "models/svm_model.pkl")

print("Models saved!")