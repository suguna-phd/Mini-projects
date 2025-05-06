``````````````````**Tomato Leaf Disease Detection using PCA and Logistic Regression**````````````````
This project demonstrates a simple machine learning pipeline for tomato leaf disease classification using Principal Component Analysis (PCA) for dimensionality reduction and Logistic Regression for classification. The dataset is organized in folders based on disease categories.

📂 Dataset Structure

The dataset must be organized like this:

dataset/
├── train/
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   └── ... (other classes)
├── val/
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   └── ... (other classes)

🔍 What This Project Does

Loads and flattens images from train and validation folders.
Standardizes the pixel data using StandardScaler.
Reduces dimensions using PCA to speed up training and avoid overfitting.
Trains a Logistic Regression classifier to predict disease classes.
Evaluates the model using precision, recall, and F1-score.

🧪 Key Features

✅ Works with labeled image datasets.
📉 Uses PCA to reduce noise and improve learning efficiency.
🔍 Classification across 10+ tomato leaf disease types.

🛠️ Technologies Used

Python
OpenCV
NumPy
Scikit-learn

📊 Sample Output

              precision    recall  f1-score   support

           0       0.87      0.83      0.85        52
           1       0.78      0.81      0.79        48
           ...
accuracy                           0.68       520

📁 How to Run

Clone the repo.
Set the correct paths for train_folder and val_folder.
Run the Python script.


Code:
import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Set paths to train and val directories
train_folder = '/content/drive/MyDrive/tomato/train'
val_folder = '/content/drive/MyDrive/tomato/val'

# Define label map exactly matching your folder names
label_map = {
    'Tomato___Early_blight': 0,
    'Tomato___Late_blight': 1,
    'Tomato___Septoria_leaf_spot': 2,
    'Tomato___Bacterial_spot': 3,
    'Tomato___Leaf_Mold': 4,
    'Tomato___Tomato_mosaic_virus': 5,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 6,
    'Tomato___Target_Spot': 7,
    'Tomato___Spider_mites_Two_spotted_spider_mite': 8,
    'Tomato___healthy': 9
}

# Function to load and flatten images
def load_images_from_folder(folder, img_size=(64, 64)):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path) and label in label_map:
            for file in os.listdir(label_path):
                img_path = os.path.join(label_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    images.append(img.flatten())
                    labels.append(label_map[label])
    return np.array(images), np.array(labels)

# Load data
X_train, y_train = load_images_from_folder(train_folder)
X_val, y_val = load_images_from_folder(val_folder)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Apply PCA
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_pca, y_train)

# Predict and evaluate
y_pred = clf.predict(X_val_pca)
print(classification_report(y_val, y_pred))
