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

