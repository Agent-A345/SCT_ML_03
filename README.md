# 🐾 Cats vs Dogs Classification using SVM

This project is part of Task 03 of the SkillCraft Technology internship program. It implements a Support Vector Machine (SVM) model to classify images of cats and dogs using pixel-level features extracted 
from images.

The classification model is trained on:
- Grayscale image data (resized to 64×64 pixels)
- Labeled images (filenames containing cat or dog)

The pipeline flattens images, splits them into training and validation sets, trains a Linear SVM, evaluates model performance (accuracy, confusion matrix, precision/recall), and generates predictions on the 
test set. Outputs include:
- Trained model (.joblib)
- Metrics report
- Submission file (.csv) in Kaggle format

## 📌 Technologies Used

- Python
- Pipenv
- Scikit-learn
- NumPy
- Pandas
- Pillow (PIL)
- Tqdm
- Joblib

## 📦 Installation
To run this project locally, follow the steps below:
1. **Clone the repository**:
```
git clone https://github.com/Agent-A345/SCT_ML_03.git
```
2. **Install Dependencies**
```
pip install scikit-learn numpy pandas pillow tqdm joblib
```
3. **Run the program**
```
python task3.py
```

## 💬 How It Works

1. Load training images from the `train/` folder where filenames include class labels (`cat` or `dog`).
2. Convert each image to grayscale, resize to 64×64 pixels, and flatten into a 1D feature vector.
3. Create labeled feature vectors (`X`, `y`) for all training images.
4. Split the data into training and validation sets (80/20 stratified split).
5. Train a Linear Support Vector Machine (SVM) on the training set.
6. Evaluate the model on the validation set using accuracy, precision, recall, F1-score, and confusion matrix.
7. Load unlabeled images from the `test1/` folder and generate predictions.
8. Save the trained model (`model.joblib`), evaluation metrics (`metrics.txt`), and submission file (`submission.csv`) to the `runs/simple_svm/` directory.

## 📂 Dataset Used – Kaggle

📎 [Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)  
Used for training and testing the image classification model.

## 🙌 Acknowledgements
Thanks to SkillCraft Technology for the opportunity to work on this internship project.

## License
This project is licensed under the MIT License.
