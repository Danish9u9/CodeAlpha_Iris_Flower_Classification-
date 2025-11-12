# iris_knn.py
# Iris Flower Classification using K-Nearest Neighbors (KNN)
# Author: Muhammad Danish

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Load the dataset
data = pd.read_csv('iris.csv')
print("First 5 rows of the dataset:")
print(data.head())

# 2. Select features and target
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']

# 3. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 4. Create KNN classifier and train
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# 7. Display confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

# 8. Save the trained model
joblib.dump(model, 'iris_model.pkl')
print("\nTrained model saved as 'iris_model.pkl'")

# 9. Optional: Interactive prediction
print("\n--- Predict a new Iris flower species ---")
try:
    sepal_length = float(input("Sepal Length (cm): "))
    sepal_width = float(input("Sepal Width (cm): "))
    petal_length = float(input("Petal Length (cm): "))
    petal_width = float(input("Petal Width (cm): "))

    new_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(new_data)
    print(f"Predicted Species: {prediction[0]}")
except Exception as e:
    print("Prediction skipped. Enter valid numbers next time.")
