import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import joblib

# Load the dataset
csv_file = "hand_gesture_landmarks.csv"  # Ensure this matches your dataset filename
df = pd.read_csv(csv_file)

# Separate features and labels
X = df.drop(columns=["label"]).values  # Features (landmarks)
y = df["label"].values  # Labels (classes)

# Load the trained SVM model
model_filename = "gesture_recognition_svm.pkl"
svm = joblib.load(model_filename)

# Predict labels using the trained SVM model
y_pred = svm.predict(X)

# Compute the confusion matrix
cm = confusion_matrix(y, y_pred)

# Define class names (adjust based on your dataset)
classes = ["thumb up", "thumb down", "open hand", "horns", "peace"]

# Create combined labels with numeric values
class_labels_with_numbers = [f"{name} ({i})" for i, name in enumerate(classes, start=1)]

# Plot the confusion matrix
plt.figure(figsize=(10, 8))  # Adjust figure size if needed
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels_with_numbers)
disp.plot(cmap="Blues", xticks_rotation=0, ax=plt.gca())  # Set rotation to 0 for upright labels

# Customize x-axis labels to include only numeric values
plt.gca().set_xticklabels([f"({i})" for i in range(1, len(classes) + 1)])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Adjust layout to prevent labels from being cut off
plt.tight_layout()
plt.show()
