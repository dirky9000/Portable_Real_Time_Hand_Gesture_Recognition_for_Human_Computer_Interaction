import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
csv_file = "hand_gesture_landmarks.csv"
df = pd.read_csv(csv_file)

# Separate features and labels
X = df.drop(columns=["label"])  # Features (landmarks)
y = df["label"]  # Labels (classes)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
svm = SVC(kernel="linear", probability=True)
svm.fit(X_train, y_train)

# Evaluate the model
y_pred = svm.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the trained model
model_filename = "poop_monster.pkl"
joblib.dump(svm, model_filename)
print(f"Model saved to {model_filename}")
