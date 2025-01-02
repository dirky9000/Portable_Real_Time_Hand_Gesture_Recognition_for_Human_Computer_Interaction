import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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

# Optional: Check model predictions on the dataset (for confidence in visualization)
predictions = svm.predict(X)
print(f"Accuracy on the dataset: {np.mean(predictions == y):.2f}")

# Dimensionality Reduction using t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=2000)
X_tsne = tsne.fit_transform(X)

# Scatter Plot for Visualization
plt.figure(figsize=(10, 6))
classes = ["thumb_up", "thumb_down", "open_hand", "horns", "peace"]  # Adjust labels as needed
for label in np.unique(y):
    plt.scatter(
        X_tsne[y == label, 0],  # X-axis
        X_tsne[y == label, 1],  # Y-axis
        label=classes[label],  # Class label
        alpha=0.7  # Transparency
    )

# Add plot details
plt.title("SVM Cluster Visualization (t-SNE)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid()
plt.show()

tsne = TSNE(n_components=3, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for label in np.unique(y):
    ax.scatter(
        X_tsne[y == label, 0],
        X_tsne[y == label, 1],
        X_tsne[y == label, 2],
        label=classes[label],
        alpha=0.7
    )
plt.title("t-SNE Visualization (3D)")
plt.legend()
plt.show()
