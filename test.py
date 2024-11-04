from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import psutil
from pathlib import Path
import os
import time

def log_memory():
    #Log memory usage
    process = psutil.Process()
    print(f"{process.memory_info().rss / 1000 / 1000}mb")  # in mb 


PATH = Path("dataset") / "GTSRB"
PATH_TRAIN = PATH / "Training"

def load_dataset():
    train_directories = [PATH_TRAIN / subdir for subdir in os.listdir(PATH_TRAIN) if os.path.isdir(PATH_TRAIN / subdir)]
    train_directories = sorted(train_directories)
    #Test data is unlabelled??? -> Just split off training data for testing
    data = [[(cv2.cvtColor(cv2.imread(path / img), cv2.COLOR_BGR2RGB),path.name) for img in os.listdir(path) if img.endswith(".ppm")] for path in train_directories]
    X = [[img[0] for img in row] for row in data]
    y = [[int(img[1]) for img in row] for row in data]
    #Flatten 2d structure   
    X = [img for row in X for img in row]
    y = np.array([label for row in y for label in row])
    return (X,y)
X,y = load_dataset()
log_memory()
_N = 32
def transform_image(img):
    return tf.image.resize(img,[_N,_N]) / 255.0

X = list(map(transform_image,X))
#Convert to numpy array
X = np.array(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

input_size = _N * _N * 3
batch_size = 64
epochs = 30
n_classes = np.unique(y).shape[0]
epsilon = 0.3
alpha = 0.1
num_iter = 10

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
log_memory()
def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(_N,_N,3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(n_classes))

    print(model.summary())
    return model
model = get_model()
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)
print(history)
model.save("model.keras")
def print_metrics(model, x, y, c):
    # Get predicted probabilities for all classes
    y_pred_prob = model.predict(x)

    # Get predicted class labels (highest probability class)
    y_pred_class = np.argmax(y_pred_prob, axis=1)

    # Calculate precision, recall, and F1-score (using macro average)
    precision = precision_score(y, y_pred_class, average='macro')
    recall = recall_score(y, y_pred_class, average='macro')
    f1 = f1_score(y, y_pred_class, average='macro')

    # Display the macro/micro/weighted average metrics
    print(f'Precision (macro): {precision:.4f}')
    print(f'Recall (macro): {recall:.4f}')
    print(f'F1-score (macro): {f1:.4f}')

    # Binarize the output (needed for multiclass ROC)
    # This turns the class labels into a one-vs-rest binary format
    y_test_bin = label_binarize(y, classes=np.arange(c))

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(c):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_pred_prob[:, i])

    # Plot the ROC curve for each class
    plt.figure(figsize=(6, 5))
    for i in range(c):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) for Each Class')
    plt.legend(loc='lower right')
    plt.savefig(f"Performance{_N}.png")
    plt.show()
print_metrics(model, X_test, y_test, n_classes)
log_memory()