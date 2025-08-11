# Create and train model

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2
IMAGE_SIZE = 160
BATCH_SIZE = 32
DATA_DIRECTORY = 'cat_breeds'
EPOCHS = 10

def load_data(data_dir):
    images, labels, class_names = [], [], sorted(os.listdir(data_dir))
    for label_index, breed in enumerate(class_names):
        cat_breeds_dir = os.path.join(data_dir, breed)
        for image_name in os.listdir(cat_breeds_dir):
            image_path = os.path.join(cat_breeds_dir, image_name)
            try:
                image = cv2.imread(image_path)
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                images.append(image)
                labels.append(label_index)
            except:
                continue
    return np.array(images), np.array(labels), class_names

def preprocess_image(image):
    return_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return return_image.astype('float32') / 255.0

def preprocess_images(images):
    return_images = images.astype("float32") / 255.0
    return return_images

def build_model(classes):
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                                    include_top=False,
                                                    weights='imagenet')
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_training_history(history, figsize=(10,4)):
    hist = history.history
    epochs = range(1, len(next(iter(hist.values()))) + 1)
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    if 'loss' in hist:
        plt.plot(epochs, hist['loss'], label='train loss', marker='o')
    if 'val_loss' in hist:
        plt.plot(epochs, hist['val_loss'], label='val loss', marker='o')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend()
    plt.subplot(1,2,2)
    acc_keys = [k for k in hist.keys() if 'acc' in k and not k.startswith('val_')]
    val_acc_keys = [k for k in hist.keys() if 'val_' in k and 'acc' in k]
    if acc_keys:
        for k in acc_keys:
            plt.plot(epochs, hist[k], label=k, marker='o')
    if val_acc_keys:
        for k in val_acc_keys:
            plt.plot(epochs, hist[val_acc_keys[0]], label=val_acc_keys[0], marker='o')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, cmap='Blues', figsize=(8,6)):
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    plt.show()
    return cm

def show_sample_predictions(model, x, y_true, class_names=None, n=12, figsize=(12,8), preprocess_fn=None):
    if preprocess_fn is not None:
        x_disp = np.array([preprocess_fn(img.copy()) for img in x])
    else:
        x_disp = x
    if x_disp.ndim == 4:
        pass
    else:
        raise ValueError("x should be an array of images with shape (N,H,W,C)")
    pred = model.predict(x_disp, verbose=0)
    if pred.ndim > 1:
        y_pred = np.argmax(pred, axis=1)
    else:
        y_pred = (pred > 0.5).astype(int).ravel()
    if y_true.ndim > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true
    n = min(n, len(x_disp))
    idx = np.random.choice(range(len(x_disp)), size=n, replace=False)
    cols = 4
    rows = (n + cols - 1) // cols
    plt.figure(figsize=figsize)
    for i, j in enumerate(idx):
        plt.subplot(rows, cols, i + 1)
        img = x_disp[j]
        if img.shape[-1] == 1:
            plt.imshow(img.squeeze(), cmap='gray')
        else:
            plt.imshow((img.astype('float32') - img.min()) / (img.max() - img.min() + 1e-12))
        true_label = class_names[y_true_labels[j]] if class_names else str(y_true_labels[j])
        pred_label = class_names[y_pred[j]] if class_names else str(y_pred[j])
        color = 'green' if y_pred[j] == y_true_labels[j] else 'red'
        plt.title(f"T:{true_label}\nP:{pred_label}", color=color, fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    print("Loading data")
    images, labels, class_names = load_data(DATA_DIRECTORY)
    images = preprocess_images(images)
    x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, stratify=labels)
    print(f"Classes found: {class_names}")
    print("Building model")
    model = build_model(classes=len(class_names))
    print("Training model")
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS)
    plot_training_history(history)
    pred = model.predict(x_train)
    plot_confusion_matrix(y_train, pred, class_names=class_names, normalize=True)
    show_sample_predictions(model, x_train[:200], y_train[:200], class_names=class_names, n=12, preprocess_fn=preprocess_image)
    model.save("cat_breed_model.keras")
    print("Model saved: 'cat_breed_model.keras'")

if __name__ == '__main__':
    main()