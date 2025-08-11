# User Interface

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import requests
import pymysql
conn = pymysql.connect(
                host="185.212.71.204",
                user="u951934835_admin",
                password="CatBook123",
                database="u951934835_catbook"
)
IMAGE_SIZE = 160
MODEL_PATH = 'cat_breed_model.keras'
CLASS_NAMES = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx']
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype("float32") / 255.0
    return image

def predict_image(image_path):
    image = preprocess_image(image_path)
    pred = model.predict(np.expand_dims(image, axis=0))[0]
    breed = CLASS_NAMES[np.argmax(pred)]
    confidence = np.max(pred) * 100
    return breed, confidence

def upload_and_predict():
    if entry.get() == "Enter your cat's name" or entry.get() == "":
        warning_label.pack(pady=10, padx=10)
        return
    else:
        warning_label.pack_forget()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    breed, confidence = predict_image(file_path)
    name = entry.get()
    name_label.config(text=f"{name}")
    name_label.pack(pady=10)
    image = Image.open(file_path)
    image = image.resize((300, 300))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo
    image_label.pack()
    url = "https://ctylo.us/upload.php"
    try:
        with open(file_path, 'rb') as file:
            response = requests.post(url, files={'image': file})
        if response.status_code == 200:
            print("Image uploaded successfully!")
            print("Server Response:", response.text)
            cursor = conn.cursor()
            sql = "INSERT INTO cats (name, breed, img, confidence) VALUES (%s, %s, %s, %s)"
            val = (entry.get(), breed, response.text, confidence)
            cursor.execute(sql, val)
            conn.commit()
            cursor.close()
            upload_button.pack_forget()
            entry.pack_forget()
            flag_button.pack(pady=10, padx=10)
            reset_button.pack(pady=10, padx=10)
        else:
            print("Failed to upload image. Status Code:", response.status_code)
    except Exception as e:
        print("An error occurred:", e)
    result_label.config(text=f"Prediction: {breed} ({confidence:.2f}%)")
    result_label.pack(pady=10)
    if confidence < 70:
        validate_breed()

def validate_breed():
    cursor = conn.cursor()
    sql = "SELECT * FROM cats ORDER BY id DESC LIMIT 1;"
    cursor.execute(sql)
    row = cursor.fetchone()
    val = None
    if row:
        id, name, breed, img, validate, confidence = row
        if validate == "true":
            val = ("false", id)
            flag_button.config(bg="green", text="Accurate Prediction")
        else:
            val = ("true", id)
            flag_button.config(bg="red", text="Inaccurate Prediction")
    else:
        print("No matching record found.")
    sql = "UPDATE cats SET validate = %s WHERE id = %s"
    cursor.execute(sql, val)
    conn.commit()
    cursor.close()

def reset():
    entry.pack(pady=10, padx=10)
    upload_button.pack(pady=10, padx=10)
    flag_button.pack_forget()
    reset_button.pack_forget()
    name_label.pack_forget()
    image_label.pack_forget()
    result_label.pack_forget()
    entry.delete(0, tk.END)
    entry.config(fg="black")

def on_entry_click(event):
    if entry.get() == default_text:
        entry.delete(0, tk.END)
        entry.config(fg="black")

def on_focus_out(event):
    if entry.get() == "":
        entry.insert(0, default_text)
        entry.config(fg="grey")

root = tk.Tk()
root.title("CatBook: Upload")
root.config(bg="light blue")
frame = tk.Frame(root, width=300, height=300)
frame.pack(padx=10, pady=10)
default_text = "Enter your cat's name"
entry = tk.Entry(frame, fg="grey", font=("Arial", 12))
entry.insert(0, default_text)
entry.bind("<FocusIn>", on_entry_click)
entry.bind("<FocusOut>", on_focus_out)
entry.pack(pady=10, padx=10)
upload_button = tk.Button(frame, text="Upload", font=("Arial", 12), command=upload_and_predict)
upload_button.pack(pady=10, padx=10)
warning_label = tk.Label(frame, text="Please enter your cat's name", fg="red")
name_label = tk.Label(frame, text="", font=("Arial", 16))
image_label = tk.Label(frame)
result_label = tk.Label(frame, text="", font=("Arial", 16))
flag_button = tk.Button(frame, text="Accurate Prediction", font=("Arial", 12), command=validate_breed)
flag_button.config(bg="green", fg="black")
reset_button = tk.Button(frame, text="Done", command=reset)
root.minsize(300, 300)
root.mainloop()
conn.close()