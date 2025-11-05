import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import cv2 # openCV for webcam

from train_minixception import model as model_minix, device
from train_scratch import model as model_scratch
from preprocessing import test_transform
from model_results import classes, LoadModel

# -- GLOBAL VARS -- #
current_model = model_minix  # default model
img_tk = None  # placeholder for displayed image
cap = None  # OpenCV VideoCapture object
webcam_on = False  # track webcam status

# load and set to evaluation mode
LoadModel()
model_minix.eval()
model_scratch.eval()

def predict_image(img):
    """Runs prediction on a PIL image."""
    img_tensor = test_transform(img).unsqueeze(0).to(device) # applying test transformations to image
    with torch.no_grad():
        output = current_model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    return classes[pred]

def load_image():
    """Loads an image from file explorer."""
    global img_tk
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).convert("RGB") # convert to RGB
        img_tk = ImageTk.PhotoImage(img.resize((250, 250))) # resize to 250x250
        image_label.config(image=img_tk)
        image_label.image = img_tk

        prediction = predict_image(img) # calling predict_image to get prediction
        result_label.config(text=f"Prediction: {prediction}")

def start_webcam():
    """Start webcam preview."""
    global cap, webcam_on
    if not webcam_on:
        cap = cv2.VideoCapture(0) # turn on webcam
        if not cap.isOpened():
            result_label.config(text="Error: Cannot access webcam.")
            return
        webcam_on = True
        update_frame() # call update frame

def stop_webcam():
    """Stop webcam preview."""
    global cap, webcam_on
    if webcam_on:
        webcam_on = False # turn off webcam
        if (cap != None):
            cap.release()
        image_label.config(image=None)
        result_label.config(text="Prediction: ")

def update_frame():
    """Continuously update webcam feed and predict in real-time."""
    global img_tk, cap, webcam_on
    if webcam_on and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # prediction 
            prediction = predict_image(img)
            result_label.config(text=f"Prediction: {prediction}")

            # display frame in tkinter
            img_tk = ImageTk.PhotoImage(img.resize((250, 250)))
            image_label.config(image=img_tk)
            image_label.image = img_tk

        # schedule the next frame update (30 FPS)
        root.after(30, update_frame)

def switch_model():
    """Switch between MiniXception and EmotionCNN models."""
    global current_model
    if model_var.get() == "MiniXception":
        current_model = model_minix
    else:
        current_model = model_scratch
    result_label.config(text=f"Model switched to {model_var.get()}")

# -- TKINTER GUI -- #
root = tk.Tk()
root.title("Emotion Recognition")

# model selection
model_var = tk.StringVar(value="MiniXception")
tk.Label(root, text="Select Model:").pack()
tk.Radiobutton(root, text="MiniXception", variable=model_var,
               value="MiniXception", command=switch_model).pack()
tk.Radiobutton(root, text="EmotionCNN (Scratch)", variable=model_var,
               value="Scratch", command=switch_model).pack()

# buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)
tk.Button(btn_frame, text="Upload Image", command=load_image).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Start Webcam", command=start_webcam).grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="Stop Webcam", command=stop_webcam).grid(row=0, column=3, padx=5)

# image + result display
image_label = tk.Label(root)
image_label.pack()
result_label = tk.Label(root, text="Prediction: ", font=("Arial", 12, "bold"))
result_label.pack(pady=10)

root.mainloop()