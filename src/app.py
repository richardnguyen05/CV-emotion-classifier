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

def capture_image():
    """Opens webcam, captures one frame, and runs prediction."""
    global img_tk
    cap = cv2.VideoCapture(0) # opens webcam
    if not cap.isOpened():
        result_label.config(text="Error: Cannot access webcam.")
        return

    ret, frame = cap.read()
    cap.release()

    if ret:
        # convert from BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        img_tk = ImageTk.PhotoImage(img.resize((250, 250)))
        image_label.config(image=img_tk)
        image_label.image = img_tk

        prediction = predict_image(img)
        result_label.config(text=f"Prediction: {prediction}")
    else:
        result_label.config(text="Error: Failed to capture image.")

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

# model selection radio buttons
model_var = tk.StringVar(value="MiniXception")
tk.Label(root, text="Select Model:").pack()

tk.Radiobutton(root, text="MiniXception", variable=model_var,
               value="MiniXception", command=switch_model).pack()
tk.Radiobutton(root, text="EmotionCNN (Scratch)", variable=model_var,
               value="Scratch", command=switch_model).pack()

# buttons for input method
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Upload Image", command=load_image).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Use Webcam", command=capture_image).grid(row=0, column=1, padx=5)

# display image and result
image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="Prediction: ", font=("Arial", 12, "bold"))
result_label.pack(pady=10)

root.mainloop()