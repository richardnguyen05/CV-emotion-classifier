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
face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_alt.xml") # for face detection

# load and set to evaluation mode
LoadModel()
model_minix.eval()
model_scratch.eval()

def predict_image(img):
    """Runs prediction on a PIL image and returns (label, confidence)."""
    img_tensor = test_transform(img).unsqueeze(0).to(device) # applying test transformations
    with torch.no_grad():
        output = current_model(img_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return classes[pred.item()], conf.item()

def load_image():
    """Loads an image from file explorer."""
    global img_tk
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).convert("RGB") # convert to RGB
        img_tk = ImageTk.PhotoImage(img.resize((250, 250))) # resize to 250x250
        image_label.config(image=img_tk)
        image_label.image = img_tk

        label, conf = predict_image(img)
        if show_conf_var.get():
            result_label.config(text=f"Prediction: {label} ({conf*100:.1f}%)")
        else:
            result_label.config(text=f"Prediction: {label}")

def start_webcam():
    """Start webcam preview."""
    global cap, webcam_on
    if not webcam_on:
        cap = cv2.VideoCapture(0) # turn on webcam
        if not cap.isOpened():
            result_label.config(text="Error: Cannot access webcam.")
            return
        webcam_on = True
        result_label.config(text="Webcam Running...")
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
    """Continuously update webcam feed and predict in real-time with bounding boxes."""
    global img_tk, cap, webcam_on
    if webcam_on and cap.isOpened():
        ret, frame = cap.read() # read frame
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # to grayscale
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces: # create bounding box
                face_img = frame_rgb[y:y+h, x:x+w]
                face_pil = Image.fromarray(face_img) # convert the image in bounding box to PIL

                label, conf = predict_image(face_pil) # obtain prediction

                # build label text
                if show_conf_var.get():
                    label_text = f"{label} ({conf*100:.1f}%)"
                else:
                    label_text = label


                # draw bounding box
                cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame_rgb, label_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # display frame in Tkinter
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(img.resize((250, 250)))
            image_label.config(image=img_tk)
            image_label.image = img_tk

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
root.geometry("600x500") # gui size

# model selection
model_var = tk.StringVar(value="MiniXception")
tk.Label(root, text="Select Model:").pack()
tk.Radiobutton(root, text="MiniXception", variable=model_var,
               value="MiniXception", command=switch_model).pack()
tk.Radiobutton(root, text="EmotionCNN (Scratch)", variable=model_var,
               value="Scratch", command=switch_model).pack()

# confidence display toggle
show_conf_var = tk.BooleanVar(value=True)
tk.Checkbutton(root, text="Show confidence score", variable=show_conf_var).pack(pady=5)

# buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)
tk.Button(btn_frame, text="Upload Image", command=load_image).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Start Webcam", command=start_webcam).grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="Stop Webcam", command=stop_webcam).grid(row=0, column=3, padx=5)

# image + result display
image_label = tk.Label(root)
image_label.pack()
# only display if image is uploaded
result_label = tk.Label(root, text="Prediction: ", font=("Arial", 12, "bold"))
result_label.pack(pady=10)

root.mainloop()