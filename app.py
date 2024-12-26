from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os

# Load YOLO model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Process and display the detection results
def display_results(image, results):
    boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    scores = results.boxes.conf.cpu().numpy()  # Confidence scores
    labels = results.boxes.cls.cpu().numpy()  # Class indices
    names = results.names  # Class names
    
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = boxes[i].astype(int)
            label = names[int(labels[i])]
            score = scores[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Main Streamlit app
def main():
    st.title("Jungle Creep Mobile legends Detection with YOLO")

    model_path = "best.pt"  # Path to your YOLO model
    model = load_model(model_path)

    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        file_type = uploaded_file.type

        if file_type.startswith("image"):
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("Processing...")
            results = model.predict(image_np, imgsz=640)  # Perform detection
            result_image = display_results(image_np, results[0])
            st.image(result_image, caption="Processed Image", use_column_width=True)

        elif file_type.startswith("video"):
            temp_video_path = tempfile.NamedTemporaryFile(delete=False).name
            with open(temp_video_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            cap = cv2.VideoCapture(temp_video_path)
            temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = None

            st_frame = st.empty()  # Placeholder for video frames

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, imgsz=640)  # Perform detection
                result_frame = display_results(frame, results[0])

                if out is None:
                    height, width, _ = frame.shape
                    out = cv2.VideoWriter(temp_output_path, fourcc, 20.0, (width, height))

                out.write(result_frame)
                st_frame.image(result_frame, channels="RGB", use_column_width=True)

            cap.release()
            if out:
                out.release()

            st.video(temp_output_path)
            os.remove(temp_video_path)
            os.remove(temp_output_path)

if __name__ == "__main__":
    main()
