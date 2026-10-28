import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="Smoke and Fire Classification", layout="centered")
st.title("—— Smoke & Fire Classification App ——")
st.write("Upload an MP4 video below to run classification using our YOLO classification model.")

# Input: Path to classification model weights
model_path = st.text_input("Model Weights Path", "/content/runs/classify/train/weights/best.pt")

# Load the YOLO Classification Model
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return YOLO(path)
    else:
        st.error(f"Model file not found at: {path}")
        return None

model = load_model(model_path)

# Video File Uploader
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mpeg"])

if uploaded_file is not None and model is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()

    # Open video using OpenCV
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.info(f"Video Loaded! Resolution: {width}x{height} | FPS: {fps} | Total Frames: {total_frames}")

    # Setup a temporary file path for the output video
    output_temp_path = os.path.join(tempfile.gettempdir(), "output_processed.mp4")

    # Use MP4V codec for output video writing
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_temp_path, fourcc, fps, (width, height))

    # Setup UI Progress Bar and Status Text
    progress_bar = st.progress(0)
    status_text = st.empty()
    preview_placeholder = st.empty()

    if st.button("Start Processing"):
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO classification prediction
            results = model(frame, verbose=False)

            # Apply classification confidence filter > 60%
            for result in results:
                if result.probs is not None:
                    top1_idx = result.probs.top1
                    confidence = float(result.probs.top1conf)

                    if confidence > 0.60:
                        class_name = result.names[top1_idx]
                        label = f"{class_name}: {confidence:.2f}"

                        # Overlay classification text at the top left of the frame
                        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

            # Write processed frame
            out.write(frame)

            # Update status metrics
            frame_count += 1
            progress_percent = int((frame_count / total_frames) * 100)
            progress_bar.progress(progress_percent)
            status_text.text(f"Processing Frame {frame_count}/{total_frames} ({progress_percent}%)")

            # Show live preview every 30 frames to save memory
            if frame_count % 30 == 0:
                preview_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        cap.release()
        out.release()

        status_text.text("Processing completed successfully!")
        st.success("Your output video is ready!")

        # Provide Download Button for processed video
        with open(output_temp_path, "rb") as file:
            st.download_button(
                label="Download Classified Video",
                data=file,
                file_name="classified_smoke_fire.mp4",
                mime="video/mp4"
            )
        
        # Clean up temporary file
        try:
            os.unlink(tfile.name)
        except Exception:
            pass
