import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from predict import picture  
from frcnn import FRCNN

# load the YOLO model
model_path1 = './models/yolo_best.pt'
model1 = YOLO(model_path1)

# title and description
st.title("Car Detection App")
st.markdown("""
    This is a car detection app that uses Faster R-CNN and YOLO to detect cars in images.    
""")

# upload image
uploaded_file = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])

def process_image_yolo(image, model):
    # Use the YOLO model to detect objects in the image
    results = model(np.array(image))
    detections = []
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    # process the detection results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = int(box.cls[0])
            confidence = box.conf[0]
            if label == 0:
                label_text = f"car {confidence:.2f}"
            else:
                label_text = f"unknown {confidence:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, y1 - 10), label_text, fill="green", font=font)
            detections.append({
                "Label": "car" if label == 0 else "unknown",
                "Confidence": confidence,
                "Bounding Box": f"[{x1}, {y1}, {x2}, {y2}]"
            })
    
    return image, detections

def process_image_fastrcnn(image):
    r_image, all_labels, all_confs, all_boxes = picture(image)
    detections2 = []
    draw = ImageDraw.Draw(r_image)
    font = ImageFont.load_default()
    for label, conf, box in zip(all_labels, all_confs, all_boxes):
        x1, y1, x2, y2 = map(int, box)
        label_text = f"car {conf:.2f}" if label == 0 else f"unknown {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        draw.text((x1, y1 - 10), label_text, fill="green", font=font)
        detections2.append({
            "Label": "car",
            "Confidence": conf,
            "Bounding Box": f"[{x1}, {y1}, {x2}, {y2}]"
        })
    return r_image, detections2

if uploaded_file is not None:
    # Transform the uploaded file into a PIL image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Process and display images for both models
    image1, detections1 = process_image_yolo(image.copy(), model1)
    image2, detections2 = process_image_fastrcnn(image.copy())
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1, caption='Predicted Image by YOLO', use_container_width=True)
    with col2:
        st.image(image2, caption='Predicted Image by Faster R-CNN', use_container_width=True)
    
    st.markdown("### YOLO Detections")
    df1 = pd.DataFrame(detections1)
    st.dataframe(df1)
    
    st.markdown("### Fast R-CNN Detections")
    df2 = pd.DataFrame(detections2)
    st.dataframe(df2)
    
    st.markdown("### Comparison of YOLO and Fast R-CNN Detections")
    comparison_df = pd.concat([df1, df2], keys=["YOLO", "Fast R-CNN"]).reset_index(level=0).rename(columns={"level_0": "Model"})
    st.dataframe(comparison_df)