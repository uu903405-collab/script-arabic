"""
Arabic script Streamlit App
Run with: streamlit run app.py
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Arabic Script Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .detection-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# CONFIGURATION
MODEL_PATH = "best.pt"  # Model file should be in the same directory as main.py


# LOAD MODEL (with caching)
@st.cache_resource
def load_model(model_path):
    """Load YOLOv8 model (cached for performance)"""
    try:
        if not os.path.exists(model_path):
            st.error(f"model file '{model_path}' not found in working directory!")
            st.info("Please place your 'best.pt' model file in the same directory as this app.")
            return None
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"error loading model: {e}")
        return None


# PREDICTION FUNCTION
def predict_image(model, image, conf_threshold, iou_threshold):
    """Run prediction on image"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Run prediction
    results = model.predict(
        source=img_array,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    return results[0]


# DRAW PREDICTIONS
def draw_predictions(image, result, show_conf=True, show_labels=True):
    """Draw bounding boxes on image"""
    img_array = np.array(image)
    
    # Get predictions
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    
    # Define colors for different classes
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128), (255, 165, 0)
    ]
    
    # Draw each detection
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = colors[cls_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 3)
        
        # Prepare label
        if show_labels:
            class_name = result.names[cls_id]
            if show_conf:
                label = f'{class_name}: {conf:.2f}'
            else:
                label = class_name
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Draw background rectangle
            cv2.rectangle(
                img_array,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                img_array,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
    
    return Image.fromarray(img_array)


# MAIN APP
def main():
    # Header
    st.markdown('<h1 class="main-header">Arabic Script Detection</h1>', unsafe_allow_html=True)
    
    # Load model
    model = load_model(MODEL_PATH)
    
    if model is None:
        st.stop()
    
    # SIDEBAR - Settings
    st.sidebar.header("⚙️ Settings")
    
    # Display model information
    st.sidebar.success("model loaded successfully!")
    with st.sidebar.expander("model Information"):
        st.write(f"**Model:** {MODEL_PATH}")
        st.write(f"**Type:** {model.model_name}")
        st.write(f"**Classes ({len(model.names)}):**")
        for i, name in model.names.items():
            st.write(f"  • {name}")
    
    # Detection parameters
    st.sidebar.header("🎚️ Detection Parameters")
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence for detection"
    )
    
    iou_threshold = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="IoU threshold for Non-Maximum Suppression"
    )
    
    # Display options
    st.sidebar.header("🎨 Display Options")
    show_conf = st.sidebar.checkbox("Show Confidence Scores", value=True)
    show_labels = st.sidebar.checkbox("Show Class Labels", value=True)
    

    # MAIN AREA - Image Source Selection    
    st.subheader("📸 Select Image Source")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Take Photo"])
    
    image = None
    
    # TAB 1: Upload Image
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Upload an image for object detection"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.success("image uploaded successfully!")
    
    # TAB 2: Camera Input
    with tab2:
        camera_photo = st.camera_input("Take a photo")
        
        if camera_photo is not None:
            image = Image.open(camera_photo)
            st.success("photo captured successfully!")
    
    # PROCESS IMAGE IF AVAILABLE
    if image is not None:
        # Create columns for before/after
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("original Image")
            st.image(image, use_container_width=True)
            st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
        
        # Run detection automatically or with button
        run_detection = st.button("run Detection", type="primary", use_container_width=True)
        
        if run_detection:
            with st.spinner("Detecting objects..."):
                # Run prediction
                result = predict_image(model, image, conf_threshold, iou_threshold)
                
                # Get detection info
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Draw predictions
                annotated_image = draw_predictions(image, result, show_conf, show_labels)
                
                with col2:
                    st.subheader("detection Results")
                    st.image(annotated_image, use_container_width=True)
                    st.caption(f"Detected {len(boxes)} object(s)")
                
                # Display detection statistics
                st.markdown("---")
                st.subheader("Detection Statistics")
                
                if len(boxes) > 0:
                    # Summary metrics
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Total Detections", len(boxes))
                    with metric_cols[1]:
                        st.metric("Avg Confidence", f"{np.mean(confidences):.2%}")
                    with metric_cols[2]:
                        st.metric("Max Confidence", f"{np.max(confidences):.2%}")
                    with metric_cols[3]:
                        st.metric("Min Confidence", f"{np.min(confidences):.2%}")
                    
                    # Class distribution
                    st.subheader("Class Distribution")
                    class_counts = {}
                    for cls_id in class_ids:
                        class_name = model.names[cls_id]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Display as columns
                    dist_cols = st.columns(min(len(class_counts), 4))
                    for idx, (class_name, count) in enumerate(class_counts.items()):
                        with dist_cols[idx % 4]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{count}</h3>
                                <p>{class_name}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Detailed detection info
                    with st.expander("Detailed Detection Information"):
                        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                            col_a, col_b = st.columns([1, 3])
                            with col_a:
                                st.write(f"**Detection #{i+1}**")
                            with col_b:
                                st.write(f"**Class:** {model.names[cls_id]} | "
                                       f"**Confidence:** {conf:.2%} | "
                                       f"**Box:** [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}] | "
                                       f"**Size:** {int(box[2]-box[0])}×{int(box[3]-box[1])}px")
                    
                    # Download section
                    st.markdown("---")
                    st.subheader("Download Results")
                    
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    
                    # Download annotated image
                    with col_dl1:
                        import io
                        buf = io.BytesIO()
                        annotated_image.save(buf, format='PNG')
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="📥 Download Image",
                            data=byte_im,
                            file_name="detected_image.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    # Download detection data as CSV
                    with col_dl2:
                        import pandas as pd
                        
                        detection_data = []
                        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                            detection_data.append({
                                'Detection': i+1,
                                'Class': model.names[cls_id],
                                'Confidence': f"{conf:.4f}",
                                'X1': int(box[0]),
                                'Y1': int(box[1]),
                                'X2': int(box[2]),
                                'Y2': int(box[3]),
                                'Width': int(box[2]-box[0]),
                                'Height': int(box[3]-box[1])
                            })
                        
                        df = pd.DataFrame(detection_data)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="📊 Download CSV",
                            data=csv,
                            file_name="detections.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    # Download detection summary
                    with col_dl3:
                        summary = f"""Detection Summary
Total Detections: {len(boxes)}
Average Confidence: {np.mean(confidences):.2%}
Max Confidence: {np.max(confidences):.2%}
Min Confidence: {np.min(confidences):.2%}

Class Distribution:
{chr(10).join([f'  - {name}: {count}' for name, count in class_counts.items()])}

Detection Details:
{chr(10).join([f'  #{i+1}: {model.names[cls_id]} ({conf:.2%}) at [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]' 
               for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids))])}
"""
                        st.download_button(
                            label="Download Report",
                            data=summary,
                            file_name="detection_report.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                
                else:
                    st.warning("No objects detected!")
                    st.info(f"Try lowering the confidence threshold (current: {conf_threshold})")
    
    else:
        st.info("Please upload an image or take a photo to start detection")
    
    # FOOTER
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; padding: 1rem;'>"
        "Built with hard work"
        "</div>",
        unsafe_allow_html=True
    )


# RUN APP
if __name__ == "__main__":
    main()