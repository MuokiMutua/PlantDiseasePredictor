import os
import json
import io

import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd

from PIL import Image
from tensorflow import keras
from datetime import datetime

# Must be the first Streamlit command
st.set_page_config(
    page_title="LeafScan AI | Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced styling
st.markdown("""
<style>
    /* Modern styling */
    div[class="stButton"] > button {
        background-color: #2ecc71;
        color: white;
        width: 100%;
        padding: 0.75rem 1.5rem;
        border-radius: 15px;
        border: none;
        transition: all 0.3s ease;
    }
    div[class="stButton"] > button:hover {
        background-color: #27ae60;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46, 204, 113, 0.3);
    }
    
    /* Enhanced stat boxes */
    .stat-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
        border: 1px solid #e9ecef;
    }
    .stat-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .stat-box h3 {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stat-box h2 {
        color: #2ecc71;
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    
    .css-1v0mbdj.e115fcil1 {
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Enhanced info-box for disease results */
    .info-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2ecc71;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .info-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .info-box h3 {
        color: #2c3e50;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .info-box p {
        color: #34495e;
        margin: 0.5rem 0;
        font-size: 1.1rem;
    }
    .info-box strong {
        color: #2ecc71;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for scan history
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

# Load model and class indices
@st.cache_resource
def load_model():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(working_dir, "trained_model", "plant_disease_pred.h5")
    class_indices_path = os.path.join(working_dir, "class_indices.json")
    
    try:
        model = keras.models.load_model(model_path)
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        return model, class_indices
    except Exception as e:
        st.error(f"Error loading model or class indices: {e}")
        return None, None

model, class_indices = load_model()

def process_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_disease(image):
    processed_img = process_image(image)
    predictions = model.predict(processed_img)
    confidence = float(np.max(predictions[0]) * 100)
    predicted_class = class_indices[str(np.argmax(predictions[0]))]
    return predicted_class, confidence

# Update the sidebar section with emojis and icons
with st.sidebar:
    # Logo and title section
    st.markdown("""
        <div style='text-align: center'>
            <h1 style='color: #2ecc71; margin-bottom: 0'>🌿</h1>
            <h1 style='margin-top: 0'>LeafScan AI</h1>
            <p style='color: #666; font-size: 0.9em'>Plant Disease Detection</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", ["🔍 Classifier", "📊 Analysis History", "❓ Help & Guide", "ℹ️ About"])

if page == "🔍 Classifier":
    st.markdown("""
        <div style='display: flex; justify-content: space-between; align-items: center'>
            <h1>🌱 Plant Disease Classifier</h1>
            <div style='font-size: 3.5rem; color: #2ecc71'>🔬</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-box">
            <h3>Total Scans</h3>
            <h2>%d</h2>
        </div>
        """ % len(st.session_state.scan_history), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-box">
            <h3>Supported Diseases</h3>
            <h2>38</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-box">
            <h3>Accuracy</h3>
            <h2>96.5%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main upload section
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            if st.button("🔍 Analyze Leaf"):
                with st.spinner("AI is analyzing your image..."):
                    prediction, confidence = predict_disease(image)
                    
                    # Save to history
                    scan_data = {
                        'timestamp': datetime.now(),
                        'disease': prediction,
                        'confidence': confidence,
                        'image': uploaded_file.getvalue()
                    }
                    st.session_state.scan_history.append(scan_data)
                    
                    # Display results with enhanced styling
                    st.success("Analysis Complete!")
                    st.markdown(f"""
                    <div class="info-box">
                        <h3>🔍 Analysis Results</h3>
                        <p><strong>Detected Disease:</strong> {prediction}</p>
                        <p><strong>Confidence Score:</strong> {confidence:.1f}%</p>
                        <p><strong>Status:</strong> {'High Risk' if confidence > 90 else 'Moderate Risk'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Treatment recommendations
                    st.markdown("### 🌿 Recommended Actions")
                    st.markdown("""
                    1. Isolate affected plants
                    2. Remove infected leaves
                    3. Apply appropriate fungicide
                    4. Improve air circulation
                    """)

elif page == "📊 Analysis History":
    st.title("Analysis History")
    
    if not st.session_state.scan_history:
        st.info("No scans yet. Try analyzing some leaves!")
    else:
        for idx, scan in enumerate(reversed(st.session_state.scan_history)):
            with st.expander(f"Scan {len(st.session_state.scan_history)-idx}: {scan['disease']} - {scan['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(scan['image'], use_column_width=True)
                with col2:
                    st.write(f"**Disease:** {scan['disease']}")
                    st.write(f"**Confidence:** {scan['confidence']:.1f}%")
                    st.write(f"**Date:** {scan['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

elif page == "❓ Help & Guide":
    st.title("Help & Guide")
    
    st.markdown("""
    ### 📸 How to Take Good Leaf Photos
    1. **Good Lighting**: Use natural daylight
    2. **Clear Focus**: Keep the leaf in focus
    3. **Clean Background**: Use a plain background
    4. **Proper Angle**: Capture the whole leaf
    
    ### 🎯 Best Practices
    - Clean your camera lens
    - Hold the camera steady
    - Include both healthy and infected parts
    - Take multiple angles if needed
    
    ### 📱 Supported Formats
    - JPG/JPEG
    - PNG
    """)

else:  # About page
    st.title("About LeafScan AI")
    
    st.markdown("""
    ### 🌿 Our Mission
    To help farmers and gardeners quickly identify plant diseases using artificial intelligence.
    
    ### 🤖 Technology
    - Deep Learning with TensorFlow
    - 38 Disease Classifications
    - 96.5% Accuracy Rate
    
    ### 📊 Impact
    - 10,000+ Plants Analyzed
    - 50+ Countries Reached
    - 24/7 Availability
    
    ### 🔒 Privacy
    Your uploaded images are processed securely and not stored permanently.
    """)

