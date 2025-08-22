import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import numpy as np
import cv2
import pydicom
from PIL import Image
from streamlit_paste_button import paste_image_button

# --- 1. CONFIGURATION ---
IMG_SIZE = 224
MODEL_PATH = 'best_pneumonia_model.h5'
LAST_CONV_LAYER_NAME = 'conv5_block16_2_conv'  # Specific to DenseNet121

st.set_page_config(layout="wide", page_title="Pneumonia Detection from Chest X-Rays")

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_app_model(model_path):
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 3. GRAD-CAM FUNCTIONS ---
def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_gradcam_overlay(img_3c, model, last_conv_layer_name):
    img_array = np.expand_dims(img_3c, axis=0) / 255.0
    pred = model.predict(img_array, verbose=0)[0][0]
    label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
    confidence = pred if pred > 0.5 else 1 - pred
    heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name)
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img_3c
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return label, confidence, superimposed_img

def analyze_and_display(image_pil, image_name, model):
    st.header(f"Analysis for: `{image_name}`")
    img = np.array(image_pil)
    
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        original_img_3c = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2RGB)
    else:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        original_img_3c = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    resized_img_3c = cv2.resize(original_img_3c, (IMG_SIZE, IMG_SIZE))

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img_3c, caption='Original X-Ray', use_container_width=True)
    with col2:
        with st.spinner(f'🤖 AI is analyzing {image_name}...'):
            label, confidence, gradcam_img = generate_gradcam_overlay(resized_img_3c, model, LAST_CONV_LAYER_NAME)
            st.image(gradcam_img, caption='Grad-CAM: AI Focus Heatmap', use_container_width=True)

    if label == "PNEUMONIA":
        st.error(f"**Diagnosis: {label}**")
    else:
        st.success(f"**Diagnosis: {label}**")
    st.info(f"**Confidence Score: {confidence:.2%}**")
    st.markdown("---")

# --- 4. SESSION STATE SETUP ---
if "images" not in st.session_state:
    st.session_state["images"] = []

def remove_image(index):
    if 0 <= index < len(st.session_state["images"]):
        del st.session_state["images"][index]

# --- 5. UI HEADER ---
model = load_app_model(MODEL_PATH)
st.title("🩺 AI-Powered Pneumonia Detection")
st.markdown("Upload or paste a chest X-ray for AI analysis with Grad-CAM visualization.")

if model:
    st.markdown("---")
    uploaded_files = st.file_uploader(
        "**Upload or Drag & Drop X-Ray Files**",
        type=["dcm", "jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    st.markdown("<h4 style='text-align: center; color: grey;'>OR</h4>", unsafe_allow_html=True)
    _, col_center, _ = st.columns([2, 1, 2])
    with col_center:
        paste_result = paste_image_button(
            "📋 Paste Image from Clipboard",
            text_color="#FFFFFF",
            background_color="#FF4B4B",
            hover_background_color="#FF6B6B"
        )
    st.markdown("---")

    # --- 6. ADD IMAGES TO SESSION STATE ---
    if paste_result.image_data is not None:
        st.session_state["images"].append(("Pasted Image", paste_result.image_data))
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.name.lower().endswith('.dcm'):
                    dcm = pydicom.dcmread(uploaded_file)
                    image_pil = Image.fromarray(dcm.pixel_array)
                else:
                    image_pil = Image.open(uploaded_file)
                st.session_state["images"].append((uploaded_file.name, image_pil))
            except Exception as e:
                st.error(f"Failed to load file {uploaded_file.name}: {e}")

    # --- Disclaimer ---
    if st.session_state["images"]:
        st.markdown("""
        > **Disclaimer:** This is an experimental AI tool and is **not a substitute for a professional medical diagnosis**.  
        > Always consult a qualified radiologist.
        """)
        st.markdown("---")

    # --- 7. DISPLAY & REMOVE IMAGES ---
    to_remove = None
    for idx, (name, image_pil) in enumerate(st.session_state["images"]):
        col_del, col_content = st.columns([0.05, 0.95])
        with col_del:
            if st.button("❌", key=f"remove_{idx}"):
                to_remove = idx
        with col_content:
            try:
                analyze_and_display(image_pil, name, model)
            except Exception as e:
                st.error(f"Error analyzing {name}: {e}")
    
    if to_remove is not None:
        remove_image(to_remove)
        st.experimental_rerun()

else:
    st.warning("Model could not be loaded. The application cannot proceed with analysis.")

