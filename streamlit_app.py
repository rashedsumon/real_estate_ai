import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Import our backend modular blocks
from data_loader import download_dataset
from model import process_bracketed_images

st.set_page_config(page_title="AI Real Estate Photo Studio", layout="wide")

# --- DATASET MANAGEMENT ---
if "dataset_path" not in st.session_state:
    with st.spinner("Syncing baseline exposure datasets from Kaggle..."):
        path = download_dataset()
        st.session_state["dataset_path"] = path

# --- UI HEADER ---
st.title("🏠 AI-Powered Real Estate Photo Solution")
st.write("Deliver professional-quality HDR enhancements and clutter removal instantly.")

st.divider()

# --- SIDEBAR INPUTS ---
st.sidebar.header("📥 User Input (Source Data)")

uploaded_files = st.sidebar.file_uploader(
    "Upload Bracketed Photos (Under, Normal, Over Exposed)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

user_commands = st.sidebar.text_input(
    "User Commands / Preferences (Optional):",
    placeholder="e.g., 'remove the trash can', 'apply twilight enhancement'"
)

# --- MAIN PAGE COMPONENT ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Uploaded Brackets")
    imgs_cv = []
    if uploaded_files:
        # Display small previews of the uploads
        for idx, file in enumerate(uploaded_files):
            # Reset pointer to read file correctly
            file.seek(0)
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is not None:
                imgs_cv.append(img)
                st.image(file, caption=f"Bracket Frame {idx+1}", use_container_width=True)
            else:
                st.error(f"Could not read frame {idx+1}. Image file may be corrupt.")
    else:
        st.info("Please upload 2 or more bracketed images via the sidebar to initiate the pipeline.")

with col2:
    st.subheader("🖼️ Model Output (The Final Product)")
    
    if uploaded_files and st.sidebar.button("🚀 Process & Enhance"):
        if len(imgs_cv) < 2:
            st.error("Please provide at least 2 valid image brackets to compute HDR blending.")
        else:
            with st.spinner("Blending HDR elements, running inpainting models & upscaling..."):
                try:
                    # Run processing code from model.py
                    output_image = process_bracketed_images(imgs_cv, user_commands)
                    
                    if output_image is not None:
                        # Convert BGR to RGB for streamlit rendering
                        output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                        pil_out = Image.fromarray(output_rgb)
                        
                        st.image(pil_out, caption="HDR Enhanced, Cleaned & Upscaled Image", use_container_width=True)
                        
                        # Download link for the final high-quality piece
                        st.success("Enhancement successful!")
                        
                        is_success, buffer = cv2.imencode(".jpg", output_image)
                        if is_success:
                            st.download_button(
                                label="⬇️ Download Professional Master Copy",
                                data=buffer.tobytes(),
                                file_name="real_estate_enhanced.jpg",
                                mime="image/jpeg"
                            )
                    else:
                        st.error("Processing returned empty layout. Verify image parameters.")
                except Exception as model_err:
                    st.error(f"Pipeline processing failed: {str(model_err)}")