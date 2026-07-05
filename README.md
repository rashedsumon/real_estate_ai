# AI-Powered Real Estate Photo Enhancer

An AI solution deployed on Streamlit Cloud that takes bracketed real estate photos (under, normal, over-exposed) and user commands to produce professionally enhanced, object-cleared HDR images.

## Features
- **Auto-Dataset Sync**: Automatically downloads the `zara2099/low-light-image-enhancement-dataset` for reference training frames.
- **HDR Exposure Blending**: Merges bracketed exposures into a balanced master image.
- **Object Removal & Commands**: Simulates intelligent inpainting and twilight enhancements.

## Setup
1. Deploy directly to Streamlit Cloud pointing to `streamlit_app.py`.
2. Ensure you add your Kaggle credentials (`KAGGLE_USERNAME` and `KAGGLE_KEY`) to Streamlit's Secrets manager if the dataset requires authentication.