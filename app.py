import streamlit as st
import requests
from PIL import Image

API_URL = "http://localhost:8000/predict/"

def main():
    st.title("Flower Classification (via API)")

    uploaded_file = st.file_uploader("Upload flower image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Send image to API
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            st.write(f"Prediction: **{result['label']}** with confidence {result['confidence']:.2f}")
        else:
            st.error("Error in prediction API")

if __name__ == "__main__":
    main()
