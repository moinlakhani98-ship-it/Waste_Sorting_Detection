import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Waste Sorting AI", page_icon="♻️", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("♻️ Waste Sorting Detection")
st.write("Waste ki image upload karo, model batayega ke woh kis category ki hai.")

conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
st.sidebar.info("Threshold barhane se sirf zyada sure predictions dikhengi.")

source = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
file = st.file_uploader("Image choose karo", type=["jpg", "jpeg", "png"]) if source == "Upload" else st.camera_input("Photo lo")

if file:
    img = Image.open(file).convert("RGB")
    result = model.predict(img, conf=conf)[0]
    plotted = result.plot()[:, :, ::-1]   # BGR -> RGB

    c1, c2 = st.columns(2)
    c1.subheader("Original")
    c1.image(img, use_container_width=True)
    c2.subheader("Prediction")
    c2.image(plotted, use_container_width=True)

    if len(result.boxes) == 0:
        st.warning("Koi object detect nahi hua. Confidence threshold kam karke dekho.")
    else:
        rows = [{"Class": model.names[int(b.cls)], "Confidence": round(float(b.conf), 2)}
                for b in result.boxes]
        df = pd.DataFrame(rows)
        st.subheader("Detections")
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df["Class"].value_counts())