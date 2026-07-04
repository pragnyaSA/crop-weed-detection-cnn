"""
streamlit_app.py
-----------------
Interactive web demo for the Crop vs Weed CNN classifier.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from model import IMG_SIZE  # noqa: E402

CLASS_NAMES = ["crop", "weed"]
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.keras"

st.set_page_config(page_title="Crop vs Weed Detection", page_icon="🌱", layout="centered")

st.title("🌱 Crop vs Weed Detection")
st.caption(
    "CNN-based classifier for precision agriculture — "
    "based on the paper *Efficient Crop vs Weed Detection in Precision "
    "Agriculture: A CNN Approach for Real-Time Decision Making* (INOCON 2024)."
)


@st.cache_resource
def load_cnn_model():
    from tensorflow.keras.models import load_model
    if not MODEL_PATH.exists():
        return None
    return load_model(MODEL_PATH)


model = load_cnn_model()

if model is None:
    st.warning(
        f"No trained model found at `{MODEL_PATH}`.\n\n"
        "Train one first with:\n\n"
        "```bash\npython src/data_preprocessing.py --source <dataset> --output data/processed\n"
        "python src/train.py --data data/processed --epochs 25\n```"
    )
else:
    uploaded = st.file_uploader("Upload a field image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        img_resized = image.resize(IMG_SIZE)
        arr = np.expand_dims(np.array(img_resized).astype("float32") / 255.0, axis=0)

        with st.spinner("Classifying..."):
            probs = model.predict(arr, verbose=0)[0]

        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        confidence = probs[idx]

        if label == "crop":
            st.success(f"✅ Predicted: **Crop** ({confidence:.1%} confidence)")
        else:
            st.error(f"⚠️ Predicted: **Weed** ({confidence:.1%} confidence)")

        st.subheader("Class probabilities")
        st.bar_chart({name: float(p) for name, p in zip(CLASS_NAMES, probs)})

st.divider()
st.caption(
    "Model: VGG16-based transfer learning CNN · Input 224×224 RGB · "
    "2 fully-connected layers (256, 128) · Softmax output"
)
