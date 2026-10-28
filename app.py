"""Streamlit Web Application: Automated Industrial Quality Inspection (MVTecADsmall)."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.config import (
    CATEGORIES,
    CHECKPOINT_DIR,
    DATA_ROOT,
    IMG_SIZE,
    OUT_DIR,
    REPORT_DIR,
    get_data_type,
)
from src.dataset import load_rgb
from src.evaluate import predict_image
from src.models import ConvAutoencoder, DefectCNN, get_device
from src.train import reconstruction_error

st.set_page_config(
    page_title="Industrial Quality Inspection - MVTecADsmall",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Automated Industrial Quality Inspection")
st.subheader("CNN Defect Classification & Autoencoder Anomaly Detection (MVTecADsmall)")

# Check if model exists
ckpt = CHECKPOINT_DIR / "cnn_defect_classifier.pt"
if not ckpt.exists():
    st.error("CNN model checkpoint not found. Please train the model first by running:")
    st.code("python run_all.py")
    st.stop()


@st.cache_resource
def load_cnn_model():
    device = get_device()
    cnn = DefectCNN().to(device)
    cnn.load_state_dict(torch.load(ckpt, map_location=device))
    cnn.eval()
    return cnn


cnn = load_cnn_model()

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Quality Inspection (Inference)", "10 Categories & Severity Tables", "Model Performance"],
)

if page == "Quality Inspection (Inference)":
    st.markdown("### Step 8: Predict on New Images")
    st.write(
        "Upload a product image or choose a sample from the test dataset to classify as "
        "**Defective** or **Non-defective**, and generate an anomaly heatmap."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        category = st.selectbox(
            "Select Industrial Product Category:",
            CATEGORIES,
            index=0,
        )
        data_type = get_data_type(category)
        st.info(f"Category: **{category}** | Data Type: **{data_type}**")

        mode = st.radio("Image Source:", ["Choose Sample Test Image", "Upload Custom Image"])
        selected_path = None

        if mode == "Choose Sample Test Image":
            cat_dir = DATA_ROOT / category / "test"
            if cat_dir.exists():
                all_test_imgs = list(cat_dir.glob("*/*.png"))
                if all_test_imgs:
                    def format_path(p: Path):
                        defect_name = p.parent.name
                        return f"[{defect_name}] {p.name}"

                    chosen = st.selectbox("Select Test Sample:", all_test_imgs, format_func=format_path)
                    selected_path = chosen
            else:
                st.warning(f"Test directory not found at {cat_dir}")
        else:
            uploaded = st.file_uploader("Upload Product Image (PNG/JPG):", type=["png", "jpg", "jpeg"])
            if uploaded is not None:
                tmp = OUT_DIR / "_upload.png"
                tmp.write_bytes(uploaded.getvalue())
                selected_path = tmp

        threshold = st.slider("Classification Threshold:", 0.1, 0.9, 0.5, 0.05)

    with col2:
        if selected_path is not None and Path(selected_path).exists():
            rgb = load_rgb(str(selected_path), IMG_SIZE)
            tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

            label, prob = predict_image(cnn, tensor, threshold=threshold)

            st.write("#### Inspection Result")
            if label == "Defective":
                st.error(f"### Result: **DEFECTIVE** (Confidence: {prob*100:.2f}%)")
            else:
                st.success(f"### Result: **NON-DEFECTIVE (GOOD)** (Confidence: {(1-prob)*100:.2f}%)")

            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(rgb, caption="Inspected Image (128x128 RGB)", use_column_width=True)

            with img_col2:
                # Autoencoder Anomaly Residual Map
                ae_path = CHECKPOINT_DIR / f"ae_{category}.pt"
                if ae_path.exists():
                    device = get_device()
                    ae = ConvAutoencoder().to(device)
                    ae.load_state_dict(torch.load(ae_path, map_location=device))
                    ae.eval()
                    with torch.no_grad():
                        x = tensor.unsqueeze(0).to(device)
                        recon = ae(x)[0].cpu().permute(1, 2, 0).numpy()
                        err = reconstruction_error(ae, x).item()
                    residual = np.abs(rgb.astype(np.float32) / 255.0 - recon).mean(axis=2)
                    norm_res = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
                    heatmap = cv2.applyColorMap((norm_res * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    st.image(heatmap, caption=f"Anomaly Residual Heatmap (Error: {err:.5f})", use_column_width=True)
                else:
                    st.info("Autoencoder checkpoint for this category not yet generated.")

elif page == "10 Categories & Severity Tables":
    st.markdown("### 10 Categories of Data Types in MVTecADsmall")
    st.write(
        "Breakdown of all 10 industrial categories (Object vs. Texture data types) with "
        "defect severity levels categorized into **no**, **low** (<1%), **medium** (1-5%), and **high** (>5%)."
    )

    summary_file = OUT_DIR / "category_summary.csv"
    if summary_file.exists():
        sum_df = pd.read_csv(summary_file)
        st.dataframe(sum_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Separate Table per Category")
    sel_cat = st.selectbox("View Table for Category:", CATEGORIES)
    cat_table_path = REPORT_DIR / f"{sel_cat}_severity_table.csv"
    if cat_table_path.exists():
        cat_df = pd.read_csv(cat_table_path)
        st.write(f"**Data Type**: `{get_data_type(sel_cat)}` | **Category**: `{sel_cat}`")
        st.dataframe(cat_df, use_container_width=True)
    else:
        st.warning("Category table CSV not found. Please run: python run_all.py --reports-only")

elif page == "Model Performance":
    st.markdown("### CNN & Anomaly Detection Performance Evaluation")

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Confusion Matrix")
        cm_path = OUT_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Test Set Confusion Matrix")
        else:
            st.info("Confusion matrix not generated yet.")

    with col2:
        st.write("#### Training Loss & Accuracy Curves")
        curve_path = OUT_DIR / "training_curves.png"
        if curve_path.exists():
            st.image(str(curve_path), caption="Training & Validation Loss / Accuracy")
        else:
            st.info("Training curves not generated yet.")

    metrics_path = OUT_DIR / "cnn_metrics.csv"
    if metrics_path.exists():
        st.write("#### Overall Metrics (Test Set)")
        m_df = pd.read_csv(metrics_path)
        st.table(m_df)

    st.markdown("---")
    st.write("#### Category-Level Accuracy & Evaluation Graphs")
    cat_graph = OUT_DIR / "category_accuracy_graph.png"
    comp_graph = OUT_DIR / "category_evaluation_comparison.png"
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        if cat_graph.exists():
            st.image(str(cat_graph), caption="Test Accuracy by Category & Data Type", use_column_width=True)
    with g_col2:
        if comp_graph.exists():
            st.image(str(comp_graph), caption="Actual vs. Predicted Defect Breakdown", use_column_width=True)
