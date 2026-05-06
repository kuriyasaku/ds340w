from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.config.paths import AUDIT_DIR


st.set_page_config(page_title="Pathology Audit Viewer", layout="wide")


def load_audit_tables():
    tables = {}
    for csv_path in sorted(AUDIT_DIR.rglob("*_audit.csv")):
        tables[str(csv_path)] = pd.read_csv(csv_path)
    return tables


def read_img(path: str):
    if path and Path(path).exists():
        return Image.open(path)
    return None


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1350px;}
    .card {background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 18px; padding: 18px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);}
    .panel-title {font-size: 18px; font-weight: 700; color: #111827; margin-bottom: 10px; text-align: center;}
    .small-title {font-size: 14px; font-weight: 700; color: #111827; margin-bottom: 6px;}
    .metric-box {background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 12px 16px; margin-bottom: 10px;}
    .big {font-size: 28px; font-weight: 800; color: #1d4ed8;}
    .green {color: #16a34a; font-weight: 700;}
    .red {color: #dc2626; font-weight: 700;}
    .orange {color: #f97316; font-weight: 700;}
    .toolbar {background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 18px; padding: 10px 14px; margin-bottom: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="toolbar"><div class="small-title">Perturbation Controls</div></div>', unsafe_allow_html=True)

all_tables = load_audit_tables()
if len(all_tables) == 0:
    st.info("No audit CSV files found in outputs/audit")
    st.stop()

selected_csv = st.selectbox("Audit table", list(all_tables.keys()))
df = all_tables[selected_csv]
perturbations = sorted(df["perturbation"].unique().tolist())
selected_perturbation = st.radio("Perturbation", perturbations, horizontal=True)
filtered = df[df["perturbation"] == selected_perturbation].reset_index(drop=True)
slide_ids = sorted(filtered["slide_id"].unique().tolist())
selected_slide = st.selectbox("Slide", slide_ids)
slide_df = filtered[filtered["slide_id"] == selected_slide].reset_index(drop=True)
row_idx = st.slider("Patch index", 0, max(0, len(slide_df) - 1), 0)
row = slide_df.iloc[row_idx] if len(slide_df) else None

if row is None:
    st.stop()

left, right = st.columns(2)
with left:
    st.markdown('<div class="card"><div class="panel-title">Original Patch</div></div>', unsafe_allow_html=True)
    original_img = read_img(row["tile_path"])
    if original_img is not None:
        st.image(original_img, use_container_width=False, width=256)
with right:
    st.markdown('<div class="card"><div class="panel-title">Perturbed Patch</div></div>', unsafe_allow_html=True)
    pert_overlay = read_img(row["perturbed_overlay"])
    if pert_overlay is not None:
        st.image(pert_overlay, use_container_width=False, width=256)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Original Prediction</div>', unsafe_allow_html=True)
    st.write(f"Tumor probability: {row['original_prob_tumor']:.6f}")
    confidence_text = "high" if row["original_prob_tumor"] >= 0.8 else "moderate" if row["original_prob_tumor"] >= 0.5 else "low"
    st.write(f"Confidence: {confidence_text}")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Perturbed Prediction</div>', unsafe_allow_html=True)
    st.write(f"Tumor probability: {row['perturbed_prob_tumor']:.6f}")
    st.write(f"Confidence drop: {row['confidence_drop']:.6f}")
    st.markdown('</div>', unsafe_allow_html=True)

hm_col, metric_col = st.columns([2, 1])
with hm_col:
    st.markdown('<div class="card"><div class="panel-title">Explanation Maps</div></div>', unsafe_allow_html=True)
    hm1, hm2 = st.columns(2)
    with hm1:
        st.markdown('<div class="small-title">Original Grad-CAM</div>', unsafe_allow_html=True)
        img = read_img(row["original_heatmap"])
        if img is not None:
            st.image(img, width=220)
    with hm2:
        st.markdown('<div class="small-title">Perturbed Grad-CAM</div>', unsafe_allow_html=True)
        img = read_img(row["perturbed_heatmap"])
        if img is not None:
            st.image(img, width=220)
with metric_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Metrics</div>', unsafe_allow_html=True)
    flip_value = "Yes" if int(row["prediction_flip"]) == 1 else "No"
    st.metric("confidence drop", f"{row['confidence_drop']:.6f}")
    st.metric("prediction flip", flip_value)
    st.metric("explanation shift", f"{row['explanation_shift']:.6f}")
    st.markdown('</div>', unsafe_allow_html=True)
