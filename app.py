import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from PIL import Image
from model import load_model, predict
from llm import get_cure


#  PAGE CONFIG 
st.set_page_config(
    page_title="PlantDoc AI | Enterprise",
    layout="wide",
    initial_sidebar_state="collapsed"
)


#CSS  unchanged design + new vision panels 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 15% 0%, rgba(16, 185, 129, 0.18) 0%, #09090B 50%) !important;
    background-color: #09090B !important;
}

#MainMenu, header, footer { display: none !important; }
.block-container { padding: 3rem 4rem !important; max-width: 1400px; }

.startup-header { margin-bottom: 3rem; }
.badge {
    display: inline-block;
    background: rgba(16, 185, 129, 0.1);
    color: #34D399;
    padding: 0.35rem 1rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    border: 1px solid rgba(16, 185, 129, 0.2);
    margin-bottom: 1.5rem;
}
.gradient-text {
    font-weight: 800;
    font-size: 3.5rem;
    margin: 0 0 0.5rem 0;
    background: linear-gradient(135deg, #FAFAFA 0%, #A7F3D0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
}
.subtext { color: #A1A1AA; font-size: 1.15rem; font-weight: 400; max-width: 600px; }

.glass-panel {
    background: rgba(24, 24, 27, 0.4);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
}
.panel-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    color: #34D399;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
.metric-box {
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 1.25rem;
    border-radius: 12px;
}
.metric-label { font-size: 0.75rem; color: #A1A1AA; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem; }
.metric-value { font-size: 1.5rem; font-weight: 700; color: #FAFAFA; }
.metric-highlight { color: #10B981; }

.conf-track {
    width: 100%; height: 6px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px; margin-top: 0.8rem; overflow: hidden;
}
.conf-fill {
    height: 100%;
    background: linear-gradient(90deg, #059669, #34D399);
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
}

/* ── Vision analysis panel (new) ── */
.vision-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1rem;
    margin-bottom: 0;
}
.vision-cell {
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.vision-cell-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
    color: #6EE7B7;
    margin-bottom: 0.75rem;
}
.seg-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    margin-top: 0.5rem;
}
.seg-ok  { background: rgba(16,185,129,0.15); color: #34D399; border: 1px solid rgba(16,185,129,0.25); }
.seg-skip{ background: rgba(161,161,170,0.1);  color: #A1A1AA; border: 1px solid rgba(255,255,255,0.08); }

/* ── Grad-CAM legend ── */
.cam-legend {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.cam-bar {
    width: 100px; height: 6px; border-radius: 4px;
    background: linear-gradient(90deg, #2563EB, #10B981, #EF4444);
}
.cam-legend-label { font-size: 0.65rem; color: #71717A; }

.cure-text { font-size: 1rem; line-height: 1.8; color: #E4E4E7; }

[data-testid="stTabs"] button { font-weight: 500; }

[data-testid="stFileUploaderDropzone"] {
    background-color: transparent !important;
    background: transparent !important;
}
[data-testid="stFileUploader"] {
    background: rgba(24, 24, 27, 0.4) !important;
    border: 1px dashed rgba(16, 185, 129, 0.4) !important;
    border-radius: 12px;
}
[data-testid="stCameraInput"], [data-testid="stCameraInput"] > div {
    background-color: transparent !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# MODEL INIT
@st.cache_resource
def get_model():
    return load_model()

model = get_model()


# HERO 
st.markdown("""
<div class="startup-header">
    <div class="badge">RESNET34 + GRADCAM + LLAMA 3</div>
    <h1 class="gradient-text">PlantDoc AI</h1>
    <div class="subtext">Enterprise-grade leaf pathology diagnostic system with automated background segmentation, neural heatmap localisation, and generative therapeutic synthesis.</div>
</div>
""", unsafe_allow_html=True)


# LAYOUT 
col_nav, col_main = st.columns([1, 2.2], gap="large")

# LEFT NAV COLUMN
with col_nav:
    st.markdown('<div class="glass-panel"><div class="panel-title">⌘ Data Ingestion</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Local Upload", "Hardware Feed"])
    image = None

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Select Tensor Matrix",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        camera = st.camera_input("Initialize Feed", label_visibility="collapsed")
        if camera:
            image = Image.open(camera).convert("RGB")

    st.markdown('</div>', unsafe_allow_html=True)

    # Model telemetry
    st.markdown("""
    <div class="glass-panel">
        <div class="panel-title">⚡ Model Telemetry</div>
        <div style="margin-bottom:1.2rem;">
            <div class="metric-label">Training Corpus</div>
            <div style="color:#FAFAFA;font-weight:600;font-size:1.1rem;">54,000+ Vectors</div>
        </div>
        <div style="margin-bottom:1.2rem;">
            <div class="metric-label">Target Taxonomy</div>
            <div style="color:#FAFAFA;font-weight:600;font-size:1.1rem;">38 Pathologies</div>
        </div>
        <div style="margin-bottom:1.2rem;">
            <div class="metric-label">Top-1 Accuracy</div>
            <div style="color:#10B981;font-weight:700;font-size:1.1rem;">99.7%</div>
        </div>
        <div style="margin-bottom:1.2rem;">
            <div class="metric-label">Segmentation</div>
            <div style="color:#FAFAFA;font-weight:600;font-size:1.1rem;">GrabCut + MorphOps</div>
        </div>
        <div>
            <div class="metric-label">Explainability</div>
            <div style="color:#FAFAFA;font-weight:600;font-size:1.1rem;">Grad-CAM (Layer4)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# MAIN OUTPUT COLUMN
with col_main:
    if image is not None:

        # Run full pipeline
        with st.spinner("Running inference pipeline..."):
            plant, disease, confidence, seg_img, cam_img, seg_ok = predict(image, model)

        disease_display = "Optimal Baseline" if disease.lower() == "healthy" else disease

        # 1. Inference results
        st.markdown('<div class="glass-panel"><div class="panel-title">⚗️ Inference Engine Output</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">Biological Class Detected</div>
                <div class="metric-value">{plant}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Pathological State</div>
                <div class="metric-value metric-highlight">{disease_display}</div>
            </div>
        </div>
        <div class="metric-box">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div class="metric-label" style="margin:0;">Network Confidence</div>
                <div class="metric-value" style="font-size:1.1rem;color:#A1A1AA;">{confidence:.1f}%</div>
            </div>
            <div class="conf-track">
                <div class="conf-fill" style="width:{min(confidence,100)}%;"></div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Vision analysis — 3-up panel (NEW) 
        st.markdown('<div class="glass-panel"><div class="panel-title">🔬 Vision Analysis Pipeline</div>', unsafe_allow_html=True)

        st.markdown('<div class="vision-grid">', unsafe_allow_html=True)

        # Three columns: original | segmented | grad-cam
        v1, v2, v3 = st.columns(3, gap="small")

        with v1:
            st.markdown('<div class="vision-cell-label">Raw Input</div>', unsafe_allow_html=True)
            st.image(image.resize((224, 224)), use_container_width=True)
            st.markdown(
                '<div class="seg-badge seg-skip">Original Frame</div>',
                unsafe_allow_html=True
            )

        with v2:
            seg_label  = "seg-ok"   if seg_ok else "seg-skip"
            seg_status = "Segmented" if seg_ok else "Fallback — Original"
            st.markdown('<div class="vision-cell-label">Background Removed</div>', unsafe_allow_html=True)
            st.image(seg_img, use_container_width=True)
            st.markdown(
                f'<div class="seg-badge {seg_label}">{seg_status}</div>',
                unsafe_allow_html=True
            )

        with v3:
            st.markdown('<div class="vision-cell-label">Grad-CAM Attention</div>', unsafe_allow_html=True)
            st.image(cam_img, use_container_width=True)
            st.markdown("""
                <div class="cam-legend">
                    <span class="cam-legend-label">Low</span>
                    <div class="cam-bar"></div>
                    <span class="cam-legend-label">High</span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # close vision-grid

        # Explanation line
        if seg_ok:
            st.markdown(
                "<p style='font-size:0.78rem;color:#6B7280;margin-top:0.8rem;'>Background removed via GrabCut before classification. Grad-CAM heatmap shows which regions of the segmented leaf activated the final ResNet34 block.</p>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<p style='font-size:0.78rem;color:#6B7280;margin-top:0.8rem;'>Segmentation fell back to original — leaf boundary detection was ambiguous. Grad-CAM heatmap still reflects model attention on the raw input.</p>",
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)  # close glass-panel

        # 3. Treatment synthesis
        st.markdown('<div class="glass-panel"><div class="panel-title">🧬 Generative Treatment Synthesis</div>', unsafe_allow_html=True)

        if disease.lower() == "healthy":
            st.markdown("""
            <div class="cure-text">
                <span style="color:#10B981;font-weight:600;">System Clear:</span>
                Target structural matrix indicates zero pathological anomalies.
                Continue standard environmental maintenance protocols.
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("LLM synthesizing treatment protocols..."):
                try:
                    cure = get_cure(plant, disease)
                    st.markdown(f'<div class="cure-text">{cure}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"API Gateway Timeout: Unable to reach Llama 3 node. Check your Groq API key.\n{e}")

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Empty state unchanged from original
        st.markdown("""
        <div class="glass-panel" style="text-align:center;padding:8rem 2rem;border-style:dashed;border-color:rgba(16,185,129,0.3);">
            <div style="background:rgba(16,185,129,0.1);width:64px;height:64px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 1.5rem auto;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
            </div>
            <div style="color:#FAFAFA;font-size:1.25rem;font-weight:600;margin-bottom:0.5rem;">Awaiting Data Ingestion</div>
            <div style="color:#A1A1AA;font-size:1rem;max-width:400px;margin:0 auto;">
                Upload a botanical tensor matrix via the local disk or initialize the live camera feed to begin inference.
            </div>
        </div>
        """, unsafe_allow_html=True)
