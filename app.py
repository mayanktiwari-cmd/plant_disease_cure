import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from PIL import Image
from model import load_model, predict
from llm import get_cure

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="PlantDoc AI | Enterprise",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# STARTUP GRADIENT & GLASSMORPHISM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* 1. THE GLOWING BACKGROUND */
/* Injects a massive, soft emerald radial glow in the top left over the dark canvas */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 15% 0%, rgba(16, 185, 129, 0.18) 0%, #09090B 50%) !important;
    background-color: #09090B !important;
}

/* Hide Streamlit Artifacts */
#MainMenu, header, footer { display: none !important; }
.block-container { 
    padding: 3rem 4rem !important; 
    max-width: 1400px; 
}

/* 2. GRADIENT TYPOGRAPHY */
.startup-header {
    margin-bottom: 3rem;
}
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
    /* The core startup gradient trick */
    background: linear-gradient(135deg, #FAFAFA 0%, #A7F3D0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
}
.subtext {
    color: #A1A1AA;
    font-size: 1.15rem;
    font-weight: 400;
    max-width: 600px;
}

/* 3. GLASSMORPHISM PANELS */
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

/* 4. SLEEK METRICS */
.metric-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}
.metric-box {
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 1.25rem;
    border-radius: 12px;
}
.metric-label {
    font-size: 0.75rem;
    color: #A1A1AA;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #FAFAFA;
}
.metric-highlight { color: #10B981; }

/* 5. GLOWING PROGRESS BAR */
.conf-track {
    width: 100%;
    height: 6px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    margin-top: 0.8rem;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    background: linear-gradient(90deg, #059669, #34D399);
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
}

/* AI Output Text */
.cure-text {
    font-size: 1rem;
    line-height: 1.8;
    color: #E4E4E7;
}

/* Streamlit Native Element Tweaks */
[data-testid="stTabs"] button { font-weight: 500; }
[data-testid="stFileUploader"] {
    background: rgba(0, 0, 0, 0.2);
    border: 1px dashed rgba(16, 185, 129, 0.4);
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MODEL INITIALIZATION
# ─────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# ─────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────
st.markdown("""
<div class="startup-header">
    <div class="badge">RESNET34 + LLAMA 3 VISION</div>
    <h1 class="gradient-text">PlantDoc AI</h1>
    <div class="subtext">Enterprise-grade leaf pathology diagnostic system equipped with automated therapeutic sequence synthesis.</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MAIN DASHBOARD LAYOUT
# ─────────────────────────────────────────
col_nav, col_main = st.columns([1, 2.2], gap="large")

with col_nav:
    st.markdown('<div class="glass-panel"><div class="panel-title">⌘ Data Ingestion</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Local Upload", "Hardware Feed"])
    image = None
    
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Select Tensor Matrix", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        camera = st.camera_input("Initialize Feed", label_visibility="collapsed")
        if camera:
            image = Image.open(camera).convert("RGB")
            
    st.markdown('</div>', unsafe_allow_html=True)

    # System Specs Panel
    st.markdown("""
    <div class="glass-panel">
        <div class="panel-title">⚡ Model Telemetry</div>
        <div style="margin-bottom: 1.2rem;">
            <div class="metric-label">Training Corpus</div>
            <div style="color: #FAFAFA; font-weight: 600; font-size: 1.1rem;">54,000+ Vectors</div>
        </div>
        <div style="margin-bottom: 1.2rem;">
            <div class="metric-label">Target Taxonomy</div>
            <div style="color: #FAFAFA; font-weight: 600; font-size: 1.1rem;">38 Pathologies</div>
        </div>
        <div>
            <div class="metric-label">Top-1 Accuracy</div>
            <div style="color: #10B981; font-weight: 700; font-size: 1.1rem;">99.7%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_main:
    if image is not None:
        
        # Inference Results Panel
        st.markdown('<div class="glass-panel"><div class="panel-title">⚗️ Inference Engine Output</div>', unsafe_allow_html=True)
        
        with st.spinner("Processing neural pathways..."):
            plant, disease, confidence = predict(image, model)
            
        disease_display = "Optimal Baseline" if disease.lower() == "healthy" else disease
            
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
            <div style="display:flex; justify-content: space-between; align-items: center;">
                <div class="metric-label" style="margin:0;">Network Confidence</div>
                <div class="metric-value" style="font-size:1.1rem; color:#A1A1AA;">{confidence:.1f}%</div>
            </div>
            <div class="conf-track">
                <div class="conf-fill" style="width: {confidence}%;"></div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Therapeutic Synthesis Panel
        st.markdown('<div class="glass-panel"><div class="panel-title">🧬 Generative Treatment Synthesis</div>', unsafe_allow_html=True)
        
        if disease.lower() == "healthy":
            st.markdown("""
            <div class="cure-text">
                <span style="color: #10B981; font-weight: 600;">System Clear:</span> Target structural matrix indicates zero pathological anomalies. Continue standard environmental maintenance protocols.
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("LLM synthesizing chemical protocols..."):
                try:
                    cure = get_cure(plant, disease)
                    st.markdown(f'<div class="cure-text">{cure}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error("API Gateway Timeout: Unable to reach Llama 3 node.")

        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        # High-end Empty State
        st.markdown("""
        <div class="glass-panel" style="text-align: center; padding: 8rem 2rem; border-style: dashed; border-color: rgba(16, 185, 129, 0.3);">
            <div style="background: rgba(16, 185, 129, 0.1); width: 64px; height: 64px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1.5rem auto;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
            </div>
            <div style="color: #FAFAFA; font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem;">
                Awaiting Data Ingestion
            </div>
            <div style="color: #A1A1AA; font-size: 1rem; max-width: 400px; margin: 0 auto;">
                Upload a botanical tensor matrix via the local disk or initialize the live camera feed to begin inference.
            </div>
        </div>
        """, unsafe_allow_html=True)