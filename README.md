# PlantDoc AI 

PlantDoc AI is an enterprise-grade agricultural diagnostic application designed to bridge the gap between lab-trained computer vision models and messy, chaotic real-world field conditions. 

Most plant pathology models achieve near-perfect accuracy on clean, backlit datasets but fail entirely when farmers upload photos containing dirt, hands, or complex natural backgrounds. This system introduces a resilient 3-stage inference pipeline—combining **computer vision segmentation**, **deep learning classification**, **neural attention heatmaps**, and **generative LLM therapeutics**—to deliver reliable, actionable insights straight to the field.

🔗 **[Live Project Demo Link]([https://github.com/mayanktiwari-cmd](https://plantdiseasecure-ejaskrjnvnnfasfwxizw3u.streamlit.app/))** 

---

## 🛠️ The Core Problem & Our Solution

### The Domain Shift Challenge
The model is trained on a comprehensive corpus of over **54,000 images across 38 distinct plant-disease taxonomy vectors** (derived from the PlantVillage dataset). While this dataset provides incredible taxonomic depth, the images are highly controlled: perfectly centered leaves against uniform backgrounds. 

When applied to raw field photos or external images from the web, standard models suffer from severe performance drops due to background noise.

### The PlantDoc AI Pipeline Architecture

[Raw Input Image]
│
▼
┌───────────┐      Fail      ┌────────────┐
│  Stage 1  ├───────────────►│  Fallback  │ (Use original bounds)
└─────┬─────┘                └─────┬──────┘
│ (HSV-Guided GrabCut)       │
▼                            ▼
┌───────────┐
│  Stage 2  │◄─────────────────────┘
└─────┬─────┘ (ResNet34 Classification on Clean Leaf Matrix)
│
▼
┌───────────┐
│  Stage 3  │─────► [Grad-CAM Visual Attention Overlay]
└─────┬─────┘
│
▼
┌───────────┐
│  Stage 4  │─────► [Llama 3.1 Contextual Treatment Generation]
└───────────┘

1. **Dual-Strategy Segmentation (`model.py`)**: Before the image ever hits the neural network, it goes through an HSV-guided GrabCut routine. By generating a vegetation mask in the HSV color space, it separates the target leaf from surrounding soil or weeds. If the background is too ambiguous, it automatically switches to a rectangle-initialized GrabCut bounding method, or falls back safely to the original image coordinates.
2. **Pathology Classification (`model.py`)**: The cleaned, isolated leaf matrix is fed into a fine-tuned **ResNet34** architecture to determine the plant species and specific pathological anomaly out of 38 target classes.
3. **Explainability Layer (`model.py`)**: Using dynamic forward and backward registration hooks on the final convolutional layer (`layer4`), the engine generates a **Grad-CAM** activation heatmap. This shows the exact structural or texture regions the network prioritized to make its decision, ensuring visual transparency for the user.
4. **Therapeutic Synthesis (`llm.py`)**: If a disease is found, the classification output triggers a low-latency context injection to a `llama-3.1-8b-instant` node via Groq, generating immediate, localized organic or chemical treatment plans under 150 words.

---

## 💻 Repository Structure

```text
├── .streamlit/               # Production runtime UI configuration settings
├── .gitignore                # Target tracking exclusion profiles
├── app.py                    # Streamlit interface execution layer & CSS architecture
├── class_names.json          # Index reference mapping for the 38 pathology classes
├── llm.py                    # Groq client core & Llama 3.1 contextual execution engine
├── model.py                  # Vision pipeline (GrabCut extraction, ResNet34, Grad-CAM hooks)
├── plant_disease_model.pth   # State dictionary containing fine-tuned ResNet34 weights
└── requirements.txt          # Python deployment dependency index
