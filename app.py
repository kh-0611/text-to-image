"""
Text-to-Image Generator
Generates images from text prompts using Hugging Face's Inference API
(FLUX.1-schnell / Stable Diffusion XL). Needs a FREE Hugging Face token.
"""
import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# =========================================================
# CUSTOM CSS — MIDNIGHT AI CREATIVE LAB
# =========================================================

st.markdown("""
<style>

/* =========================================================
   GLOBAL BACKGROUND
   ========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 12% 8%,
            rgba(49, 88, 140, 0.28),
            transparent 28%
        ),
        radial-gradient(
            circle at 88% 12%,
            rgba(76, 65, 130, 0.24),
            transparent 30%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(25, 52, 88, 0.35),
            transparent 40%
        ),
        #07111f;

    color: #e8eef7;
}


/* =========================================================
   MAIN CONTAINER
   ========================================================= */

.block-container {
    max-width: 950px;
    padding-top: 3.5rem;
    padding-bottom: 4rem;
}


/* =========================================================
   TITLE
   ========================================================= */

h1 {
    font-family: Georgia, "Times New Roman", serif !important;

    font-size: 3.2rem !important;
    font-weight: 700 !important;

    text-align: center;

    color: #edf4ff !important;

    letter-spacing: -1px;

    margin-bottom: 0.4rem !important;
}

h2, h3 {
    font-family: Georgia, "Times New Roman", serif !important;

    color: #dce9f8 !important;
}


/* =========================================================
   GENERAL TEXT
   ========================================================= */

.stApp p {
    color: #9eafc5;
}

.stMarkdown p {
    line-height: 1.7;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0c1a2c 0%,
            #091522 50%,
            #07111f 100%
        );

    border-right: 1px solid rgba(148, 177, 210, 0.12);
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #dce9f8 !important;

    font-family: Georgia, "Times New Roman", serif !important;
}

section[data-testid="stSidebar"] label {
    color: #aebed1 !important;

    font-weight: 600;
}


/* =========================================================
   SIDEBAR INPUT
   ========================================================= */

section[data-testid="stSidebar"] input {
    background: #101f32 !important;

    color: #e4edf8 !important;

    border: 1px solid #29415e !important;

    border-radius: 12px !important;
}

section[data-testid="stSidebar"] input:focus {
    border-color: #628db8 !important;

    box-shadow:
        0 0 0 2px rgba(98, 141, 184, 0.12);
}


/* =========================================================
   SELECTBOX
   ========================================================= */

div[data-baseweb="select"] > div {
    background: #101f32;

    border: 1px solid #29415e;

    border-radius: 12px;

    color: #dce8f5;
}

div[data-baseweb="select"] > div:hover {
    border-color: #557da5;
}


/* =========================================================
   PROMPT TEXT AREA
   ========================================================= */

textarea {
    background:
        rgba(13, 28, 45, 0.92) !important;

    color: #e8eef7 !important;

    border: 1px solid #29445f !important;

    border-radius: 20px !important;

    padding: 20px !important;

    font-size: 15px !important;

    line-height: 1.7 !important;

    box-shadow:
        0 15px 40px rgba(0, 0, 0, 0.20);
}

textarea::placeholder {
    color: #70859d !important;
}

textarea:focus {
    border-color: #638eb8 !important;

    box-shadow:
        0 0 0 2px rgba(99, 142, 184, 0.13),
        0 15px 40px rgba(0, 0, 0, 0.25);
}


/* =========================================================
   NEGATIVE PROMPT
   ========================================================= */

div[data-baseweb="input"] {
    background: #0d1c2e;

    border: 1px solid #29445f !important;

    border-radius: 13px !important;
}

div[data-baseweb="input"] input {
    background: transparent !important;

    color: #e3edf8 !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: #638eb8 !important;
}


/* =========================================================
   GENERATE BUTTON
   ========================================================= */

.stButton > button {
    width: 100%;

    border: 1px solid rgba(126, 165, 205, 0.25) !important;

    border-radius: 16px;

    padding: 0.9rem 1.2rem;

    font-size: 16px;

    font-weight: 700;

    color: #f4f8fd !important;

    background:
        linear-gradient(
            135deg,
            #183a5b,
            #28557d,
            #486b91
        ) !important;

    box-shadow:
        0 10px 30px rgba(0, 0, 0, 0.25);

    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 14px 35px rgba(24, 58, 91, 0.40);

    background:
        linear-gradient(
            135deg,
            #204a70,
            #32658f,
            #55799e
        ) !important;
}

.stButton > button:active {
    transform: scale(0.98);
}


/* =========================================================
   GENERATED IMAGE
   ========================================================= */

[data-testid="stImage"] {
    margin-top: 28px;

    border-radius: 24px;

    overflow: hidden;

    border: 1px solid #29445f;

    box-shadow:
        0 25px 65px rgba(0, 0, 0, 0.40),
        0 0 30px rgba(64, 112, 157, 0.08);
}


/* =========================================================
   IMAGE CAPTION
   ========================================================= */

[data-testid="stImage"] figcaption {
    color: #91a6bd !important;

    font-size: 14px;

    padding-top: 10px;
}


/* =========================================================
   DOWNLOAD BUTTON
   ========================================================= */

.stDownloadButton > button {
    width: 100%;

    border-radius: 14px !important;

    border: 1px solid #385673 !important;

    background: #10243a !important;

    color: #bcd0e5 !important;

    font-weight: 700 !important;

    transition: all 0.25s ease;
}

.stDownloadButton > button:hover {
    background: #17314d !important;

    border-color: #5d82a5 !important;

    color: #e5eef8 !important;

    transform: translateY(-1px);
}


/* =========================================================
   ALERTS
   ========================================================= */

div[data-testid="stAlert"] {
    background: #102238 !important;

    border: 1px solid #2c4965 !important;

    border-radius: 16px !important;

    color: #c5d5e6 !important;
}

div[data-testid="stAlert"] p {
    color: #b9cadd !important;
}


/* =========================================================
   SPINNER
   ========================================================= */

div[data-testid="stSpinner"] {
    color: #79a3ca;
}


/* =========================================================
   DIVIDERS
   ========================================================= */

hr {
    border-color: rgba(130, 160, 190, 0.12) !important;
}


/* =========================================================
   DECORATIVE STARS
   ========================================================= */

.stApp::before {
    content: "✦";

    position: fixed;

    top: 75px;
    left: 5%;

    font-size: 25px;

    color: #6d94b8;

    opacity: 0.45;

    pointer-events: none;
}

.stApp::after {
    content: "✧";

    position: fixed;

    bottom: 75px;
    right: 6%;

    font-size: 32px;

    color: #789bbd;

    opacity: 0.35;

    pointer-events: none;
}


/* =========================================================
   HEADER
   ========================================================= */

[data-testid="stHeader"] {
    background: transparent;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 2rem;
    }

    h1 {
        font-size: 2.35rem !important;
    }

    [data-testid="stImage"] {
        border-radius: 18px;
    }

}

</style>
""", unsafe_allow_html=True)
load_dotenv()
st.set_page_config(page_title="Text to Image", page_icon="🎨", layout="centered")

MODELS = {
    "FLUX.1 Schnell (fast, high quality)": "black-forest-labs/FLUX.1-schnell",
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
}

STYLES = {
    "None": "",
    "Photorealistic": ", photorealistic, 8k, highly detailed, sharp focus",
    "Anime": ", anime style, vibrant colors, studio ghibli inspired",
    "Digital Art": ", digital art, trending on artstation, concept art",
    "Oil Painting": ", oil painting, textured brush strokes, classical art",
    "3D Render": ", 3d render, octane render, soft lighting",
}

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
token = st.sidebar.text_input(
    "Hugging Face Token",
    value=os.getenv("HF_TOKEN", ""),
    type="password",
    help="Get a free token at https://huggingface.co/settings/tokens",
)
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
style = st.sidebar.selectbox("Style", list(STYLES.keys()))
size = st.sidebar.selectbox("Size", ["1024x1024", "768x768", "512x512", "1024x768", "768x1024"])
width, height = map(int, size.split("x"))

# ---------------- Main ----------------
st.title("🎨 Text to Image Generator")
st.write("Describe an image and let AI draw it.")

prompt = st.text_area(
    "Your prompt",
    placeholder="A cozy tea stall on a rainy evening in Hyderabad, warm lights, cinematic",
    height=100,
)
negative = st.text_input("Negative prompt (optional)", placeholder="blurry, low quality, distorted")

if st.button("✨ Generate", type="primary"):
    if not token:
        st.error("Please add your Hugging Face token in the sidebar (or in a .env file).")
    elif not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        client = InferenceClient(token=token)
        full_prompt = prompt.strip() + STYLES[style]
        kwargs = {"width": width, "height": height}
        if negative.strip():
            kwargs["negative_prompt"] = negative.strip()

        with st.spinner("Generating image... (10-60 seconds)"):
            try:
                image = client.text_to_image(full_prompt, model=MODELS[model_name], **kwargs)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.info(
                    "Common fixes: check your token, wait a minute if the model is loading, "
                    "or try the other model / a smaller size."
                )
                st.stop()

        st.image(image, caption=prompt)
        buf = BytesIO()
        image.save(buf, format="PNG")
        st.download_button("⬇️ Download PNG", buf.getvalue(), "generated.png", "image/png")