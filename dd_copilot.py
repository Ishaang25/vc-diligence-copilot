import streamlit as st
import fitz  # PyMuPDF
import base64
from openai import OpenAI

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Due Diligence Copilot",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Custom CSS (Dark FinTech Theme)
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }

    html, body, [class*="css"] {
        background-color: #0e1117;
        color: white;
    }

    h1, h2, h3, h4, h5, h6, p, label, div {
        color: white !important;
    }

    .main-title {
        text-align: center;
        font-size: 2.7rem;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        color: #9ca3af;
        margin-bottom: 35px;
    }

    .report-container {
        background: linear-gradient(145deg, #151b28, #111827);
        border: 1px solid #2a3441;
        border-radius: 16px;
        padding: 30px;
        margin-top: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg,#2563eb,#3b82f6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg,#1d4ed8,#2563eb);
        color: white;
    }

    .stFileUploader {
        background-color: #161b22;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">AI Due Diligence Copilot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload a startup pitch deck (PDF) and receive an AI-powered VC diligence report.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# OpenAI API Key (Using Streamlit Secrets)
# -----------------------------
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("OpenAI API key not found in Streamlit Secrets. Please add it in the app settings.")
    st.stop()

if not api_key:
    st.error("API key is missing.")
    st.stop()

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF Pitch Deck",
    type=["pdf"]
)

# -----------------------------
# PDF to Image Conversion Function
# -----------------------------
def pdf_to_base64_images(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    images = []
    # Process all pages
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        images.append(img_b64)
    return images

# -----------------------------
# Analyze Button
# -----------------------------
if uploaded_file is not None:

    if st.button("Run Due Diligence Analysis"):

        with st.spinner("Converting PDF slides to images..."):
            try:
                images = pdf_to_base64_images(uploaded_file)
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
                st.stop()

        if len(images) == 0:
            st.error("Could not extract any pages from this PDF.")
            st.stop()

        system_prompt = """
You are a ruthless, forensic VC Principal at Peak XV. You are reviewing a pitch deck.

CRITICAL INSTRUCTION: Look closely at the charts, tables, and numbers in the images. You MUST extract the specific numbers the founders are presenting (e.g., "The deck shows $50k MRR and 5% monthly churn"). Do not use generic assumptions; use the numbers from the deck. If the deck is missing the data, state what is missing.

Do not use LaTeX or weird math formatting. Use plain text and standard Markdown numbers (e.g., $50,000 or 5%).

Return your response ONLY in Markdown using exactly these headers:

## 1. The Unvarnished Truth
Strip away the marketing fluff. What is the core transaction? Who pays whom? What is the "wedge" product, and what is the ultimate end-game?

## 2. Forensic Market Sizing
Extract their stated TAM/SAM/SOM from the slides. Do not accept a top-down approach (e.g., "1% of a $10B market"). Calculate a bottom-up TAM based on realistic pricing and target user volumes. State the realistic SOM they can capture in Years 1-3.

## 3. Unit Economics Stress Test
Extract their actual MRR, ARR, CAC, LTV, and Churn from the slides. If they didn't include them, say "Not provided in deck." Then, build a hypothetical P&L using standard industry benchmarks to see if their business model actually works. At what scale do they break even? 

## 4. Top 3 Fatal Flaws
Identify the three most likely reasons this company will fail. Focus on distribution bottlenecks, regulatory risk, or whether a Big Tech company (Google, Meta, Amazon) can copy this as a side project. Be hyper-specific to this company's actual product.

## 5. The "Sweat" Diligence Questions
Provide three highly numerical, aggressive questions. Use the numbers from the deck to trap the founder. Example: "Your deck shows $50k MRR but a $500 CAC. At your current growth rate, how do you survive the cash flow gap until you hit $500k MRR?"
"""

        user_prompt = "Analyze the following startup pitch deck slides provided as images."

        try:
            with st.spinner("Running GPT-4o Vision Analysis on the slides..."):
                client = OpenAI(api_key=api_key)

                # Construct the message with text + multiple images
                content = [{"type": "text", "text": user_prompt}]
                for img_b64 in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    })

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}
                    ],
                    temperature=0.3,
                )

                report = response.choices[0].message.content

            st.markdown(
                '<div class="report-container">',
                unsafe_allow_html=True
            )

            st.markdown(report)

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"OpenAI API Error: {e}")
