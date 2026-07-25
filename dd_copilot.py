import streamlit as st
from PyPDF2 import PdfReader
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
    '<div class="subtitle">Upload a startup pitch deck or investment memo and receive an AI-powered VC diligence report.</div>',
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
    "Upload a PDF",
    type=["pdf"]
)

# -----------------------------
# PDF Extraction Function
# -----------------------------
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()

# -----------------------------
# Analyze Button
# -----------------------------
if uploaded_file is not None:

    if st.button("Run Due Diligence Analysis"):

        with st.spinner("Extracting PDF text..."):
            try:
                extracted_text = extract_text(uploaded_file)
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
                st.stop()

        if len(extracted_text) < 100:
            st.error("Could not extract text (might be image-based PDF)")
            st.stop()

        system_prompt = """
You are a ruthless, top-quartile VC Principal at Peak XV Partners. You are known for killing deals because founders cannot pass your stress tests. You do not get impressed by buzzwords. You care about distribution, unit economics, and defensible moats.

Analyze the provided pitch deck text using rigorous VC frameworks. Do not just summarize the deck; tear it apart.

Return your response ONLY in Markdown using exactly these headers:

## 1. The Unvarnished Truth
Provide one paragraph on what the company *actually* does. What is their initial "wedge" into the market, and what is the ultimate vision? Strip away the marketing fluff.

## 2. Market Sizing Reality Check
Critique their TAM/SAM/SOM. Is it a top-down "1% of a huge market" fallacy, or is it a bottom-up calculation based on actual pricing and target customer count? State clearly if their market sizing is lazy or realistic.

## 3. Unit Economics & Moat Analysis
Evaluate their business model. What are the likely Customer Acquisition Costs (CAC) vs. Lifetime Value (LTV)? Do they have a defensible moat (network effects, switching costs, proprietary data), or is this a feature that a larger competitor (like Google or Amazon) could clone in a weekend?

## 4. Top 3 Fatal Flaws
Identify the three biggest reasons this startup will die. Focus on missing financials, unrealistic conversion rates, execution risks, or regulatory hurdles. Be highly critical.

## 5. The "Sweat" Diligence Questions
Provide exactly three aggressive, probing questions that will make the founder sweat in the room. Do not ask generic questions; ask questions that expose the holes in their deck (e.g., "You project 5% conversion, but your CAC is $50—how do you survive if conversion is 0.5%?").
"""

        user_prompt = f"""
Analyze the following startup document.

Document:

{extracted_text}
"""

        try:
            with st.spinner("Running AI Due Diligence..."):

                client = OpenAI(api_key=api_key)

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
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
