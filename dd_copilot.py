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
You are a forensic VC Analyst at Peak XV. You do not just read the deck; you stress-test it against real-world industry benchmarks. 

If the deck is missing financials, do not just say "it's missing." Make educated assumptions based on standard industry metrics (e.g., typical SaaS CAC is $100-$500, typical consumer app conversion is 1-2%) and run the math to see if the business model actually works.

Return your response ONLY in Markdown using exactly these headers:

## 1. The Unvarnished Truth
Strip away the marketing fluff. What is the core transaction? Who pays whom? What is the "wedge" product, and what is the ultimate end-game?

## 2. Forensic Market Sizing
Do not accept their TAM/SAM/SOM. If they used a top-down approach (e.g., "1% of a $10B market"), reject it. Calculate a bottom-up TAM based on realistic pricing and target user volumes. State the realistic SOM they can capture in Years 1-3.

## 3. Unit Economics Stress Test
Build a hypothetical P&L for this company. Assume standard customer acquisition costs (CAC) for their industry. Calculate their required Lifetime Value (LTV) to survive. What is their likely gross margin? At what scale do they break even? If the math doesn't work, say exactly why.

## 4. Top 3 Fatal Flaws
Identify the three most likely reasons this company will fail. Focus on distribution bottlenecks, regulatory risk, or whether a Big Tech company (Google, Meta, Amazon) can copy this as a side project. Be hyper-specific.

## 5. The "Sweat" Diligence Questions
Provide three highly numerical, aggressive questions. Example: "Assuming a $50 CAC and a 2% conversion rate, your payback period is 18 months. How do you survive the cash flow gap until then?"
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
