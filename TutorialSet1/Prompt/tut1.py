# app.py
import streamlit as st
from huggingface_hub import InferenceApi
from dotenv import load_dotenv
import os
import requests

load_dotenv()
st.set_page_config(page_title="Research Assistant", layout="centered")
st.title("Research Assistant — Streamlit + Hugging Face Inference API (working)")

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_API_KEY")
if not HF_TOKEN:
    st.error("Hugging Face token not found. Put HUGGINGFACEHUB_API_TOKEN in a .env file.")
    st.stop()

# Only include models that are known to work with HF Inference API
MODEL_CHOICES = {
    "GPT-2 (text-generation)": ("gpt2", "text-generation"),
    "facebook/bart-large-cnn (summarization)": ("facebook/bart-large-cnn", "summarization"),
}

model_label = st.selectbox("Model", list(MODEL_CHOICES.keys()), index=1)
repo_id, task = MODEL_CHOICES[model_label]
st.markdown(f"**Repo:** `{repo_id}` — **Task:** `{task}`")

user_input = st.text_area("Enter text or question:", height=180)
col1, col2 = st.columns([1, 1])
with col1:
    max_tokens = st.number_input("Max new tokens", min_value=10, max_value=1024, value=200, step=10)
with col2:
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

def extract_text(resp):
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for k in ("generated_text", "summary_text", "data", "text"):
            if k in resp:
                return resp[k] if isinstance(resp[k], str) else str(resp[k])
        return str(resp)
    if isinstance(resp, list) and resp:
        return extract_text(resp[0])
    return str(resp)

def call_fallback_http(repo_id, token, prompt, params):
    url = f"https://api-inference.huggingface.co/models/{repo_id}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt, "parameters": params}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    try:
        return r.json()
    except ValueError:
        return r.text

if st.button("Generate") and user_input.strip():
    prompt = user_input.strip()
    # Add instruction depending on task
    if task == "summarization":
        prompt = "summarize: " + prompt

    params = {"max_new_tokens": int(max_tokens), "temperature": float(temperature)}

    with st.spinner("Generating..."):
        try:
            inf = InferenceApi(repo_id=repo_id, token=HF_TOKEN, task=task)
            resp = inf.call(inputs=prompt, parameters=params)
            text = extract_text(resp).strip() or "(no output)"
            st.subheader("Result")
            st.write(text)
            st.session_state["last"] = text
        except Exception as e1:
            st.warning(f"InferenceApi failed: {e1}\nTrying HTTP fallback...")
            try:
                resp2 = call_fallback_http(repo_id, HF_TOKEN, prompt, params)
                text2 = extract_text(resp2).strip() or "(no output)"
                st.subheader("Result (fallback)")
                st.write(text2)
                st.session_state["last"] = text2
            except Exception as e2:
                st.error(f"HTTP fallback also failed: {e2}")

if "last" in st.session_state:
    with st.expander("Last result"):
        st.write(st.session_state["last"])
