import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_ibm import WatsonxLLM
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import json
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyPDF2 import PdfReader # <-- Added for PDF support
import zipfile # <-- Added for ZIP file creation
from tenacity import retry, stop_after_attempt, wait_exponential # <-- Added for network retries

# --- 1. SETUP & CONFIGURATION ---

# Load environment variables from .env file (for API keys)
load_dotenv()

# Get credentials from environment variables
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", None)
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", None)

# --- Helper function to display errors and stop if credentials are not set ---
def check_credentials():
    if not WATSONX_API_KEY or not PROJECT_ID:
        st.error("🚨 Missing Credentials! Please set WATSONX_API_KEY and WATSONX_PROJECT_ID in your .env file.")
        st.stop()

# --- Function to initialize the LLM, wrapped with caching ---
@st.cache_resource
def initialize_llm(model_id):
    """Initializes and returns the Watsonx LLM instance. Cached for performance."""
    check_credentials()
    st.info(f"Initializing model: {model_id}...")
    llm = WatsonxLLM(
        model_id=model_id,
        url="https://au-syd.ml.cloud.ibm.com", 
        project_id=PROJECT_ID,
        apikey=WATSONX_API_KEY,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 2048,
            "temperature": 0.1,
            "repetition_penalty": 1.05
        }
    )
    st.info("Model initialized and ready.")
    return llm

# --- 2. HELPER FUNCTIONS ---

def get_text_from_file(uploaded_file):
    """Extracts text from an uploaded file (supports .txt and .pdf)."""
    text = ""
    file_name = uploaded_file.name
    if file_name.endswith('.txt'):
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        text = stringio.read()
    elif file_name.endswith('.pdf'):
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_summarization_chain(_llm, chain_type):
    """Prepares the LangChain summarization chain based on the selected type."""
    
    # **MODIFIED**: Prompts are now even more direct to prevent the model from creating unwanted sub-keys.
    if chain_type == "map_reduce":
        map_prompt_template = """
Your task is to analyze the following chunk of a court judgment and extract key information into a structured, flat JSON format.
**IMPORTANT**: Your output MUST be a single JSON object. The top-level keys must be the ones specified (e.g., "Court", "Case No"). Do not nest the entire summary inside a parent key like "summary" or "judgment_details". Generate ONLY the JSON object and nothing else.

Analyze the text below:
---
{text}
---

Generate the flat JSON output now:
"""
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

        combine_prompt_template = """
You have been provided with a series of partial JSON summaries from a single court judgment. Your task is to consolidate these into one final, comprehensive, and flat JSON object.
**IMPORTANT**: Your output MUST be the final, combined JSON object. The top-level keys must be the ones specified. Do not nest the result inside any other key. Remove any duplicate information.

The partial summaries are below:
---
{text}
---

Combine these into a single, clean, flat JSON output now:
"""
        combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
        
        return load_summarize_chain(llm=_llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt, verbose=True)
    
    elif chain_type == "refine":
        question_prompt_template = """
Your task is to analyze the following text from a court judgment and extract key information into a structured, flat JSON format.
**IMPORTANT**: Your output MUST be a single JSON object. The top-level keys must be the ones specified (e.g., "Court", "Case No"). Do not nest the entire summary inside a parent key like "summary" or "judgment_details". Generate ONLY the JSON object and nothing else.

The text is below:
---
{text}
---

Generate the flat JSON output now:
"""
        question_prompt = PromptTemplate(template=question_prompt_template, input_variables=["text"])

        refine_prompt_template = """
You are refining an existing JSON summary with new information. Here is the existing summary:
---
{existing_answer}
---
Here is the new context from the judgment. Use it to update and complete the existing summary.
**IMPORTANT**: Your output MUST be the final, refined, flat JSON object. The top-level keys must be the ones specified. Do not nest the result inside any other key.

Refine the original JSON summary using the new context and output the result now:
"""
        refine_prompt = PromptTemplate(template=refine_prompt_template, input_variables=["existing_answer", "text"])

        return load_summarize_chain(llm=_llm, chain_type="refine", question_prompt=question_prompt, refine_prompt=refine_prompt, verbose=True)

def extract_json_from_string(text):
    """Finds and parses a JSON object from a string."""
    try:
        start_index = text.find('{')
        end_index = text.rfind('}') + 1
        if start_index != -1 and end_index != 0:
            json_str = text[start_index:end_index]
            return json.loads(json_str)
        else:
            return {"error": "Could not find a JSON object in the model's response.", "raw_output": text}
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON from the model's response.", "raw_output": text}

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def process_file(uploaded_file, chain):
    """Worker function for parallel processing. Now with retry logic."""
    try:
        judgment_text = get_text_from_file(uploaded_file)
        if not judgment_text:
            return uploaded_file.name, {"error": "Could not extract any text from the file."}
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
        docs = text_splitter.create_documents([judgment_text])
        
        response_text = chain.run(docs)
        json_data = extract_json_from_string(response_text)
        return uploaded_file.name, json_data
    except Exception as e:
        return uploaded_file.name, {"error": f"An unexpected error occurred after multiple retries: {e}"}

# --- 3. MAIN APPLICATION LOGIC ---

def main():
    st.set_page_config(page_title="Judgment Summary Generator", layout="wide")
    st.header("⚖️ Structured Summary Generator (IBM Watsonx)")
    st.markdown("Upload court judgment files (`.txt` or `.pdf`) and generate a structured JSON summary for each.")

    check_credentials()

    if 'results' not in st.session_state:
        st.session_state.results = {}

    with st.sidebar:
        st.subheader("⚙️ Settings")
        model_options = ['mistralai/mistral-large', 'ibm/granite-13b-instruct-v2', 'ibm/granite-8b-code-instruct', 'ibm/granite-3-8b-instruct', 'ibm/granite-3-2b-instruct', 'meta-llama/llama-3-2-11b-vision-instruct', 'meta-llama/llama-3-2-90b-vision-instruct', 'meta-llama/llama-guard-3-11b-vision']
        selected_model = st.selectbox("Choose a Model:", model_options)
        chain_type_options = ["map_reduce", "refine"]
        selected_chain_type = st.selectbox("Choose a Strategy:", chain_type_options)
        
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload Judgment Files", 
            type=["txt", "pdf"],
            accept_multiple_files=True
        )

        if st.button("Clear All Results"):
            st.session_state.results = {}
            st.rerun()

    if uploaded_files:
        files_to_process = [f for f in uploaded_files if f.name not in st.session_state.results]
        if st.button(f"Generate Summaries for {len(files_to_process)} Remaining File(s)"):
            if files_to_process:
                llm = initialize_llm(selected_model)
                chain = get_summarization_chain(llm, selected_chain_type)
                
                progress_bar = st.progress(0, text="Starting parallel processing...")
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(process_file, f, chain): f for f in files_to_process}
                    completed_count = 0
                    total_files = len(files_to_process)
                    for future in as_completed(futures):
                        filename, result_data = future.result()
                        st.session_state.results[filename] = result_data
                        completed_count += 1
                        progress_bar.progress(completed_count / total_files, text=f"Processed {completed_count}/{total_files}: {filename}")
                progress_bar.empty()
                st.rerun()

    if st.session_state.results:
        st.subheader("Generated Summaries")
        
        successful_summaries = {k: v for k, v in st.session_state.results.items() if "error" not in v}
        if successful_summaries:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for filename, data in successful_summaries.items():
                    json_filename = f"{os.path.splitext(filename)[0]}.json"
                    zip_file.writestr(json_filename, json.dumps(data, indent=4))
            
            st.download_button(
                label="📥 Download All Summaries as ZIP",
                data=zip_buffer.getvalue(),
                file_name="summaries.zip",
                mime="application/zip",
            )
        
        for filename, result_data in st.session_state.results.items():
            with st.expander(f"📄 Summary for: {filename}", expanded=True):
                if "error" in result_data:
                    st.error(result_data["error"])
                    if "raw_output" in result_data:
                        st.text_area("Model's Raw Output:", result_data["raw_output"], height=150)
                else:
                    st.json(result_data)
                    json_string_for_download = json.dumps(result_data, indent=4)
                    st.download_button(
                        label="Download JSON",
                        data=json_string_for_download,
                        file_name=f"{os.path.splitext(filename)[0]}.json",
                        mime="application/json",
                        key=f"download_{filename}"
                    )
    elif not uploaded_files:

if __name__ == "__main__":
    main()
