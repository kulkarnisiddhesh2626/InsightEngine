import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="InsightEngine Elite", page_icon="💠", layout="wide")

# --- CUSTOM CSS: PREMIUM UI ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #FFFDE4 0%, #FFE5B4 100%) !important; }
    [data-testid="stSidebar"] { display: none; }
    .stMarkdown, p, span, div, label, .stText { color: #2F4F4F !important; }
    h1, h2, h3 { color: #D35400 !important; font-weight: 800 !important; }
    
    /* Buttons */
    .stButton > button {
        background-color: #D35400 !important; 
        color: #FFFFFF !important;      
        border-radius: 8px !important;
        font-weight: bold;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "vector_ready" not in st.session_state: st.session_state["vector_ready"] = False
if "raw_text" not in st.session_state: st.session_state["raw_text"] = ""
if "last_response" not in st.session_state: st.session_state["last_response"] = "No queries yet."

# --- HEADER LAYOUT (TITLE & 3 TOP-RIGHT DROPDOWNS) ---
header_col, g_data_col, exp_col, auth_col = st.columns([0.4, 0.2, 0.2, 0.2])

with header_col:
    st.title("💠 InsightEngine")

with g_data_col:
    with st.expander("📥 Get Data"):
        st.markdown("**1. Upload Your Own**")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
        
        st.markdown("**2. Sample Datasets**")
        sample_choice = st.selectbox("Choose a sample:", ["None", "Tesla 2023 Annual Report", "AI Act Final Draft", "Bitcoin Whitepaper", "UN Climate Report"])
        
        # Logic for Sample Data 
        samples = {
            "Tesla 2023 Annual Report": "samples/tesla_2023.pdf",
            "AI Act Final Draft": "samples/ai_act.pdf",
            "Bitcoin Whitepaper": "samples/bitcoin.pdf",
            "UN Climate Report": "samples/un_climate.pdf"
        }
        
        process_button = st.button("🚀 Initialize Index")
        
        st.divider()
        with st.popover("👀 Data Preview"):
            st.text_area("Live Stream", st.session_state["raw_text"][:1000], height=200)

with exp_col:
    with st.expander("📤 Export"):
        st.download_button("Download Last Response", st.session_state["last_response"], file_name="insight_export.txt")
        if st.button("🚨 Reset System"):
            st.session_state.clear()
            st.rerun()

with auth_col:
    with st.expander("👨‍💻 Author"):
        st.markdown("**Created by:** Siddhesh Kulkarni")
        st.markdown("📧 [kulkarnisiddhesh2626@gmail.com](mailto:kulkarnisiddhesh2626@gmail.com)")
        st.markdown("🔗 [LinkedIn Profile](https://www.linkedin.com/in/siddhesh-kulkarni-b2a600207/)")
        st.markdown("💻 [GitHub](https://github.com/kulkarnisiddhesh2626)")

# --- MAIN DOCUMENTATION DROPDOWN ---
with st.expander("📘 Documentation & Architecture"):
    tab_about, tab_arch = st.tabs(["About InsightEngine", "Neural Architecture"])
    with tab_about:
        st.markdown("""
        InsightEngine uses **Agentic RAG** to solve complex information retrieval tasks.
        * **Legal:** Instant clause extraction.
        * **Finance:** Financial statement analysis.
        * **Academia:** Literature review synthesis.
        """)
    with tab_arch:
        st.markdown("1. PDF Ingestion → 2. Recursive Chunking → 3. HF Embeddings → 4. FAISS Index → 5. Groq Llama Reasoning")

st.divider()

# --- LLM PARAMETERS (Top Control) ---
with st.expander("⚙️ LLM Parameters"):
    c1, c2 = st.columns(2)
    with c1:
        ai_temp = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.1)
        st.caption("Lower = Factual, Higher = Creative")
    with c2:
        # UPDATED: Replaced deprecated models with current active Groq models
        model_choice = st.selectbox(
            "Model", 
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        )
        st.caption("70b is smarter but has lower rate limits.")

# --- PROCESSING LOGIC ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Missing GROQ_API_KEY in Secrets.")
    st.stop()

# Helper to process file
def process_pdf(source):
    with st.status("💠 Processing Knowledge...", expanded=True) as status:
        loader = PyPDFLoader(source)
        data = loader.load()
        st.session_state["raw_text"] = "\n".join([p.page_content for p in data])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("faiss_index")
        st.session_state["vector_ready"] = True
        status.update(label="✅ Ready", state="complete")

if process_button:
    if uploaded_file:
        with open("temp.pdf", "wb") as f: f.write(uploaded_file.getbuffer())
        process_pdf("temp.pdf")
    elif sample_choice != "None":
        process_pdf(samples[sample_choice])

# --- MULTI-CHAT INTERFACE ---
st.markdown("### 🧠 Workspace")
chat_tab1, chat_tab2, chat_tab3 = st.tabs(["Chat Sequence Alpha", "Chat Sequence Beta", "Comparison View"])

def run_query(query):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=4)
    context = "\n\n".join([d.page_content for d in docs])
    
    # This now uses the updated model_choice from the dropdown
    llm = ChatGroq(model_name=model_choice, temperature=ai_temp, groq_api_key=GROQ_API_KEY)
    response = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
    st.session_state["last_response"] = response.content
    return response.content

with chat_tab1:
    if st.session_state["vector_ready"]:
        q1 = st.text_input("Query Alpha:", key="q1")
        if q1: st.success(run_query(q1))
    else: st.info("Initialize data to start.")

with chat_tab2:
    if st.session_state["vector_ready"]:
        q2 = st.text_input("Query Beta:", key="q2")
        if q2: st.info(run_query(q2))
    else: st.info("Initialize data to start.")

with chat_tab3:
    st.markdown("Compare results from Alpha and Beta sequences here.")
