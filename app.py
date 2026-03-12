import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="InsightEngine", page_icon="💠", layout="wide")

# --- CUSTOM CSS: THE "SAFFRON & SLATE" THEME ---
st.markdown("""
<style>
    /* Gradient Background: Light Yellow to Light Orange */
    .stApp {
        background: linear-gradient(135deg, #FFFDE4 0%, #FFE5B4 100%) !important;
    }

    /* Top-Nav Simulation (Removing Sidebar padding) */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Global Text: Dark Slate */
    .stMarkdown, p, span, div, label, .stText {
        color: #2F4F4F !important;
    }

    /* Professional Headers */
    h1, h2, h3 {
        color: #D35400 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 800 !important;
    }

    /* Export Button Positioning (Top Right) */
    .export-container {
        position: absolute;
        top: 0px;
        right: 0px;
        z-index: 999;
    }

    /* High-Contrast Professional Buttons */
    .stButton > button {
        background-color: #D35400 !important; 
        color: #FFFFFF !important;      
        border-radius: 8px !important;
        border: none !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "vector_ready" not in st.session_state: st.session_state["vector_ready"] = False
if "raw_text" not in st.session_state: st.session_state["raw_text"] = ""
if "last_response" not in st.session_state: st.session_state["last_response"] = "No data yet."

# --- TOP BAR: EXPORT & BRANDING ---
col_title, col_export = st.columns([0.8, 0.2])
with col_title:
    st.title("💠 InsightEngine // Elite")
with col_export:
    st.download_button(
        label="📥 Export Last Intelligence",
        data=st.session_state["last_response"],
        file_name="insight_export.txt",
        mime="text/plain",
    )

# --- NEW TOP-DOWN NAVIGATION (SIMULATED SIDEBAR) ---
st.markdown("### 🛠️ Strategic Control Center")
with st.expander("📂 Control Panel & Configuration", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 💠 Model Tuning")
        ai_temp = st.slider("Temperature", 0.0, 1.0, 0.1)
        with st.popover("❔ How this works"):
            st.write("**Temperature** controls randomness.")
            st.write("* **0.1 (Strict):** Best for technical manuals, medical data, and legal PDFs.")
            st.write("* **0.8 (Creative):** Best for summarizing stories or brainstorming marketing copy.")
    
    with c2:
        st.markdown("#### 💠 Document Ingestion")
        uploaded_file = st.file_uploader("Upload PDF Payload", type="pdf", label_visibility="collapsed")
        process_button = st.button("🚀 Initialize Neural Index")
        
    with c3:
        st.markdown("#### 💠 System Guardrails")
        max_tokens = st.select_slider("Response Depth", options=[256, 512, 1024, 2048], value=1024)
        with st.popover("❔ Understanding Depth"):
            st.write("Limits the length of the AI's output to prevent 'rambling' and save API credits.")

# --- MIDDLE SECTION: INFORMATION HUB ---
tab1, tab2, tab3, tab4 = st.tabs(["📘 About", "🏗️ Architecture", "📝 Preview", "👨‍💻 Author"])

with tab1:
    st.markdown("""
    ### 💠 High-Fidelity Retrieval Augmented Generation (RAG)
    InsightEngine is a professional-grade intelligence tool designed to bridge the gap between static documents and actionable insights. Unlike standard LLMs, it uses **Semantic Chunking** to ensure the bot doesn't just read words, but understands context.
    
    **🎯 Strategic Usecases:**
    * **Legal Discovery:** Cross-reference clauses across 100+ page contracts in seconds.
    * **Academic Synthesis:** Analyze research papers for methodology vs. results instantly.
    * **Financial Audits:** Extract quarterly earnings trends from complex annual reports.
    * **Technical Support:** Convert dense hardware manuals into a searchable troubleshooting bot.
    """)

with tab2:
    st.markdown("### 🏗️ Advanced Neural Pipeline")
    
    st.markdown("""
    1. **Preprocessing:** PDF is loaded via `PyPDFLoader` and cleaned of artifacts.
    2. **Granular Chunking:** `RecursiveCharacterTextSplitter` creates overlapping segments (800 chars) to maintain narrative flow.
    3. **Vectorization:** `HuggingFace (all-MiniLM-L6-v2)` transforms text into 384-dimensional mathematical vectors.
    4. **Indexing:** `FAISS` (Facebook AI Similarity Search) builds a high-speed local vector database.
    5. **Synthesis:** `Llama-3-8b` via **Groq LPUs** performs high-speed reasoning on retrieved context.
    """)

with tab3:
    st.text_area("Live Data Stream", st.session_state["raw_text"][:2000], height=250, placeholder="Preview will appear after indexing...")

with tab4:
    st.markdown("### 💠 Author Intelligence")
    st.info("""
    **Name:** Siddhesh  
    **Email:** [Your Email ID]  
    **Professional Profile:** [LinkedIn Link]  
    *Expertise in Agentic RAG Systems and Local LLM Deployment.*
    """)

st.divider()

# --- PROCESSING LOGIC ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Missing GROQ_API_KEY in Secrets.")
    st.stop()

if uploaded_file and process_button:
    with st.status("💠 Building Intelligence Matrix...", expanded=True) as status:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write("Reading PDF...")
        loader = PyPDFLoader("temp.pdf")
        data = loader.load()
        st.session_state["raw_text"] = "\n".join([p.page_content for p in data])
        
        st.write("Chunking Text...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        
        st.write("Generating Embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("faiss_index")
        
        st.session_state["vector_ready"] = True
        status.update(label="✅ System Ready", state="complete")

# --- CHAT INTERFACE ---
if st.session_state["vector_ready"]:
    user_query = st.text_input("💠 Query the Knowledge Base:")
    if user_query:
        with st.spinner("💠 Synthesizing Answer..."):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = db.similarity_search(user_query, k=4)
            context = "\n\n".join([d.page_content for d in docs])
            
            llm = ChatGroq(model_name="llama3-8b-8192", temperature=ai_temp, groq_api_key=GROQ_API_KEY)
            prompt = f"System: You are an elite analyst. Use the context to answer precisely.\nContext: {context}\nQuestion: {user_query}"
            
            response = llm.invoke(prompt)
            st.session_state["last_response"] = f"Query: {user_query}\n\nAnswer: {response.content}"
            st.markdown("### 🧠 Result:")
            st.success(response.content)
else:
    st.info("💠 Upload a document and initialize the engine to begin.")
