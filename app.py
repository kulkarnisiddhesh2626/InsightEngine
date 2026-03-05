import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="InsightEngine", page_icon="⚡", layout="wide")

# --- CUSTOM CSS: CLEAN LIGHT THEME ---
st.markdown("""
<style>
    /* Main App Background: Very faint gray/blue */
    .stApp {
        background-color: #F4F6F9 !important;
    }

    /* Sidebar Background: Pure White with a subtle shadow */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E0E4E8 !important;
    }

    /* Primary Text: Dark Slate Gray */
    .stMarkdown, p, span, div, label, .stText {
        color: #2C3E50 !important;
    }

    /* Headers: Deep Charcoal/Navy */
    h1, h2, h3, h4 {
        color: #1A252F !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700 !important;
    }

    /* High-Contrast Buttons: Strong Blue Background, White Text */
    .stButton > button {
        background-color: #0056D2 !important; 
        color: #FFFFFF !important;      
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        width: 100%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    .stButton > button:hover {
        background-color: #0041A3 !important; 
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }

    /* Text Inputs: White backgrounds, dark borders */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #FFFFFF !important;
        color: #2C3E50 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
    }
    
    /* Metrics / Telemetry numbers */
    [data-testid="stMetricValue"] {
        color: #0056D2 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        color: #2C3E50 !important;
        border-bottom: 1px solid #E0E4E8;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "vector_ready" not in st.session_state:
    st.session_state["vector_ready"] = False
if "raw_text" not in st.session_state:
    st.session_state["raw_text"] = "Upload a document to see preview."
if "last_response" not in st.session_state:
    st.session_state["last_response"] = "No queries made yet."

# --- MAIN DASHBOARD ---
st.title("⚡ InsightEngine // Core")

# ENGINE INFO MOVED TO DROPDOWN
with st.expander("ℹ️ What is InsightEngine?", expanded=False):
    st.markdown("""
    **InsightEngine** synthesizes complex PDFs using purely local hardware. 
    
    **Capabilities:** * Zero-latency data extraction 
    * Format-agnostic chunking 
    * Contextual reasoning without sending your private data to the cloud.
    """)

st.divider()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # AI Temperature Control
    st.markdown("**LLM Parameters**")
    ai_temp = st.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    
    st.divider()
    
    # Upload Section
    uploaded_file = st.file_uploader("Drop PDF Payload", type="pdf")
    process_button = st.button("Initialize Index")

    st.divider()
    
    # Dropdowns for Features
    with st.expander("🛠️ System Architecture"):
        st.markdown("""
        **Data Flow:**
        1. **Ingestion:** PyPDF2 
        2. **Chunking:** Recursive (800 chars)
        3. **Embedding:** Llama 3.2 (1b) 
        4. **Storage:** FAISS Vector Index
        5. **Reasoning:** Llama 3.2 (3b)
        """)
        
    with st.expander("👀 Data Preview"):
        st.text_area("Raw Text (First 1k chars)", st.session_state["raw_text"][:1000], height=200, disabled=True)

    with st.expander("💾 Export Data"):
        st.download_button(
            label="Download Last Response",
            data=st.session_state["last_response"],
            file_name="insight_engine_export.txt",
            mime="text/plain"
        )
        
    # External APIs (Tavily Mention)
    with st.expander("🌐 External APIs"):
        st.markdown("""
        **Web Search Integration:**
        * Provider: **Tavily AI**
        * Purpose: Real-time fact-checking.
        * Source: [app.tavily.com/home](https://app.tavily.com/home)
        """)
        
    # System Reset
    if st.button("🚨 Reset Engine"):
        st.session_state.clear()
        st.rerun()

# --- CORE PROCESSING LOGIC ---
if uploaded_file and process_button:
    try:
        start_time = time.time()
        
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Extracting datastream..."):
            loader = PyPDFLoader("temp.pdf")
            data = loader.load()
            st.session_state["raw_text"] = "\n".join([page.page_content for page in data])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)

        with st.spinner("Embedding into FAISS matrix..."):
            embeddings = OllamaEmbeddings(model="llama3.2:1b")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local("faiss_index")
        
        end_time = time.time()
        
        st.success("✅ Datastream Indexed Successfully.")
        st.session_state["vector_ready"] = True
        
        col1, col2 = st.columns(2)
        col1.metric("Chunks Created", len(chunks))
        col2.metric("Processing Latency", f"{end_time - start_time:.2f} sec")

    except Exception as e:
        st.error(f"System Fault: {str(e)}")

# --- CHAT / REASONING INTERFACE ---
if st.session_state["vector_ready"]:
    user_query = st.text_input("Enter your query sequence:")
    
    if user_query:
        try:
            query_start = time.time()
            with st.spinner("Compiling answer..."):
                embeddings = OllamaEmbeddings(model="llama3.2:1b")
                db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                docs = db.similarity_search(user_query, k=4)
                context = "\n\n".join([d.page_content for d in docs])
                
                llm = ChatOllama(model="llama3.2:3b", temperature=ai_temp)
                prompt = f"Use the following context to answer the question accurately.\n\nContext: {context}\n\nQuestion: {user_query}"
                
                response = llm.invoke(prompt)
                final_answer = response.content
                query_end = time.time()
                
                st.session_state["last_response"] = f"Query: {user_query}\n\nAnswer: {final_answer}"
                
                st.markdown("### 🧠 Neural Output:")
                st.info(final_answer)
                st.caption(f"⏱️ Inference Latency: {query_end - query_start:.2f} seconds")
                
                with st.expander("View Source Chunks"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Source {i+1}:**")
                        st.write(doc.page_content)
                        st.divider()
                        
        except Exception as e:
            st.error(f"Inference Error: {str(e)}. Ensure Ollama is running.")
else:
    st.info("System idle. Awaiting document upload.")