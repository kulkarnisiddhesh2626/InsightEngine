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
    .stButton > button { background-color: #D35400 !important; color: #FFFFFF !important; border-radius: 8px !important; font-weight: bold; width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "vector_ready" not in st.session_state: st.session_state["vector_ready"] = False
if "raw_text" not in st.session_state: st.session_state["raw_text"] = "No data loaded yet."
if "last_response" not in st.session_state: st.session_state["last_response"] = "No queries yet."
if "processed_source" not in st.session_state: st.session_state["processed_source"] = None
if "chat_count" not in st.session_state: st.session_state["chat_count"] = 1

# --- SAMPLE DATA SUMMARIES (For Laymen) ---
sample_info = {
    "Tesla 2023 Annual Report": "This is a real financial document (Form 10-K) detailing Tesla's revenue, vehicle deliveries, and corporate risks for the year 2023. \n\n**Best for:** Testing the AI's ability to find specific numbers, summarize financial strategies, and extract company risks.",
    "AI Act Final Draft": "This is the comprehensive legal framework proposed by the European Union to regulate Artificial Intelligence. \n\n**Best for:** Testing how the AI handles dense legal jargon, compliance rules, and specific definitions.",
    "Bitcoin Whitepaper": "This is the original 9-page academic paper written by Satoshi Nakamoto in 2008 that introduced cryptocurrency to the world. \n\n**Best for:** Testing the AI on technical concepts, cryptographic explanations, and peer-to-peer network theories.",
    "UN Climate Report": "A dense scientific summary created by the IPCC for global policymakers regarding the current state of global warming. \n\n**Best for:** Testing the AI's ability to synthesize scientific data, climate projections, and policy suggestions."
}

# --- HEADER LAYOUT ---
header_col, g_data_col, exp_col, auth_col = st.columns([0.4, 0.2, 0.2, 0.2])

with header_col:
    st.title("💠 InsightEngine")
    st.caption("Your Intelligent Document Analysis Assistant")

# Helper function to process data automatically
def process_pdf(source_path, source_name):
    with st.spinner(f"Processing '{source_name}'... Please wait."):
        loader = PyPDFLoader(source_path)
        data = loader.load()
        st.session_state["raw_text"] = "\n".join([p.page_content for p in data])
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("faiss_index")
        
        st.session_state["vector_ready"] = True
        st.session_state["processed_source"] = source_name

with g_data_col:
    with st.expander("Get Data", help="Upload your own file or pick a sample to start analyzing."):
        st.markdown("**1. Upload Your Own**")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
        
        st.markdown("**2. Sample Datasets**")
        sample_choice = st.selectbox("Choose a sample:", ["None", "Tesla 2023 Annual Report", "AI Act Final Draft", "Bitcoin Whitepaper", "UN Climate Report"], label_visibility="collapsed")
        
        samples = {
            "Tesla 2023 Annual Report": "samples/tesla_2023.pdf",
            "AI Act Final Draft": "samples/ai_act.pdf",
            "Bitcoin Whitepaper": "samples/bitcoin.pdf",
            "UN Climate Report": "samples/un_climate.pdf"
        }
        
        # 1. Feature: About this data
        if sample_choice != "None":
            with st.popover("About this Data"):
                st.markdown(f"**{sample_choice}**")
                st.write(sample_info[sample_choice])

        # 2. Feature: Auto-process data
        current_selection = uploaded_file.name if uploaded_file else sample_choice
        if current_selection != "None" and current_selection != st.session_state["processed_source"]:
            if uploaded_file:
                with open("temp.pdf", "wb") as f: f.write(uploaded_file.getbuffer())
                process_pdf("temp.pdf", uploaded_file.name)
            else:
                process_pdf(samples[sample_choice], sample_choice)
            st.rerun()

        st.divider()
        # 3. Feature: Data Preview
        with st.popover("Preview Loaded Data", help="Click to see the raw text the AI is currently reading."):
            if st.session_state["vector_ready"]:
                st.success(f"Currently analyzing: {st.session_state['processed_source']}")
                st.text_area("Document Text Preview", st.session_state["raw_text"][:2500] + "...\n\n(Text truncated for preview)", height=300)
            else:
                st.info("Please select or upload data first.")

with exp_col:
    with st.expander("Export"):
        st.download_button("Download Last Response", st.session_state["last_response"], file_name="insight_export.txt", help="Save the AI's last answer to your computer as a text file.")
        if st.button("Reset System", help="Clear all data and start over."):
            st.session_state.clear()
            st.rerun()

with auth_col:
    with st.expander("Author", help="Information about the creator."):
        st.markdown("**Created by:** Siddhesh Kulkarni")
        st.markdown("**Email:** [kulkarnisiddhesh2626@gmail.com](mailto:kulkarnisiddhesh2626@gmail.com)")
        st.markdown("**LinkedIn:** [Profile Link](https://www.linkedin.com/in/siddhesh-kulkarni-b2a600207/)")
        st.markdown("**GitHub:** [Profile Link](https://github.com/kulkarnisiddhesh2626)")

st.divider()

# --- DOCUMENTATION & PARAMETERS (Side-by-Side Layout) ---
doc_col, param_col = st.columns(2)

with doc_col:
    with st.expander("Documentation & Architecture", help="Learn how this application actually works behind the scenes."):
        tab_about, tab_arch = st.tabs(["About InsightEngine", "How It Works (Architecture)"])
        with tab_about:
            st.markdown("""
            **What is InsightEngine?**
            Imagine giving an AI an "open book test." Standard AI bots only know what they were trained on months ago. InsightEngine uses a technique called **RAG** (Retrieval-Augmented Generation). 
            
            This means you upload a document, and the AI actively reads it, finds the exact paragraphs related to your question, and uses them to write a factual answer. 
            
            **Why is this useful?**
            * **No Hallucinations:** The AI is forced to rely on your document, not its imagination.
            * **Massive Time Saver:** Instead of reading a 100-page report, you can just ask the bot to summarize the key points.
            """)
        with tab_arch:
            st.markdown("""
            **The 5-Step Brain of the Bot:**
            1. **Reading (Ingestion):** The system takes your PDF and extracts all the raw text.
            2. **Slicing (Chunking):** Since AI can't read a whole book in one glance, we slice the text into small, readable paragraphs.
            3. **Translating (Embeddings):** The system translates these paragraphs into numbers (vectors) that the computer can easily sort and search.
            4. **Filing Cabinet (FAISS):** These numbers are stored in a highly efficient local database.
            5. **Thinking (LLM):** When you ask a question, the system finds the most relevant "slices" from the filing cabinet and sends them to a massive cloud AI (Groq Llama 3) to construct a human-like answer.
            """)

with param_col:
    with st.expander("LLM Parameters (Settings)", help="Tweak the AI's behavior to suit your needs."):
        st.markdown("**1. Temperature (Creativity Dial)**")
        ai_temp = st.slider("Select Temperature", 0.0, 1.0, 0.1, label_visibility="collapsed")
        st.markdown("""
        *Think of this as the AI's imagination dial.*
        * **Low (0.0 - 0.2):** Very strict, factual, and robotic. Perfect for legal or financial documents where you want exact facts.
        * **High (0.7 - 1.0):** Creative and conversational. Better for brainstorming or writing stories.
        """)
        
        st.divider()
        st.markdown("**2. AI Brain (Model Selection)**")
        model_choice = st.selectbox("Select Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"], label_visibility="collapsed")
        st.markdown("""
        *Choose how powerful you want the AI to be.*
        * **8b-instant:** A smaller, lightning-fast brain. Great for simple summaries and quick questions.
        * **70b-versatile:** A massive, highly intelligent brain. Use this for complex reasoning, deep analysis, and difficult questions. (Slightly slower).
        """)

# --- PROCESSING LOGIC (Secrets Check) ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Missing GROQ_API_KEY in Secrets.")
    st.stop()

# --- MULTI-CHAT INTERFACE ---
st.markdown("### Workspace")

# Feature: Dynamically add new chats
col_chat_title, col_new_chat = st.columns([0.8, 0.2])
with col_new_chat:
    if st.button("➕ New Chat Window", help="Open a fresh chat tab without deleting the current ones."):
        st.session_state["chat_count"] += 1
        st.rerun()

# Generate tabs dynamically based on user clicks
chat_tabs = st.tabs([f"Conversation {i+1}" for i in range(st.session_state["chat_count"])])

def run_query(query):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=4)
    context = "\n\n".join([d.page_content for d in docs])
    
    llm = ChatGroq(model_name=model_choice, temperature=ai_temp, groq_api_key=GROQ_API_KEY)
    response = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
    st.session_state["last_response"] = response.content
    return response.content

# Populate all tabs
for i, tab in enumerate(chat_tabs):
    with tab:
        if st.session_state["vector_ready"]:
            # Unique keys for each input box so Streamlit doesn't get confused
            user_q = st.text_input("Ask a question about the document:", key=f"chat_input_{i}")
            if user_q:
                with st.spinner("Synthesizing answer..."):
                    result = run_query(user_q)
                    st.success(result)
        else:
            st.info("👈 Please select a sample dataset or upload a file from the 'Get Data' menu in the top right to start chatting.")
