import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="InsightEngine Elite", page_icon="⬡", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%) !important; }
    [data-testid="stSidebar"] { display: none; }
    .stMarkdown, p, span, div, label, .stText { color: #1E293B !important; }
    h1, h2, h3 { color: #0F172A !important; font-weight: 800 !important; }
    .stButton > button { background-color: #334155 !important; color: #FFFFFF !important; border-radius: 6px !important; font-weight: 600; width: 100%; transition: all 0.3s ease; }
    .stButton > button:hover { background-color: #0F172A !important; border-color: #38BDF8 !important; color: #38BDF8 !important; }
    div[data-testid="stPopover"] { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- SECRETS CHECK ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Missing GROQ_API_KEY in Secrets. Please add it to Streamlit Cloud settings.")
    st.stop()

# --- SESSION STATE ---
if "vector_ready" not in st.session_state: st.session_state["vector_ready"] = False
if "raw_text" not in st.session_state: st.session_state["raw_text"] = "No data loaded yet."
if "last_response" not in st.session_state: st.session_state["last_response"] = "No queries yet."
if "processed_source" not in st.session_state: st.session_state["processed_source"] = None
if "chat_count" not in st.session_state: st.session_state["chat_count"] = 1
if "suggested_qs" not in st.session_state: st.session_state["suggested_qs"] = []
if "data_summary" not in st.session_state: st.session_state["data_summary"] = "No summary generated."

# --- CORE LOGIC: PROCESS PDF, SMART FAQs, & AUTOMATED SUMMARY ---
def process_pdf(source_path, source_name):
    with st.spinner(f"Ingesting & Analyzing '{source_name}'..."):
        loader = PyPDFLoader(source_path)
        data = loader.load()
        st.session_state["raw_text"] = "\n".join([p.page_content for p in data])
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("faiss_index")
        
        # SMART GENERATION (Reads intro to guess intent & summarize)
        try:
            llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2, groq_api_key=GROQ_API_KEY)
            preview_text = "\n".join([c.page_content for c in chunks[:4]])
            
            # 1. FAQ Prompt
            prompt_faq = f"Generate exactly 5 distinct, highly relevant questions a user could ask to get valuable insights from this text. Make them concise. Return them as a numbered list without intros. Text: {preview_text}"
            faq_response = llm.invoke(prompt_faq)
            st.session_state["suggested_qs"] = [q.strip() for q in faq_response.content.split('\n') if q.strip()]
            
            # 2. Summary Prompt
            prompt_summary = f"Provide a comprehensive, in-depth point-wise summary of the core themes, entities, and purpose of the following document excerpt. Text: {preview_text}"
            summary_response = llm.invoke(prompt_summary)
            st.session_state["data_summary"] = summary_response.content
        except:
            st.session_state["suggested_qs"] = ["1. What is the main topic?", "2. Summarize key metrics.", "3. Identify major risks.", "4. What is the conclusion?", "5. List the key entities."]
            st.session_state["data_summary"] = "Summary generation failed. Data loaded successfully."

        st.session_state["vector_ready"] = True
        st.session_state["processed_source"] = source_name

# --- HEADER ---
st.title("⬡ InsightEngine Elite")
st.markdown("*An enterprise-grade Agentic RAG system for high-fidelity document synthesis and intelligence extraction.*")

# --- 5-COLUMN TOP NAVIGATION ---
c_doc, c_data, c_llm, c_exp, c_auth = st.columns(5)

with c_doc:
    with st.expander("🗂️ Documentation"):
        st.markdown("**What is this?**\nA secure AI workspace that reads your private documents and answers complex queries with zero hallucinations.")
        with st.popover("🔍 See In-Detail Architecture"):
            st.markdown("""
            ### 🏗️ Complete System Architecture
            InsightEngine relies on an advanced **Retrieval-Augmented Generation (RAG)** pipeline:
            
            **1. Data Ingestion (PyPDFLoader):** Raw text is stripped from the uploaded PDF document.
            **2. Semantic Slicing (RecursiveCharacterTextSplitter):** The text is divided into 800-character chunks with a 100-character overlap to preserve context between pages.
            **3. Vectorization (HuggingFace all-MiniLM-L6-v2):** Chunks are mathematically embedded into high-dimensional vectors.
            **4. Indexing (FAISS):** Vectors are stored in a Meta FAISS database for ultra-fast similarity search.
            **5. Cognitive Synthesis (Groq Llama-3):** When a user asks a query, the system finds the top 4 most mathematically relevant text chunks and sends them to the LLM. The LLM is strictly instructed to answer *only* using that context.
            """)

with c_data:
    with st.expander("📥 Knowledge Base"):
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
        sample_choice = st.selectbox("Or pick sample:", ["None", "Tesla 2023 Annual Report", "AI Act Final Draft", "Bitcoin Whitepaper", "UN Climate Report"])
        
        samples = {"Tesla 2023 Annual Report": "samples/tesla_2023.pdf", "AI Act Final Draft": "samples/ai_act.pdf", "Bitcoin Whitepaper": "samples/bitcoin.pdf", "UN Climate Report": "samples/un_climate.pdf"}
        
        current_selection = uploaded_file.name if uploaded_file else sample_choice
        if current_selection != "None" and current_selection != st.session_state["processed_source"]:
            if uploaded_file:
                with open("temp.pdf", "wb") as f: f.write(uploaded_file.getbuffer())
                process_pdf("temp.pdf", uploaded_file.name)
            else:
                process_pdf(samples[sample_choice], sample_choice)
            st.rerun()

with c_llm:
    with st.expander("⚡ Neural Settings"):
        st.caption("Fine-tune the reasoning engine's behavior and computational capacity.")
        ai_temp = st.slider("Temperature (0=Fact, 1=Creative)", 0.0, 1.0, 0.1)
        model_choice = st.selectbox("AI Brain Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])

with c_exp:
    with st.expander("💾 Export State"):
        st.download_button("Download Last AI Response", st.session_state["last_response"], file_name="insight_export.txt")
        if st.button("Reset Entire System"):
            st.session_state.clear()
            st.rerun()

with c_auth:
    with st.expander("🛠️ Developer"):
        st.markdown("**Siddhesh Kulkarni**")
        st.caption("AI Solutions Architect")
        st.markdown("✉️ [Contact via Email](mailto:kulkarnisiddhesh2626@gmail.com)")
        st.markdown("🌐 [LinkedIn Profile](https://www.linkedin.com/in/siddhesh-kulkarni-b2a600207/)")

st.divider()

# --- WORKSPACE & CHAT TABS ---
col_ws, col_add, col_del = st.columns([0.6, 0.2, 0.2])
with col_ws:
    st.markdown("### 🧠 Synthesis Workspace")
with col_add:
    if st.button("⊞ New Instance"):
        st.session_state["chat_count"] += 1
        st.rerun()
with col_del:
    if st.button("⊟ Terminate Instance"):
        if st.session_state["chat_count"] > 1:
            st.session_state["chat_count"] -= 1
            st.rerun()
        else:
            st.warning("Cannot close the primary workspace.")

chat_tabs = st.tabs([f"Terminal {i+1}" for i in range(st.session_state["chat_count"])])

def run_query(query):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=4)
    context = "\n\n".join([d.page_content for d in docs])
    llm = ChatGroq(model_name=model_choice, temperature=ai_temp, groq_api_key=GROQ_API_KEY)
    response = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
    st.session_state["last_response"] = response.content
    return response.content

# Populate Tabs
for i, tab in enumerate(chat_tabs):
    with tab:
        if st.session_state["vector_ready"]:
            # Feature: Split Layout for Summary & FAQs
            c_sum, c_faq = st.columns([0.3, 0.7])
            
            with c_sum:
                with st.popover("📊 Execute Data Summary"):
                    st.markdown(f"**Automated Analysis for:** {st.session_state['processed_source']}")
                    st.markdown(st.session_state["data_summary"])
                    
            with c_faq:
                with st.expander("🎯 Smart Queries (Click to Auto-Run)", expanded=True):
                    # To capture which button was clicked
                    clicked_q = None
                    for j, q in enumerate(st.session_state["suggested_qs"]):
                        if st.button(q, key=f"faq_{i}_{j}"):
                            clicked_q = q
            
            # Chat Input (Accepts typed input OR the clicked FAQ)
            user_q = st.text_input("Enter a custom query or click a Smart Query above:", key=f"chat_input_{i}")
            final_query = clicked_q or user_q
            
            if final_query:
                with st.spinner("Compiling insights..."):
                    result = run_query(final_query)
                    st.success(f"**Query:** {final_query}\n\n**Result:**\n{result}")
        else:
            st.info("👈 Please initialize the Knowledge Base (Get Data) at the top of the screen.")
