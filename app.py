import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="InsightEngine Elite", page_icon="💠", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #FFFDE4 0%, #FFE5B4 100%) !important; }
    [data-testid="stSidebar"] { display: none; }
    .stMarkdown, p, span, div, label, .stText { color: #2F4F4F !important; }
    h1, h2, h3 { color: #D35400 !important; font-weight: 800 !important; }
    .stButton > button { background-color: #D35400 !important; color: #FFFFFF !important; border-radius: 8px !important; font-weight: bold; width: 100%; }
    /* Make popovers look clean in narrow columns */
    div[data-testid="stPopover"] { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- SECRETS CHECK (Moved to top for dynamic AI generation) ---
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

# --- CORE LOGIC: PROCESS PDF & GENERATE SMART FAQs ---
def process_pdf(source_path, source_name):
    with st.spinner(f"Ingesting & Analyzing '{source_name}'..."):
        # 1. Load Data
        loader = PyPDFLoader(source_path)
        data = loader.load()
        st.session_state["raw_text"] = "\n".join([p.page_content for p in data])
        
        # 2. Chunk & Embed
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("faiss_index")
        
        # 3. SMART FAQ GENERATION (Reads the first 3 chunks to guess user intent)
        try:
            llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3, groq_api_key=GROQ_API_KEY)
            preview_text = "\n".join([c.page_content for c in chunks[:3]])
            prompt = f"You are a helpful data analyst. Based on the following document introduction, generate exactly 5 distinct, highly relevant questions a user could ask to get valuable insights. Make them concise. Return them as a simple numbered list without any bold text, intros, or outros.\n\nText: {preview_text}"
            faq_response = llm.invoke(prompt)
            # Filter out empty lines to keep it clean
            st.session_state["suggested_qs"] = [q.strip() for q in faq_response.content.split('\n') if q.strip()]
        except:
            # Fallback if AI gets tired
            st.session_state["suggested_qs"] = ["1. What is the main topic of this document?", "2. Summarize the key metrics.", "3. What are the major risks?", "4. Extract the main conclusion.", "5. List the key entities mentioned."]

        st.session_state["vector_ready"] = True
        st.session_state["processed_source"] = source_name

# --- HEADER ---
st.title("💠 InsightEngine Elite")

# --- 5-COLUMN TOP NAVIGATION (Side-by-Side) ---
c_doc, c_data, c_llm, c_exp, c_auth = st.columns(5)

with c_doc:
    with st.expander("📘 Documentation"):
        st.markdown("**What is this?**\nAn AI tool using **RAG** (Retrieval-Augmented Generation) to read your PDFs and answer questions factually without hallucinating.")
        st.divider()
        st.markdown("**How it Works:**\n1. Ingest PDF\n2. Chunk Text\n3. Vectorize (FAISS)\n4. Groq LLM Synthesis")

with c_data:
    with st.expander("📥 Get Data"):
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
        sample_choice = st.selectbox("Or pick sample:", ["None", "Tesla 2023 Annual Report", "AI Act Final Draft", "Bitcoin Whitepaper", "UN Climate Report"])
        
        samples = {"Tesla 2023 Annual Report": "samples/tesla_2023.pdf", "AI Act Final Draft": "samples/ai_act.pdf", "Bitcoin Whitepaper": "samples/bitcoin.pdf", "UN Climate Report": "samples/un_climate.pdf"}
        
        # Auto-trigger Processing
        current_selection = uploaded_file.name if uploaded_file else sample_choice
        if current_selection != "None" and current_selection != st.session_state["processed_source"]:
            if uploaded_file:
                with open("temp.pdf", "wb") as f: f.write(uploaded_file.getbuffer())
                process_pdf("temp.pdf", uploaded_file.name)
            else:
                process_pdf(samples[sample_choice], sample_choice)
            st.rerun()

        if st.session_state["vector_ready"]:
            with st.popover("Preview Data"):
                st.caption(f"Loaded: {st.session_state['processed_source']}")
                st.text_area("Preview", st.session_state["raw_text"][:1500] + "...", height=150)

with c_llm:
    with st.expander("⚙️ LLM Config"):
        ai_temp = st.slider("Temperature (0=Fact, 1=Creative)", 0.0, 1.0, 0.1)
        model_choice = st.selectbox("AI Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])

with c_exp:
    with st.expander("📤 Export"):
        st.download_button("Download Last AI Response", st.session_state["last_response"], file_name="insight_export.txt")
        if st.button("Reset Entire System"):
            st.session_state.clear()
            st.rerun()

with c_auth:
    with st.expander("👨‍💻 Author"):
        st.markdown("**Siddhesh Kulkarni**")
        st.caption("AI Solutions Architect")
        st.markdown("✉️ [Email Me](mailto:kulkarnisiddhesh2626@gmail.com)")
        st.markdown("🌐 [LinkedIn](https://www.linkedin.com/in/siddhesh-kulkarni-b2a600207/)")
        st.markdown("💻 [GitHub](https://github.com/kulkarnisiddhesh2626)")

st.divider()

# --- WORKSPACE & CHAT TABS ---
col_ws, col_add, col_del = st.columns([0.6, 0.2, 0.2])
with col_ws:
    st.markdown("### 🧠 Dynamic Workspace")
with col_add:
    if st.button("➕ Add Chat Window"):
        st.session_state["chat_count"] += 1
        st.rerun()
with col_del:
    if st.button("❌ Close Chat Window"):
        if st.session_state["chat_count"] > 1:
            st.session_state["chat_count"] -= 1
            st.rerun()
        else:
            st.warning("Cannot delete the last window.")

# Generate Tabs
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

# Populate Tabs
for i, tab in enumerate(chat_tabs):
    with tab:
        if st.session_state["vector_ready"]:
            # Feature: Dynamic FAQs Specific to the Dataset
            with st.expander("💡 Recommended Questions for this Dataset", expanded=True):
                st.caption(f"AI-generated based on: **{st.session_state['processed_source']}**")
                for q in st.session_state["suggested_qs"]:
                    st.markdown(f"> {q}")
            
            # Chat Input
            user_q = st.text_input("Query your document:", key=f"chat_input_{i}", placeholder="e.g., Summarize the risk factors...")
            if user_q:
                with st.spinner("Analyzing neural pathways..."):
                    result = run_query(user_q)
                    st.success(result)
        else:
            st.info("👈 Open the 'Get Data' dropdown at the top to load a document.")
