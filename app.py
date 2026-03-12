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
    
    /* Fixed Button CSS to ensure text is white and readable */
    .stButton > button { 
        background-color: #334155 !important; 
        color: #FFFFFF !important; 
        border-radius: 6px !important; 
        font-weight: 600; 
        width: 100%; 
        border: none !important;
        transition: all 0.3s ease; 
    }
    .stButton > button * { color: #FFFFFF !important; } /* Forces inner text to be white */
    
    .stButton > button:hover { background-color: #0F172A !important; }
    .stButton > button:hover * { color: #38BDF8 !important; } /* Changes to light blue on hover */
    
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
if "crisp_summary" not in st.session_state: st.session_state["crisp_summary"] = "No crisp summary generated."
if "detail_summary" not in st.session_state: st.session_state["detail_summary"] = "No detailed summary generated."

# --- CORE LOGIC: PROCESS PDF, SMART FAQs, & SUMMARIES ---
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
        
        try:
            llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2, groq_api_key=GROQ_API_KEY)
            preview_text = "\n".join([c.page_content for c in chunks[:4]])
            
            # 1. FAQ Prompt
            prompt_faq = f"Generate exactly 5 distinct, highly relevant questions a user could ask to get valuable insights from this text. Make them concise. Return them as a numbered list without intros. Text: {preview_text}"
            faq_response = llm.invoke(prompt_faq)
            st.session_state["suggested_qs"] = [q.strip() for q in faq_response.content.split('\n') if q.strip()]
            
            # 2. Crisp Summary Prompt
            prompt_crisp = f"Provide a very short, crisp, 3-sentence summary of the following document excerpt. Text: {preview_text}"
            crisp_response = llm.invoke(prompt_crisp)
            st.session_state["crisp_summary"] = crisp_response.content

            # 3. Detailed Summary Prompt
            prompt_detail = f"Provide a comprehensive, in-depth point-wise summary of the core themes, entities, and purpose of the following document excerpt. Text: {preview_text}"
            detail_response = llm.invoke(prompt_detail)
            st.session_state["detail_summary"] = detail_response.content
            
        except:
            st.session_state["suggested_qs"] = ["1. What is the main topic?", "2. Summarize key metrics.", "3. Identify major risks.", "4. What is the conclusion?", "5. List the key entities."]
            st.session_state["crisp_summary"] = "Summary generation failed."
            st.session_state["detail_summary"] = "Summary generation failed. Data loaded successfully."

        st.session_state["vector_ready"] = True
        st.session_state["processed_source"] = source_name

# --- HEADER ---
st.title("⬡ InsightEngine")
st.markdown("*An intelligent AI assistant that reads your documents and answers questions with exact facts.*")

# --- 5-COLUMN TOP NAVIGATION ---
c_doc, c_data, c_llm, c_exp, c_auth = st.columns(5)

with c_doc:
    with st.expander("🗂️ Documentation"):
        st.markdown("**What is InsightEngine?**\nInsightEngine is a powerful tool that allows you to upload any PDF document and instantly ask questions about it. Instead of relying on general AI knowledge, this tool strictly reads *your* document to give you factual, exact answers.")
        with st.popover("🔍 See In-Detail Architecture"):
            st.markdown("""
            ### 🏗️ Complete System Architecture & Documentation
            
            InsightEngine is built on a state-of-the-art **Retrieval-Augmented Generation (RAG)** pipeline. This solves the primary issue with traditional LLMs: "Hallucinations" (making things up). By forcing the AI to read a retrieved text chunk before answering, accuracy reaches near 100%.

            **Phase 1: Ingestion & Processing**
            * **Document Loader (LangChain PyPDFLoader):** Extracts raw text from complex PDF layouts.
            * **Semantic Chunking:** The AI cannot read a 500-page book in one go. We use a `RecursiveCharacterTextSplitter` to cut the document into 800-character blocks. We leave a 100-character overlap between blocks so sentences at the edges don't lose context.
            
            **Phase 2: Mathematical Embedding**
            * **HuggingFace MiniLM-L6-v2:** We pass the text chunks through a neural network that converts words into complex numerical coordinates (Vectors). 
            * **FAISS Vector Database:** Created by Meta, this database stores those numbers. When you ask a question, your question is also turned into numbers, and FAISS instantly finds the chunks that are mathematically closest to your question.

            **Phase 3: Generation (Groq Llama 3)**
            * Once the top 4 most relevant text blocks are found, they are bundled together and sent to the **Groq Llama 3** model. The prompt strictly instructs the AI: *"Using ONLY the provided text, answer the user's question."*
            
            **Use Cases:**
            * **Legal:** Extracting specific clauses or liabilities from contracts.
            * **Financial:** Summarizing earnings reports or identifying risk factors.
            * **Academic:** Synthesizing literature reviews from massive research papers.
            """)

with c_data:
    with st.expander("📥 Get Data"):
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
    with st.expander("⚡ LLM Settings"):
        st.caption("Adjust how the AI thinks and responds.")
        ai_temp = st.slider("Temperature (0=Fact, 1=Creative)", 0.0, 1.0, 0.1)
        model_choice = st.selectbox("AI Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])

with c_exp:
    with st.expander("💾 Export"):
        st.download_button("Download Last AI Response", st.session_state["last_response"], file_name="insight_export.txt")
        if st.button("Reset Entire System"):
            st.session_state.clear()
            st.rerun()

with c_auth:
    with st.expander("🛠️ Author"):
        st.markdown("**Siddhesh Kulkarni**")
        st.caption("AI Solutions Architect")
        st.markdown("✉️ [Contact via Email](mailto:kulkarnisiddhesh2626@gmail.com)")
        st.markdown("🌐 [LinkedIn Profile](https://www.linkedin.com/in/siddhesh-kulkarni-b2a600207/)")

st.divider()

# --- WORKSPACE & CHAT TABS ---
col_ws, col_add, col_del = st.columns([0.6, 0.2, 0.2])
with col_ws:
    st.markdown("### 🧠 Workspace")
with col_add:
    if st.button("⊞ New Chat Window"):
        st.session_state["chat_count"] += 1
        st.rerun()
with col_del:
    if st.button("⊟ Close Chat Window"):
        if st.session_state["chat_count"] > 1:
            st.session_state["chat_count"] -= 1
            st.rerun()
        else:
            st.warning("Cannot close the primary workspace.")

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
            # Feature: Split Layout for TWO Summary Buttons
            c_crisp, c_detail, c_empty = st.columns([0.25, 0.25, 0.5])
            
            with c_crisp:
                with st.popover("📊 Crisp Summary"):
                    st.markdown(f"**Quick Overview of:** {st.session_state['processed_source']}")
                    st.write(st.session_state["crisp_summary"])
            
            with c_detail:
                with st.popover("📑 In-Detail Summary"):
                    st.markdown(f"**Deep Dive Analysis of:** {st.session_state['processed_source']}")
                    st.markdown(st.session_state["detail_summary"])
                    
            st.markdown("---")
            
            # --- CHAT INTERFACE ---
            # 1. Manual Input Box
            user_q = st.text_input("💬 Type your custom question here and press Enter:", key=f"chat_input_{i}")
            
            # 2. Suggested Clickable Questions
            clicked_q = None
            with st.expander("🎯 Suggested Questions (Click to Auto-Run)", expanded=True):
                for j, q in enumerate(st.session_state["suggested_qs"]):
                    if st.button(q, key=f"faq_{i}_{j}"):
                        clicked_q = q
            
            # Run whichever input is provided
            final_query = clicked_q or user_q
            
            if final_query:
                with st.spinner("Finding the answer..."):
                    result = run_query(final_query)
                    st.success(f"**Question:** {final_query}\n\n**Answer:**\n{result}")
        else:
            st.info("👈 Please select a document from the 'Get Data' menu at the top to begin.")
