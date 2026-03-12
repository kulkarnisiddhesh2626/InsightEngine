# ⬡ InsightEngine Elite

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)
![Groq](https://img.shields.io/badge/Groq-Llama_3-orange.svg)

**InsightEngine** is an enterprise-grade Agentic RAG (Retrieval-Augmented Generation) system. It allows users to upload complex PDF documents and instantly extract insights, summaries, and exact factual answers using ultra-fast Groq LLM inference.

## ✨ Core Features
* **Zero Hallucination Q&A:** The AI strictly reads your uploaded document and answers using only retrieved facts.
* **Smart Synthesis:** Automatically generates a "Crisp" 3-sentence summary and an "In-Detail" point-wise breakdown upon document load.
* **Dynamic FAQ Generation:** Reads the document intent and auto-generates 10 highly contextual, clickable questions to jumpstart your analysis.
* **Multi-Instance Workspace:** Run multiple isolated chat windows simultaneously to cross-examine different parts of your data.
* **Adjustable Neural Settings:** Toggle between Llama 3.1 (8B) for speed and Llama 3.3 (70B) for deep reasoning, while controlling creativity (Temperature).

## 🏗️ System Architecture
InsightEngine utilizes a 5-step RAG pipeline to ensure high fidelity and speed:
1. **Ingestion:** Extracts raw text from complex PDF layouts (`PyPDFLoader`).
2. **Semantic Slicing:** Divides text into 800-character chunks with 100-character overlap (`RecursiveCharacterTextSplitter`) to preserve page-to-page context.
3. **Vectorization:** Converts text chunks into mathematical arrays using `HuggingFace all-MiniLM-L6-v2`.
4. **Indexing:** Stores vectors in a highly efficient local `FAISS` database.
5. **Cognitive Synthesis:** Retrieves the Top 4 most relevant mathematical chunks and injects them into the **Groq Llama 3** model for context-aware answering.

## 🚀 Installation & Local Setup

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/InsightEngine.git](https://github.com/yourusername/InsightEngine.git)
cd InsightEngine
