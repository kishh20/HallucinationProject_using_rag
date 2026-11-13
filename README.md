# 🏛️ Institutional Document RAG & Hallucination Detector

> **Search institutional syllabi, academic curricula, and university regulations with multi-source retrieval, grounded synthesis, and ML-backed hallucination verification.**

A local, production-grade AI system engineered to process institutional documents, maintain vector knowledge repositories, answer queries strictly from retrieved evidence, and provide explainable verification with sentence-level claim attribution and an XGBoost V2 model.

---

## 🌟 Key Features

1. **📄 Institutional Document Ingestion & Chunking**:
   - Parses **PDF**, **DOCX**, **TXT**, **MD**, and **CSV** institutional documents.
   - Preserves page numbers, course codes, departments, and category tags.
   - Intelligent recursive character chunking with configurable overlap.

2. **🗄️ Searchable Knowledge Repositories (Multi-Engine)**:
   - **ChromaDB**: Persistent, on-disk semantic collections.
   - **FAISS**: High-speed IndexFlatIP vector search.
   - **Pinecone**: Cloud vector index integration.
   - Dense vector embeddings via `sentence-transformers/all-MiniLM-L6-v2`.

3. **🔍 Multi-Source Hybrid Retrieval**:
   - Fuses dense semantic search with course code keyword boosting (e.g. `CS401`, `DS302`, `REG-2025`).
   - Grouping, ranking, and deduplication of evidence chunks across multiple institutional files.

4. **🧠 Grounded Multi-LLM Synthesis**:
   - Seamless switching across **Google Gemini**, **OpenAI (GPT-4o)**, **Anthropic Claude**, and **OpenRouter**.
   - Includes a **100% Offline / Zero-Key Local Extractive Synthesizer** for instant local testing without API keys.
   - Strict citation enforcement: every claim references `[Document, Page X]`.

5. **🛡️ Explainable Hallucination Detection & Verification**:
   - **Sentence-Level Decomposition**: Breaks answers down into individual factual claims.
   - **Evidence Attribution**: Matches each claim against retrieved evidence and assigns trust levels (🟢 Grounded / 🟡 Ambiguous / 🔴 Hallucinated).
   - **XGBoost V2 Classifier**: Evaluates 4 semantic-lexical features (`knowledge_answer_similarity`, `question_answer_similarity`, `knowledge_question_similarity`, `word_overlap`).
   - **Explainability Highlighting**: Color-codes the answer so users can immediately distinguish verified facts from potential hallucinations.

6. **💻 100% Local Execution**:
   - Designed to run locally on your system using Streamlit.
   - No external deployment dependencies required.

---

## 🚀 Quick Start Guide

### 1. Requirements & Environment
The system runs on Python 3.10+ (tested on Python 3.14). All dependencies are listed in `requirements.txt`.

To install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
Run the Streamlit application from the project root:
```bash
streamlit run app.py
```
Open your web browser at:
```
http://localhost:8501
```

---

## 📁 Project Structure

```
HallucinationProject_using_rag/
├── app.py                      # Streamlit frontend with 4 interactive tabs
├── ui_theme.py                 # Modern Claude/ChatGPT-inspired UI theme and CSS
├── document_processor.py       # PDF/DOCX/TXT/CSV parser & recursive chunker
├── vector_store.py             # Unified Vector Repository (ChromaDB / FAISS / Pinecone)
├── retriever.py                # Multi-source hybrid retrieval & context packager
├── generator.py                # Multi-provider LLM answer generation with citation rules
├── hallucination_detector.py   # XGBoost V2 model & claim-level explainability engine
├── hallucination_model_v2.pkl  # Trained XGBoost hallucination detection classifier
├── sample_documents/           # Preloaded institutional documents
│   ├── CS401_Artificial_Intelligence_and_Neural_Networks_Syllabus.txt
│   ├── DS302_Applied_Data_Science_and_Big_Data_Analytics.txt
│   ├── University_Academic_Regulations_and_Grading_Policy_2025.txt
│   └── Campus_Examination_Conduct_and_Academic_Integrity_Code.txt
├── data/                       # Local persistent storage for vector stores
│   ├── chroma_db/
│   └── faiss_db/
├── .streamlit/
│   └── config.toml             # Custom styling and server configurations
└── requirements.txt            # Python dependencies
```

---

## 🔬 How Hallucination Verification Works

```
User Query ──► Multi-Source Retrieval (ChromaDB / FAISS)
                         │
                         ▼
             Grounded LLM Generator
                         │
                         ├────────────────────────┐
                         ▼                        ▼
           Claim Decomposition & Match      XGBoost V2 Classifier
           (Sentence vs Evidence Chunks)    (4 Semantic & Lexical Features)
                         │                        │
                         └───────────┬────────────┘
                                     ▼
                      Interactive Explainable Report
             (Faithfulness %, Risk Meter, Sentence Highlights)
```
