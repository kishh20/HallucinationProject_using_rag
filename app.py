"""
====================================================================
Institutional Document RAG & Hallucination Detector
====================================================================
A clean, production-grade AI system for processing institutional documents
(curricula, syllabi, academic regulations, exam codes), maintaining vector
knowledge repositories (ChromaDB / FAISS / Pinecone), and providing
explainable, citation-grounded answers with XGBoost hallucination verification.
"""
import os
import html
import pandas as pd
import streamlit as st

# Configure Streamlit Page
st.set_page_config(
    page_title="Axiom | Evidence Workspace",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply UI Theme
from ui_theme import apply_theme, render_hero
from vector_store import KnowledgeRepository
from document_processor import process_institutional_document
from retriever import retrieve_multi_source_evidence
from generator import generate_grounded_answer
from hallucination_detector import (
    detect_institutional_hallucination,
    calculate_ml_features,
    get_detector_model,
)

# Initialize Session State
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Dark"
if "vector_backend" not in st.session_state:
    st.session_state["vector_backend"] = "ChromaDB"
if "llm_provider" not in st.session_state:
    st.session_state["llm_provider"] = "Gemini"
if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = "gemini-3-flash-preview"
if "api_key" not in st.session_state:
    st.session_state["api_key"] = os.getenv("GEMINI_API_KEY", "")
if "pinecone_key" not in st.session_state:
    st.session_state["pinecone_key"] = os.getenv("PINECONE_API_KEY", "")
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "pending_starter" not in st.session_state:
    st.session_state["pending_starter"] = None

apply_theme(st.session_state["theme_mode"])


# Cache Knowledge Repository Singleton
@st.cache_resource
def get_cached_repository(backend: str, pinecone_key: str):
    repo = KnowledgeRepository(store_type=backend, pinecone_api_key=pinecone_key)
    # Seed sample institutional documents if fresh
    repo.seed_sample_documents_if_empty()
    return repo


repo = get_cached_repository(st.session_state["vector_backend"], st.session_state["pinecone_key"])


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

with st.sidebar:
    st.markdown("### ◈ Axiom")
    st.caption("Evidence workspace · Institutional knowledge")
    st.markdown("---")

    # Theme Switcher
    theme_choice = st.radio(
        "Appearance",
        ["Dark", "Light"],
        horizontal=True,
        index=0 if st.session_state["theme_mode"] == "Dark" else 1
    )
    if theme_choice != st.session_state["theme_mode"]:
        st.session_state["theme_mode"] = theme_choice
        st.rerun()

    st.markdown("#### 🧠 Model Provider")
    provider_options = ["Gemini", "OpenAI", "Claude", "OpenRouter", "Local Extractive (Offline)"]
    cur_p_idx = provider_options.index(st.session_state["llm_provider"]) if st.session_state["llm_provider"] in provider_options else 0
    sel_provider = st.selectbox("LLM Provider", provider_options, index=cur_p_idx)
    st.session_state["llm_provider"] = sel_provider

    if sel_provider == "Gemini":
        gem_models = ["gemini-3-flash-preview", "gemini-flash-latest", "gemini-3.8-flash"]
        cur_m_idx = gem_models.index(st.session_state["llm_model"]) if st.session_state["llm_model"] in gem_models else 0
        st.session_state["llm_model"] = st.selectbox("Model", gem_models, index=cur_m_idx)
        api_val = st.text_input("Gemini API Key", value=st.session_state.get("api_key", ""), type="password")
        st.session_state["api_key"] = api_val
    elif sel_provider == "OpenAI":
        oa_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        st.session_state["llm_model"] = st.selectbox("Model", oa_models, index=0)
        api_val = st.text_input("OpenAI API Key", value=st.session_state.get("api_key", ""), type="password")
        st.session_state["api_key"] = api_val
    elif sel_provider == "Claude":
        cl_models = ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]
        st.session_state["llm_model"] = st.selectbox("Model", cl_models, index=0)
        api_val = st.text_input("Anthropic API Key", value=st.session_state.get("api_key", ""), type="password")
        st.session_state["api_key"] = api_val
    elif sel_provider == "OpenRouter":
        or_models = [
            "nvidia/nemotron-3.5-lightning:free",
            "google/gemma-4-31b-it:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "openrouter/auto"
        ]
        st.session_state["llm_model"] = st.selectbox("Model", or_models, index=0)
        api_val = st.text_input("OpenRouter Key", value=st.session_state.get("api_key", ""), type="password")
        st.session_state["api_key"] = api_val
    else:
        st.caption("⚡ Zero-Key Mode: Grounded extractive synthesis directly from retrieved context.")

    st.markdown("---")
    st.markdown("#### 🗄️ Knowledge Vector Engine")
    vec_options = ["ChromaDB", "FAISS", "Pinecone"]
    cur_v_idx = vec_options.index(st.session_state["vector_backend"]) if st.session_state["vector_backend"] in vec_options else 0
    sel_backend = st.selectbox("Vector Database", vec_options, index=cur_v_idx)

    if sel_backend != st.session_state["vector_backend"]:
        st.session_state["vector_backend"] = sel_backend
        st.cache_resource.clear()
        st.rerun()

    if sel_backend == "Pinecone":
        pc_key = st.text_input("Pinecone Key", value=st.session_state["pinecone_key"], type="password")
        if pc_key != st.session_state["pinecone_key"]:
            st.session_state["pinecone_key"] = pc_key
            st.cache_resource.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("#### 📊 Knowledge Base Stats")
    docs_list = repo.list_documents()
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Documents", len(docs_list))
    with col_m2:
        st.metric("Chunks", repo.count())

    st.write("")
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state["chat_messages"] = []
        st.rerun()


# ============================================================
# FIXED BOTTOM CHAT INPUT BAR & TAB ROUTING
# ============================================================

# Docked cleanly to the bottom of the browser viewport (like Claude / ChatGPT)
chat_prompt = st.chat_input("Ask a question about institutional courses, syllabi, or regulations...")
pending_starter = st.session_state.pop("pending_starter", None)
active_query = (pending_starter or chat_prompt or "").strip()

# ============================================================
# APP TABS
# ============================================================

tab_chat, tab_kb, tab_lab, tab_guide = st.tabs([
    "Ask & verify",
    "Knowledge base",
    "Verification lab",
    "System guide"
])


# ============================================================
# TAB 1: ASK & VERIFY (CHAT INTERFACE)
# ============================================================

with tab_chat:
    show_empty_state = (len(st.session_state["chat_messages"]) == 0) and (not active_query)

    # Empty State: Hero Heading Banner & 4 Starter Cards (Disappear after first question, like Claude)
    if show_empty_state:
        render_hero(
            title="Answers grounded in evidence.",
            subtitle="Search academic policies, course material, and institutional guidance. Every response is checked against its source documents.",
            icon="◈"
        )
        st.markdown("<p style='color: var(--muted); font-size: 0.93rem; margin: 1rem 0 0.6rem 0;'>Select a sample question below or type your query in the bottom chat bar:</p>", unsafe_allow_html=True)
        st.markdown('<div class="starter-card-box">', unsafe_allow_html=True)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("📋 What is the minimum attendance required for exams?", use_container_width=True):
                st.session_state["pending_starter"] = "What is the minimum attendance required to appear for the End-Semester Examination?"
                st.rerun()
            if st.button("🤖 What topics are covered in Unit 4 of CS401 AI?", use_container_width=True):
                st.session_state["pending_starter"] = "What topics and algorithms are covered in Unit IV of CS401 Artificial Intelligence and Neural Networks?"
                st.rerun()
        with col_s2:
            if st.button("⚖️ What are the penalties for Category 2 academic malpractice?", use_container_width=True):
                st.session_state["pending_starter"] = "What constitutes a Category 2 malpractice infraction and what are the specific penalties?"
                st.rerun()
            if st.button("🎓 What is the criteria for First Class with Distinction?", use_container_width=True):
                st.session_state["pending_starter"] = "What is the CGPA requirement and conditions for graduating with First Class with Distinction?"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Render Existing Chat Messages
    for msg in st.session_state["chat_messages"]:
        with st.chat_message("user"):
            st.markdown(f"**{msg['question']}**")

        with st.chat_message("assistant"):
            v = msg["verification"]
            srcs = msg.get("sources", [])
            faithfulness = v.get("faithfulness_score", 0.0)
            risk_label = v.get("hallucination_risk", "Verified")
            risk_color = v.get("risk_color", "success")

            # 1. Verification Status Pill
            pill_class = f"hd-pill-{risk_color}"
            icon = "✓" if risk_color == "success" else ("⚠️" if risk_color == "warning" else "✕")
            st.markdown(
                f'<div class="hd-pill {pill_class}">{icon} {html.escape(risk_label)} · {faithfulness:.0f}% supported</div>',
                unsafe_allow_html=True
            )

            # 2. Main Grounded Answer (Clean Markdown Rendering)
            st.markdown(msg["answer"])

            # Diagnostic note if local fallback occurred
            if msg.get("error_note"):
                st.caption(f"ℹ️ {msg['error_note']}")

            # 3. Source Reference Tags
            if srcs:
                st.markdown(
                    '<div style="font-size: 0.76rem; font-weight: 600; color: var(--muted); text-transform: uppercase; margin-top: 0.75rem;">Source Documents:</div>',
                    unsafe_allow_html=True
                )
                tags_html = '<div class="hd-sources-container">'
                seen_sources = set()
                for s in srcs:
                    fn = s.get("filename", "Doc")
                    pg = s.get("page_number", 1)
                    key = f"{fn}_{pg}"
                    if key not in seen_sources:
                        seen_sources.add(key)
                        tags_html += f'<span class="hd-source-tag">📄 {html.escape(fn)} (Pg {pg})</span>'
                tags_html += '</div>'
                st.markdown(tags_html, unsafe_allow_html=True)

            # 4. Detailed Verification & Diagnostics Expander
            with st.expander("🛡️ Verification Analysis & Claim Attribution"):
                ml_pred = v.get("ml_prediction", "Faithful")
                ml_conf = v.get("ml_confidence", 0.0)
                grounded_c = v.get("claims_grounded", 0)
                total_c = v.get("claims_total", 0)
                feat = v.get("features", {})
                k_a_sim = feat.get("knowledge_answer_similarity", 0.0)

                # Metric 4-Grid
                st.markdown(
                    f"""
                    <div class="metric-grid-4">
                        <div class="metric-box">
                            <div class="metric-box-title">Faithfulness Score</div>
                            <div class="metric-box-value" style="color: {'var(--success)' if faithfulness >= 70 else ('var(--warning)' if faithfulness >= 50 else 'var(--danger)')};">{faithfulness:.1f}%</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-box-title">XGBoost V2 Model</div>
                            <div class="metric-box-value">{ml_pred} <span style="font-size:0.8rem; font-weight:400; color:var(--muted);">({ml_conf:.0f}%)</span></div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-box-title">Claims Grounded</div>
                            <div class="metric-box-value">{grounded_c} / {total_c}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-box-title">Knowledge Match</div>
                            <div class="metric-box-value">{k_a_sim:.2f}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Sentence-by-Sentence Claim Breakdown
                claims = v.get("claims", [])
                if claims:
                    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: var(--text); margin: 0.8rem 0 0.4rem;'>Claim-by-Claim Evidence Mapping:</div>", unsafe_allow_html=True)
                    for c_idx, cl in enumerate(claims, 1):
                        badge_color = "var(--success)" if cl["status"] == "GROUNDED" else ("var(--warning)" if cl["status"] == "PARTIALLY_GROUNDED" else "var(--danger)")
                        st.markdown(
                            f"""
                            <div style="background: var(--surface); border: 1px solid var(--border); border-left: 3px solid {badge_color}; border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                    <span style="font-size: 0.82rem; font-weight: 600; color: var(--muted);">Claim {c_idx}</span>
                                    <span style="font-size: 0.78rem; font-weight: 600; color: {badge_color};">{cl['badge']} ({cl['similarity']}% similarity)</span>
                                </div>
                                <div style="font-size: 0.92rem; color: var(--text); margin-bottom: 6px;">"{html.escape(cl['claim'])}"</div>
                                <div style="font-size: 0.8rem; color: var(--muted);">
                                    <strong>Source:</strong> {html.escape(cl['best_source_doc'])} (Page {cl['best_source_page']}) | <strong>Lexical Overlap:</strong> {cl['lexical_overlap']}%
                                </div>
                                <div style="font-size: 0.8rem; background: var(--surface-2); padding: 5px 8px; border-radius: 6px; margin-top: 4px; color: var(--text);">
                                    <em>"{html.escape(cl['best_evidence_text'])}"</em>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                # Retrieved Context Chunks Details
                if srcs:
                    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: var(--text); margin: 0.8rem 0 0.4rem;'>Retrieved Chunks from Vector Store:</div>", unsafe_allow_html=True)
                    for s_idx, src in enumerate(srcs, 1):
                        st.caption(f"**Chunk {s_idx}**: {src.get('filename')} (Page {src.get('page_number', 1)}) — Relevance: {src.get('fused_score', 0.5):.2f}")
                        st.text_area(f"Raw Chunk #{s_idx}", value=src.get("text", ""), height=90, disabled=True, key=f"src_box_{msg['id']}_{s_idx}")

    # Process Active Query (from bottom input bar or starter card)
    if active_query:
        active_query = active_query.strip()
        with st.chat_message("user"):
            st.markdown(f"**{active_query}**")

        with st.chat_message("assistant"):
            with st.spinner("🔎 Retrieving relevant institutional evidence chunks..."):
                retrieval = retrieve_multi_source_evidence(
                    repo=repo,
                    query=active_query,
                    top_k=4,
                    min_similarity=0.25
                )

            srcs = retrieval["sources"]
            context = retrieval["combined_context"]

            if not srcs:
                st.warning("No relevant institutional documents found. Please upload or index syllabi in the 'Knowledge Base Manager' tab.")
            else:
                with st.spinner(f"🧠 Synthesizing answer with {st.session_state['llm_provider']}..."):
                    gen_res = generate_grounded_answer(
                        question=active_query,
                        context=context,
                        sources=srcs,
                        provider=st.session_state["llm_provider"],
                        api_key=st.session_state["api_key"],
                        model_name=st.session_state["llm_model"]
                    )
                    answer_text = gen_res["answer"]
                    err_note = gen_res.get("error_note")

                with st.spinner("🛡️ Verifying claims & predicting hallucination confidence..."):
                    verif_res = detect_institutional_hallucination(
                        question=active_query,
                        answer=answer_text,
                        retrieved_sources=srcs,
                        combined_context=context
                    )

                # Save to session history
                new_msg = {
                    "id": len(st.session_state["chat_messages"]) + 1,
                    "question": active_query,
                    "answer": answer_text,
                    "provider": gen_res["provider"],
                    "sources": srcs,
                    "verification": verif_res,
                    "error_note": err_note,
                }
                st.session_state["chat_messages"].append(new_msg)
                st.rerun()


# ============================================================
# TAB 2: KNOWLEDGE BASE MANAGER
# ============================================================

with tab_kb:
    st.markdown("### Knowledge base")
    st.caption("Add and organize source documents for evidence grounded answers.")

    col_u1, col_u2 = st.columns([3, 2])
    with col_u1:
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "docx", "txt", "md", "csv"],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT, Markdown, CSV"
        )
    with col_u2:
        doc_category = st.selectbox(
            "Category",
            ["Curriculum & Syllabi", "Academic Regulations", "Examination Code", "General Handbooks"]
        )
        doc_dept = st.text_input("Department / Academic Unit", value="Academic Affairs")
        doc_course = st.text_input("Course Code (Optional)", placeholder="e.g. CS401, DS302")

    if st.button("📥 Index Uploaded Documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("Please choose at least one document to upload.")
        else:
            total_added = 0
            prog = st.progress(0)
            for idx, file in enumerate(uploaded_files):
                file_bytes = file.read()
                chunks = process_institutional_document(
                    file_bytes=file_bytes,
                    filename=file.name,
                    category=doc_category,
                    course_code=doc_course,
                    department=doc_dept,
                )
                added = repo.add_chunks(chunks)
                total_added += added
                prog.progress((idx + 1) / len(uploaded_files))

            st.success(f"🎉 Successfully ingested {len(uploaded_files)} file(s) ({total_added} vector chunks) into {st.session_state['vector_backend']}!")
            st.rerun()

    st.markdown("---")
    st.markdown("#### 📑 Current Document Inventory")
    doc_inv = repo.list_documents()

    if not doc_inv:
        st.info("No documents are currently indexed in the active vector store.")
    else:
        df_inv = pd.DataFrame(doc_inv)
        st.dataframe(df_inv, use_container_width=True)

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            doc_to_delete = st.selectbox("Select document to delete", [d["filename"] for d in doc_inv])
            if st.button("🗑️ Delete Selected Document", use_container_width=True):
                if repo.delete_document(doc_to_delete):
                    st.success(f"Deleted '{doc_to_delete}'.")
                    st.rerun()
        with col_d2:
            if st.button("🔄 Reload Sample Syllabi", use_container_width=True):
                repo.clear()
                repo.seed_sample_documents_if_empty()
                st.success("Default institutional syllabi and policies re-indexed.")
                st.rerun()
        with col_d3:
            if st.button("⚠️ Clear Entire Repository", use_container_width=True):
                repo.clear()
                st.warning("Knowledge repository cleared.")
                st.rerun()


# ============================================================
# TAB 3: HALLUCINATION LAB
# ============================================================

with tab_lab:
    st.markdown("### Verification lab")
    st.caption("Inspect model signals and trace individual claims back to their supporting evidence.")

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        test_q = st.text_input("Test Question", value="What is the minimum pass mark for CS401?")
        test_k = st.text_area(
            "Knowledge Evidence Chunk",
            value="A student must obtain at least 40% in the End-Semester Examination and an overall aggregate of 45% (continuous assessment + final exam combined) to successfully earn credit for CS401 Artificial Intelligence and Neural Networks.",
            height=120
        )
    with col_l2:
        test_a = st.text_area(
            "Generated Answer to Verify",
            value="To pass CS401, students must obtain 40% in the final exam and 45% aggregate. Students must also present a research paper at an IEEE conference.",
            height=185
        )

    if st.button("🧪 Run Diagnostic Hallucination Test", type="primary"):
        feat_df, feat_dict = calculate_ml_features(test_k, test_q, test_a)
        detector = get_detector_model()

        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            st.metric("Knowledge-Answer Sim", f"{feat_dict['knowledge_answer_similarity']:.3f}")
        with col_f2:
            st.metric("Question-Answer Sim", f"{feat_dict['question_answer_similarity']:.3f}")
        with col_f3:
            st.metric("Knowledge-Question Sim", f"{feat_dict['knowledge_question_similarity']:.3f}")
        with col_f4:
            st.metric("Lexical Overlap", f"{feat_dict['word_overlap'] * 100:.1f}%")

        if detector is not None:
            pred = detector.predict(feat_df)[0]
            prob = detector.predict_proba(feat_df)[0]
            is_faithful = (pred == 0)
            conf = prob[0] * 100 if is_faithful else prob[1] * 100
            st.markdown(
                f"""
                <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; margin-top: 1rem;">
                    <h4 style="margin: 0; color: {'var(--success)' if is_faithful else 'var(--danger)'};">
                        XGBoost V2 Output: {'Faithful (Grounded)' if is_faithful else 'Hallucinated (Unsupported)'}
                    </h4>
                    <p style="margin: 4px 0 0; color: var(--muted); font-size: 0.9rem;">
                        Classification Confidence: <strong>{conf:.1f}%</strong> | Predicted Class: <strong>{pred}</strong> (0=Faithful, 1=Hallucinated)
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Sentence-Level Attribution
        mock_evidence = [{"filename": "Test_Context.txt", "page_number": 1, "course_code": "CS401", "text": test_k}]
        claims_test = detect_institutional_hallucination(test_q, test_a, mock_evidence, test_k)

        st.markdown("<div style='font-size: 0.9rem; font-weight: 600; color: var(--text); margin: 1rem 0 0.5rem;'>Sentence-by-Sentence Breakdown:</div>", unsafe_allow_html=True)
        for cl in claims_test["claims"]:
            badge_color = "var(--success)" if cl["status"] == "GROUNDED" else ("var(--warning)" if cl["status"] == "PARTIALLY_GROUNDED" else "var(--danger)")
            st.markdown(
                f"""
                <div style="background: var(--surface); border: 1px solid var(--border); border-left: 3px solid {badge_color}; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px;">
                    <div style="font-size: 0.88rem; color: var(--text); font-weight: 500;">"{html.escape(cl['claim'])}"</div>
                    <div style="font-size: 0.8rem; color: {badge_color}; margin-top: 3px;">
                        {cl['badge']} ({cl['similarity']}% similarity) | Lexical Overlap: {cl['lexical_overlap']}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# TAB 4: ARCHITECTURE & GUIDE
# ============================================================

with tab_guide:
    st.markdown("### How Axiom works")
    st.markdown(
        """
        ```
        Institutional Documents (PDF / DOCX / TXT / MD / CSV)
                         │
                         ▼
        [Document Processing & Recursive Character Chunking]
                         │
                         ▼
           [Vector Store: ChromaDB / FAISS / Pinecone]
                         │
        User Query ──────┼────► [Hybrid Dense + Keyword Retrieval]
                         │                       │
                         ▼                       ▼
            [Multi-Source Ranked Chunks] ──► [Multi-LLM Grounded Generator]
                         │                       │
                         │                       ▼
                         │               Generated Answer with Citations
                         │                       │
                         └──────────────┬────────┘
                                        ▼
             [Hallucination Detector & Verification Engine]
                         ├─ XGBoost V2 Classifier (4 Features)
                         ├─ Sentence-Level Evidence Attribution
                         └─ Explainability Scorecard & Metrics
        ```

        #### 🌟 Features & Best Practices:
        1. **Multi-Source Retrieval**: Gathers and ranks evidence across multiple uploaded syllabi and policies.
        2. **Multi-LLM Compatibility**: Switch between Gemini, OpenAI, Claude, OpenRouter, or Local Extractive mode.
        3. **Explainable Hallucination Verification**: Evaluates each sentence against retrieved evidence chunks to highlight verified facts vs hallucinations.
        4. **Local Execution**: Completely self-contained; runs on your machine without external cloud deployment dependencies.
        """
    )
