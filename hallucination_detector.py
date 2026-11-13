"""
Institutional Hallucination Detector & Explainability Engine
Combines:
1. Sentence-Level Claim Decomposition & Semantic Grounding against retrieved chunks.
2. XGBoost V2 ML 4-Feature Relationship Classifier.
3. Ensemble Calibration: Faithfulness Score & Risk Meter.
"""
import os
import re
import html
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_V2_PATH = os.path.join(BASE_DIR, "hallucination_model_v2.pkl")

FEATURE_NAMES = [
    "knowledge_answer_similarity",
    "question_answer_similarity",
    "knowledge_question_similarity",
    "word_overlap"
]

_DETECTOR_MODEL = None


def get_detector_model():
    global _DETECTOR_MODEL
    if _DETECTOR_MODEL is None and os.path.exists(MODEL_V2_PATH):
        try:
            _DETECTOR_MODEL = joblib.load(MODEL_V2_PATH)
        except Exception as e:
            print(f"Warning: Could not load XGBoost model from {MODEL_V2_PATH}: {e}")
            _DETECTOR_MODEL = None
    return _DETECTOR_MODEL


def get_encoder_model():
    from vector_store import get_embedding_model
    return get_embedding_model()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def split_into_sentence_claims(text: str) -> List[str]:
    """Decompose generated response into testable factual claim sentences."""
    if not text:
        return []

    # Strip any bracketed citations like [Document.txt, Page 1] or [Source 1, Page 2]
    cleaned = re.sub(r"\[[^\]]+\]", "", str(text))
    # Strip markdown headers, lists, numbers, and bullet points
    cleaned = re.sub(r"^[#*\-\d.]+\s*", "", cleaned, flags=re.M)
    # Strip bold labels like '**Attendance Threshold**:' or '**Note**:'
    cleaned = re.sub(r"\*\*[^*]+?\*\*:\s*", "", cleaned)

    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", cleaned.strip())
    claims = []
    for s in raw_sentences:
        s_clean = s.strip().strip("*#-_: ")
        # Only keep full factual sentences (length > 25, at least 4 words, not ending in colon)
        if len(s_clean) > 25 and not s_clean.lower().startswith((
            "based on", "here is", "in summary", "according to", "note:",
            "the following", "please refer", "as follows"
        )):
            if not s_clean.endswith(":") and len(s_clean.split()) >= 4:
                claims.append(s_clean)

    if not claims and cleaned.strip():
        fallback = cleaned.strip()
        if len(fallback) > 15:
            claims = [fallback]
    return claims


def calculate_ml_features(
    knowledge: str,
    question: str,
    answer: str,
    knowledge_embedding: Optional[np.ndarray] = None,
    question_embedding: Optional[np.ndarray] = None,
    answer_embedding: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute the 4 core relationship features for XGBoost V2."""
    clean_k = re.sub(r"\[[^\]]+\]", "", knowledge)
    clean_a = re.sub(r"\[[^\]]+\]", "", answer)

    if knowledge_embedding is None or question_embedding is None or answer_embedding is None:
        encoder = get_encoder_model()
        embs = encoder.encode([clean_k, question, clean_a], normalize_embeddings=True, show_progress_bar=False)
        knowledge_embedding = embs[0]
        question_embedding = embs[1]
        answer_embedding = embs[2]

    k_a_sim = cosine_similarity(knowledge_embedding, answer_embedding)
    q_a_sim = cosine_similarity(question_embedding, answer_embedding)
    k_q_sim = cosine_similarity(knowledge_embedding, question_embedding)

    # Word overlap
    knowledge_words = set(re.findall(r"\b\w{3,}\b", clean_k.lower()))
    answer_words = set(re.findall(r"\b\w{3,}\b", clean_a.lower()))

    if answer_words:
        word_overlap = len(knowledge_words.intersection(answer_words)) / len(answer_words)
    else:
        word_overlap = 0.0

    features_dict = {
        "knowledge_answer_similarity": round(float(k_a_sim), 4),
        "question_answer_similarity": round(float(q_a_sim), 4),
        "knowledge_question_similarity": round(float(k_q_sim), 4),
        "word_overlap": round(float(word_overlap), 4),
    }

    features_df = pd.DataFrame([[
        k_a_sim,
        q_a_sim,
        k_q_sim,
        word_overlap
    ]], columns=FEATURE_NAMES)

    return features_df, features_dict


def verify_claims_against_evidence(
    claims: List[str],
    evidence_chunks: List[Dict[str, Any]],
    claim_embeddings: Optional[np.ndarray] = None,
    evidence_embeddings: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Verify each extracted sentence claim against retrieved institutional evidence chunks.
    Calculates semantic cosine similarity and lexical token overlap.
    """
    if not claims or not evidence_chunks:
        return []

    if claim_embeddings is None or evidence_embeddings is None:
        encoder = get_encoder_model()
        evidence_texts = [re.sub(r"\[[^\]]+\]", "", e["text"]) for e in evidence_chunks]
        all_embs = encoder.encode(claims + evidence_texts, normalize_embeddings=True, show_progress_bar=False)
        claim_embeddings = all_embs[:len(claims)]
        evidence_embeddings = all_embs[len(claims):]

    claim_results = []
    for c_idx, claim in enumerate(claims):
        c_emb = claim_embeddings[c_idx]
        best_sim = -1.0
        best_evidence_idx = 0

        # Compare claim against all retrieved chunks
        for e_idx, e_emb in enumerate(evidence_embeddings):
            sim = float(np.dot(c_emb, e_emb))
            if sim > best_sim:
                best_sim = sim
                best_evidence_idx = e_idx

        best_chunk = evidence_chunks[best_evidence_idx]

        # Lexical token overlap with the best matching chunk
        c_words = set(re.findall(r"\b\w{3,}\b", claim.lower()))
        e_words = set(re.findall(r"\b\w{3,}\b", best_chunk["text"].lower()))
        overlap = len(c_words.intersection(e_words)) / max(len(c_words), 1)

        # High-precision grounding boundaries:
        if best_sim >= 0.52 or (best_sim >= 0.44 and overlap >= 0.25):
            status = "GROUNDED"
            badge = "🟢 Grounded"
        elif best_sim >= 0.38 or overlap >= 0.18:
            status = "PARTIALLY_GROUNDED"
            badge = "🟡 Ambiguous"
        else:
            status = "HALLUCINATED"
            badge = "🔴 Unsupported"

        claim_results.append({
            "claim": claim,
            "status": status,
            "badge": badge,
            "similarity": round(best_sim * 100, 1),
            "lexical_overlap": round(overlap * 100, 1),
            "best_source_doc": best_chunk.get("filename", "Unknown"),
            "best_source_page": best_chunk.get("page_number", 1),
            "best_source_course": best_chunk.get("course_code", ""),
            "best_evidence_text": best_chunk["text"][:260] + ("..." if len(best_chunk["text"]) > 260 else ""),
        })

    return claim_results


def detect_institutional_hallucination(
    question: str,
    answer: str,
    retrieved_sources: List[Dict[str, Any]],
    combined_context: str
) -> Dict[str, Any]:
    """
    Master Hallucination Detection & Verification Pipeline:
    1. Runs Sentence-Level Claim Evidence Attribution with single-pass batch embeddings.
    2. Runs XGBoost V2 Feature Extraction.
    3. Calibrates Ensemble Faithfulness Score & Risk Status.
    """
    if not answer.strip() or not retrieved_sources:
        return {
            "verified": False,
            "faithfulness_score": 0.0,
            "hallucination_risk": "HIGH RISK (No context)",
            "risk_color": "danger",
            "ml_prediction": "Hallucinated",
            "ml_confidence": 0.0,
            "features": {},
            "claims": [],
            "claims_total": 0,
            "claims_grounded": 0,
            "claims_partial": 0,
            "claims_unsupported": 0,
        }

    # 1. Claim-Level Evidence Grounding (Primary Verification)
    claims = split_into_sentence_claims(answer)
    clean_k = re.sub(r"\[[^\]]+\]", "", combined_context)
    clean_a = re.sub(r"\[[^\]]+\]", "", answer)
    evidence_texts = [re.sub(r"\[[^\]]+\]", "", e["text"]) for e in retrieved_sources]

    # Ultra-fast single batched embedding pass for ALL items
    encoder = get_encoder_model()
    batch_texts = claims + evidence_texts + [clean_k, question, clean_a]
    batch_embs = encoder.encode(batch_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)

    n_c = len(claims)
    n_e = len(evidence_texts)
    c_embs = batch_embs[:n_c]
    e_embs = batch_embs[n_c:n_c + n_e]
    k_emb = batch_embs[n_c + n_e]
    q_emb = batch_embs[n_c + n_e + 1]
    a_emb = batch_embs[n_c + n_e + 2]

    claim_verifications = verify_claims_against_evidence(
        claims, retrieved_sources, claim_embeddings=c_embs, evidence_embeddings=e_embs
    )

    claims_total = len(claim_verifications)
    grounded_count = sum(1 for c in claim_verifications if c["status"] == "GROUNDED")
    partial_count = sum(1 for c in claim_verifications if c["status"] == "PARTIALLY_GROUNDED")
    unsupported_count = sum(1 for c in claim_verifications if c["status"] == "HALLUCINATED")

    # 2. XGBoost Feature Computation (reusing batch embeddings)
    features_df, features_dict = calculate_ml_features(
        combined_context, question, answer,
        knowledge_embedding=k_emb, question_embedding=q_emb, answer_embedding=a_emb
    )
    detector = get_detector_model()

    if detector is not None:
        try:
            pred = detector.predict(features_df)[0]
            prob = detector.predict_proba(features_df)[0]
            is_faithful_ml = (pred == 0)
            ml_confidence = float(prob[0] * 100) if is_faithful_ml else float(prob[1] * 100)
            ml_pred_label = "Faithful" if is_faithful_ml else "Hallucinated"
        except Exception:
            is_faithful_ml = features_dict["knowledge_answer_similarity"] >= 0.60
            ml_confidence = round(features_dict["knowledge_answer_similarity"] * 100, 1)
            ml_pred_label = "Faithful" if is_faithful_ml else "Hallucinated"
    else:
        is_faithful_ml = features_dict["knowledge_answer_similarity"] >= 0.60
        ml_confidence = round(features_dict["knowledge_answer_similarity"] * 100, 1)
        ml_pred_label = "Faithful" if is_faithful_ml else "Hallucinated"

    # 3. Ensemble Calibration
    if claims_total > 0:
        base_claim_score = ((grounded_count * 1.0) + (partial_count * 0.5)) / claims_total * 100.0
    else:
        base_claim_score = 92.0 if features_dict["knowledge_answer_similarity"] > 0.65 else 50.0

    # Calibrated composite:
    if unsupported_count == 0 and grounded_count == claims_total:
        # 100% of claims are directly substantiated in institutional documents!
        faithfulness_score = min(100.0, max(95.0, base_claim_score))
        hallucination_risk = "LOW RISK (Faithful)"
        risk_color = "success"
        verified = True
    elif unsupported_count == 0:
        # High grounding with minor ambiguous phrasing
        faithfulness_score = round(max(85.0, base_claim_score), 1)
        hallucination_risk = "LOW RISK (Faithful)"
        risk_color = "success"
        verified = True
    elif unsupported_count == 1 and grounded_count >= 2:
        # Exactly one isolated unsupported statement
        faithfulness_score = round(max(50.0, min(68.0, base_claim_score)), 1)
        hallucination_risk = "MODERATE RISK (Partially Verified)"
        risk_color = "warning"
        verified = False
    else:
        # Multiple unsupported/hallucinated claims
        faithfulness_score = round(min(40.0, base_claim_score), 1)
        hallucination_risk = "HIGH RISK (Potential Hallucination)"
        risk_color = "danger"
        verified = False

    return {
        "verified": verified,
        "faithfulness_score": round(faithfulness_score, 1),
        "hallucination_risk": hallucination_risk,
        "risk_color": risk_color,
        "ml_prediction": ml_pred_label,
        "ml_confidence": round(ml_confidence, 1),
        "features": features_dict,
        "claims": claim_verifications,
        "claims_total": claims_total,
        "claims_grounded": grounded_count,
        "claims_partial": partial_count,
        "claims_unsupported": unsupported_count,
    }
