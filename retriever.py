"""
Multi-Source Retrieval Engine
Combines dense semantic vector search with keyword/course-code matching.
Fuses and ranks evidence chunks across multiple institutional sources.
"""
import re
from typing import List, Dict, Any, Optional
from vector_store import KnowledgeRepository


def keyword_overlap_score(query: str, text: str) -> float:
    """Compute normalized token overlap between query and document chunk."""
    q_tokens = set(re.findall(r"\b\w{3,}\b", query.lower()))
    if not q_tokens:
        return 0.0
    t_tokens = set(re.findall(r"\b\w{3,}\b", text.lower()))
    intersection = q_tokens.intersection(t_tokens)
    return len(intersection) / len(q_tokens)


def course_code_bonus(query: str, chunk: Dict[str, Any]) -> float:
    """Give strong priority if specific course code or document identifier is mentioned."""
    q_upper = query.upper()
    course_code = (chunk.get("course_code") or "").upper().replace(" ", "")
    filename = (chunk.get("filename") or "").upper()
    if course_code and course_code != "INST" and course_code in q_upper:
        return 0.35
    # Check for keywords like "CS401", "DS302", "REG-2025", "ETH-2025"
    codes_in_query = re.findall(r"\b[A-Z]{2,4}\s*\d{3,4}\b", q_upper)
    for code in codes_in_query:
        norm_code = code.replace(" ", "")
        if norm_code in filename or norm_code in course_code:
            return 0.35
    return 0.0


def retrieve_multi_source_evidence(
    repo: KnowledgeRepository,
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
    min_similarity: float = 0.25,
) -> Dict[str, Any]:
    """
    Perform hybrid retrieval from the repository:
    1. Dense vector search.
    2. Course-code and keyword relevance boosting.
    3. Fused ranking and context packaging.
    """
    # 1. Retrieve candidate vector chunks (fetch slightly more candidates for re-ranking)
    raw_results = repo.search(query=query, top_k=max(top_k * 3, 10), category=category)

    if not raw_results:
        return {
            "sources": [],
            "combined_context": "",
            "total_found": 0,
            "distinct_documents": 0,
        }

    # 2. Score and re-rank with hybrid fusion
    ranked_chunks = []
    for chunk in raw_results:
        vec_sim = chunk.get("similarity_score", 0.0)
        kw_sim = keyword_overlap_score(query, chunk["text"])
        bonus = course_code_bonus(query, chunk)

        # Fused score: 60% semantic similarity + 25% keyword overlap + bonus
        fused_score = (vec_sim * 0.60) + (kw_sim * 0.25) + bonus
        fused_score = min(1.0, fused_score)

        if fused_score >= min_similarity:
            ranked_chunks.append({
                **chunk,
                "fused_score": round(fused_score, 4),
                "vector_score": round(vec_sim, 4),
                "keyword_score": round(kw_sim, 4),
            })

    # Sort descending by fused score
    ranked_chunks.sort(key=lambda x: x["fused_score"], reverse=True)
    top_chunks = ranked_chunks[:top_k]

    # If all were below min_similarity, take top 2 raw results to prevent total blank if query is vague
    if not top_chunks and raw_results:
        top_chunks = raw_results[:2]
        for c in top_chunks:
            c["fused_score"] = c.get("similarity_score", 0.3)

    # 3. Format structured combined context with source markers
    context_blocks = []
    distinct_docs = set()

    for idx, c in enumerate(top_chunks, 1):
        doc_name = c.get("filename", "Document")
        page_num = c.get("page_number", 1)
        course_cd = c.get("course_code", "Institutional")
        distinct_docs.add(doc_name)

        header = f"[Source {idx} | File: {doc_name} | Page: {page_num} | Ref: {course_cd}]"
        context_blocks.append(f"{header}\n{c['text']}")

    combined_context = "\n\n".join(context_blocks)

    return {
        "sources": top_chunks,
        "combined_context": combined_context,
        "total_found": len(top_chunks),
        "distinct_documents": len(distinct_docs),
    }
