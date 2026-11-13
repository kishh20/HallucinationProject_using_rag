"""
Multi-Provider LLM Answer Generation Engine
Supports Google Gemini (gemini-3-flash-preview, gemini-flash-latest, etc.),
OpenAI, Anthropic Claude, OpenRouter, and Local Extractive Fallback.
Enforces grounded generation with strict source citations.
"""
import os
import re
import json
import requests
from typing import Dict, Any, List, Optional


SYSTEM_PROMPT = """You are an official Institutional Academic AI Assistant.
Your mission is to answer user queries accurately and strictly based on the provided institutional documents, course syllabi, curricula, and university regulations.

STRICT GROUNDING & CITATION RULES:
1. Base every single claim strictly and exclusively on the provided context. Never hallucinate course codes, dates, percentages, syllabus topics, or rules.
2. For each key claim or fact, cite the exact source using bracketed citations, e.g.: [University_Academic_Regulations_and_Grading_Policy_2025.txt, Page 1] or [CS401_Syllabus, Page 2].
3. If the provided context does not contain the answer, explicitly state: "The indexed institutional documents do not contain information to answer this query." Do NOT guess or use ungrounded external assumptions.
4. Structure your response clearly using markdown bullet points, bold key terms, and concise paragraphs.
"""


def format_prompt(question: str, context: str) -> str:
    return f"""INSTITUTIONAL KNOWLEDGE CONTEXT:
----------------------------------------
{context}
----------------------------------------

USER QUERY:
{question}

Please provide an accurate, grounded answer with clear citations referencing the source documents and page numbers above:"""


# ============================================================
# PROVIDER IMPLEMENTATIONS
# ============================================================

_GEMINI_CLIENT = None
_LAST_GEMINI_KEY = None


def _get_gemini_client(api_key: str):
    global _GEMINI_CLIENT, _LAST_GEMINI_KEY
    if _GEMINI_CLIENT is None or _LAST_GEMINI_KEY != api_key:
        from google import genai
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
        _LAST_GEMINI_KEY = api_key
    return _GEMINI_CLIENT


def generate_with_gemini(
    question: str,
    context: str,
    api_key: str,
    model_name: str = "gemini-3-flash-preview"
) -> str:
    """Generate answer using Google Gemini with automatic model rotation on deprecation/outage."""
    prompt = format_prompt(question, context)
    full_prompt = SYSTEM_PROMPT + "\n\n" + prompt

    # Candidate models to try in order
    candidate_models = [model_name]
    fallback_pool = ["gemini-3-flash-preview", "gemini-flash-latest", "gemini-3.8-flash"]
    for m in fallback_pool:
        if m not in candidate_models:
            candidate_models.append(m)

    last_error = None
    for m in candidate_models:
        # 1. Try google-genai SDK with pooled client
        try:
            client = _get_gemini_client(api_key)
            response = client.models.generate_content(
                model=m,
                contents=[full_prompt]
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e_sdk:
            last_error = e_sdk

        # 2. Try REST API endpoint
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": full_prompt}]}]
            }
            res = requests.post(url, json=payload, timeout=25)
            if res.status_code == 200:
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                last_error = Exception(f"Gemini API error ({res.status_code}) for model {m}: {res.text}")
        except Exception as e_rest:
            last_error = e_rest

    raise last_error or Exception("Failed to generate response from Gemini API.")


def generate_with_openai(question: str, context: str, api_key: str, model_name: str = "gpt-4o-mini") -> str:
    """Generate answer using OpenAI Chat Completions API."""
    prompt = format_prompt(question, context)
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
    }
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    if res.status_code == 200:
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"OpenAI API error ({res.status_code}): {res.text}")


def generate_with_claude(question: str, context: str, api_key: str, model_name: str = "claude-3-5-sonnet-20241022") -> str:
    """Generate answer using Anthropic Claude API."""
    prompt = format_prompt(question, context)
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model_name,
        "max_tokens": 1500,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    if res.status_code == 200:
        data = res.json()
        return data["content"][0]["text"].strip()
    else:
        raise Exception(f"Anthropic Claude API error ({res.status_code}): {res.text}")


def generate_with_openrouter(question: str, context: str, api_key: str, model_name: str = "nvidia/nemotron-3.5-lightning:free") -> str:
    """Generate answer using OpenRouter API."""
    prompt = format_prompt(question, context)
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Institutional RAG Hallucination Detector",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
    }
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    if res.status_code == 200:
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"OpenRouter API error ({res.status_code}): {res.text}")


def local_extractive_synthesizer(question: str, context: str, sources: List[Dict[str, Any]]) -> str:
    """
    Zero-key extractive synthesizer.
    Selects complete, clean sentences from top chunks matching the query.
    Never outputs truncated fragments or duplicate lines.
    """
    if not sources or not context.strip():
        return "The provided institutional documents do not contain relevant information to answer this query."

    q_words = set(re.findall(r"\b\w{3,}\b", question.lower()))
    seen_sentences = set()
    scored_sentences = []

    for src in sources:
        doc_name = src.get("filename", "Institutional Document")
        page = src.get("page_number", 1)
        raw_text = src.get("text", "")

        # Split into individual sentences
        for raw_line in re.split(r"(?<=[.!?])\s+|\n+", raw_text):
            line = raw_line.strip().strip("*#- ")
            # Ensure it is a complete, well-formed sentence
            if len(line) < 25 or len(line.split()) < 5:
                continue
            # Filter out truncated word fragments (e.g. lines starting with punctuation or odd fragments)
            if not line[0].isalnum() and line[0] not in ['"', "'"]:
                continue
            # Normalize for deduplication
            norm_key = re.sub(r"\W+", "", line.lower())[:80]
            if norm_key in seen_sentences:
                continue
            seen_sentences.add(norm_key)

            line_words = set(re.findall(r"\b\w{3,}\b", line.lower()))
            overlap = len(q_words.intersection(line_words))

            scored_sentences.append({
                "sentence": line,
                "score": overlap,
                "citation": f"[{doc_name}, Page {page}]"
            })

    scored_sentences.sort(key=lambda x: x["score"], reverse=True)
    top_sentences = [s for s in scored_sentences if s["score"] > 0][:5]

    if not top_sentences and scored_sentences:
        top_sentences = scored_sentences[:3]

    if not top_sentences:
        top_src = sources[0]
        return f"According to **{top_src.get('filename')}** (Page {top_src.get('page_number', 1)}):\n\n{top_src.get('text')[:300]}... [{top_src.get('filename')}, Page {top_src.get('page_number', 1)}]"

    paragraphs = ["Based on the retrieved institutional documents:"]
    for s in top_sentences:
        paragraphs.append(f"* {s['sentence']} {s['citation']}")

    return "\n\n".join(paragraphs)


# ============================================================
# MASTER GENERATOR DISPATCHER
# ============================================================

def generate_grounded_answer(
    question: str,
    context: str,
    sources: List[Dict[str, Any]],
    provider: str = "Gemini",
    api_key: str = "",
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Dispatches generation to the chosen provider and returns the answer and metadata.
    NEVER appends runtime error strings into the answer text.
    """
    if not context.strip() or not sources:
        return {
            "answer": "The indexed institutional knowledge repository does not contain relevant documents for this query. Please upload or index relevant syllabi, curricula, or regulation documents.",
            "provider": provider,
            "status": "no_context",
            "error_note": None,
        }

    provider_clean = provider.strip().lower()

    try:
        if "gemini" in provider_clean:
            key = api_key or os.getenv("GEMINI_API_KEY", "")
            if not key:
                ans = local_extractive_synthesizer(question, context, sources)
                return {"answer": ans, "provider": "Local Extractive (Offline)", "status": "success", "error_note": None}
            # Use active model
            model = model_name or "gemini-3-flash-preview"
            ans = generate_with_gemini(question, context, key, model)
            return {"answer": ans, "provider": f"Gemini ({model})", "status": "success", "error_note": None}

        elif "openai" in provider_clean:
            key = api_key or os.getenv("OPENAI_API_KEY", "")
            if not key:
                ans = local_extractive_synthesizer(question, context, sources)
                return {"answer": ans, "provider": "Local Extractive (Offline)", "status": "success", "error_note": None}
            model = model_name or "gpt-4o-mini"
            ans = generate_with_openai(question, context, key, model)
            return {"answer": ans, "provider": f"OpenAI ({model})", "status": "success", "error_note": None}

        elif "claude" in provider_clean or "anthropic" in provider_clean:
            key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
            if not key:
                ans = local_extractive_synthesizer(question, context, sources)
                return {"answer": ans, "provider": "Local Extractive (Offline)", "status": "success", "error_note": None}
            model = model_name or "claude-3-5-sonnet-20241022"
            ans = generate_with_claude(question, context, key, model)
            return {"answer": ans, "provider": f"Claude ({model})", "status": "success", "error_note": None}

        elif "openrouter" in provider_clean or "router" in provider_clean:
            key = api_key or os.getenv("OPENROUTER_API_KEY", "")
            if not key:
                ans = local_extractive_synthesizer(question, context, sources)
                return {"answer": ans, "provider": "Local Extractive (Offline)", "status": "success", "error_note": None}
            model = model_name or "nvidia/nemotron-3.5-lightning:free"
            ans = generate_with_openrouter(question, context, key, model)
            return {"answer": ans, "provider": f"OpenRouter ({model})", "status": "success", "error_note": None}

        else:
            ans = local_extractive_synthesizer(question, context, sources)
            return {"answer": ans, "provider": "Local Extractive Synthesizer", "status": "success", "error_note": None}

    except Exception as e:
        # Fallback gracefully to clean extractive synthesizer without corrupting the answer with error text!
        fallback_ans = local_extractive_synthesizer(question, context, sources)
        return {
            "answer": fallback_ans,
            "provider": f"{provider} (Local Extractive Fallback)",
            "status": "fallback_error",
            "error_note": f"Remote model unavailable ({str(e)[:160]}...). Synthesized locally from documents.",
        }
