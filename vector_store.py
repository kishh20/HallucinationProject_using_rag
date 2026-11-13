"""
Unified Vector Store & Searchable Knowledge Repository Engine
Supports ChromaDB, FAISS, Pinecone, and Local NumPy Cosine indexing.
Integrates Sentence-Transformers (all-MiniLM-L6-v2) for local dense retrieval.
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# Storage directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
FAISS_DIR = os.path.join(BASE_DIR, "data", "faiss_db")
LOCAL_DIR = os.path.join(BASE_DIR, "data", "local_store")

for d in [CHROMA_DIR, FAISS_DIR, LOCAL_DIR]:
    os.makedirs(d, exist_ok=True)

# Shared embedding model singleton
_EMBED_MODEL = None


def get_embedding_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        # Prevent huggingface hub network latency/timeouts
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


# ============================================================
# BASE REPOSITORY INTERFACE
# ============================================================

class BaseVectorStore:
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def list_documents(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def delete_document(self, filename: str) -> bool:
        raise NotImplementedError

    def clear(self) -> bool:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


# ============================================================
# CHROMADB STORE
# ============================================================

class ChromaDBStore(BaseVectorStore):
    def __init__(self, persist_directory: str = CHROMA_DIR):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="institutional_knowledge",
            metadata={"description": "Institutional Syllabi & Regulations"}
        )
        self.encoder = get_embedding_model()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        embeddings = self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()
        ids = [c["id"] for c in chunks]
        metadatas = []
        for c in chunks:
            metadatas.append({
                "doc_id": c.get("doc_id", ""),
                "filename": c.get("filename", "unknown"),
                "category": c.get("category", "General"),
                "course_code": c.get("course_code", "INST"),
                "department": c.get("department", "Academic"),
                "page_number": int(c.get("page_number", 1)),
                "char_length": int(c.get("char_length", len(c["text"]))),
            })

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        total = self.count()
        if total == 0:
            return []
        effective_k = min(top_k, total)
        query_emb = self.encoder.encode([query], normalize_embeddings=True).tolist()
        where_filter = {"category": category} if category and category != "All" else None

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=effective_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        output = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0] if "distances" in results else [0.0] * len(ids)

        for i in range(len(ids)):
            dist = dists[i] if i < len(dists) else 0.5
            # Convert cosine distance to similarity score
            sim = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            meta = metas[i]
            output.append({
                "id": ids[i],
                "text": docs[i],
                "filename": meta.get("filename", "Unknown"),
                "category": meta.get("category", "General"),
                "course_code": meta.get("course_code", ""),
                "department": meta.get("department", ""),
                "page_number": meta.get("page_number", 1),
                "similarity_score": round(sim, 4),
            })
        return output

    def list_documents(self) -> List[Dict[str, Any]]:
        all_data = self.collection.get(include=["metadatas"])
        if not all_data or not all_data["metadatas"]:
            return []
        doc_stats = {}
        for m in all_data["metadatas"]:
            fn = m.get("filename", "Unknown")
            if fn not in doc_stats:
                doc_stats[fn] = {
                    "filename": fn,
                    "category": m.get("category", "General"),
                    "course_code": m.get("course_code", ""),
                    "department": m.get("department", ""),
                    "chunk_count": 0,
                    "max_page": 1,
                }
            doc_stats[fn]["chunk_count"] += 1
            pg = m.get("page_number", 1)
            if pg > doc_stats[fn]["max_page"]:
                doc_stats[fn]["max_page"] = pg

        return list(doc_stats.values())

    def delete_document(self, filename: str) -> bool:
        all_data = self.collection.get(include=["metadatas"])
        if not all_data or not all_data["ids"]:
            return False
        ids_to_del = []
        for i, m in zip(all_data["ids"], all_data["metadatas"]):
            if m.get("filename") == filename:
                ids_to_del.append(i)
        if ids_to_del:
            self.collection.delete(ids=ids_to_del)
            return True
        return False

    def clear(self) -> bool:
        self.client.delete_collection("institutional_knowledge")
        self.collection = self.client.get_or_create_collection("institutional_knowledge")
        return True

    def count(self) -> int:
        return self.collection.count()


# ============================================================
# FAISS STORE
# ============================================================

class FAISSStore(BaseVectorStore):
    def __init__(self, persist_directory: str = FAISS_DIR):
        import faiss
        self.faiss = faiss
        self.persist_dir = persist_directory
        self.index_file = os.path.join(persist_directory, "faiss_index.bin")
        self.meta_file = os.path.join(persist_directory, "faiss_meta.json")
        self.encoder = get_embedding_model()
        self.dim = 384
        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            try:
                self.index = self.faiss.read_index(self.index_file)
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
            except Exception:
                self._reset()
        else:
            self._reset()

    def _reset(self):
        # IndexFlatIP with normalized vectors computes exact cosine similarity
        self.index = self.faiss.IndexFlatIP(self.dim)
        self.chunks = []
        self._save()

    def _save(self):
        if self.index is not None:
            self.faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        embeddings = self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings_np = np.array(embeddings, dtype=np.float32)

        self.index.add(embeddings_np)
        self.chunks.extend(chunks)
        self._save()
        return len(chunks)

    def search(self, query: str, top_k: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.count() == 0:
            return []
        query_emb = self.encoder.encode([query], normalize_embeddings=True)
        query_np = np.array(query_emb, dtype=np.float32)

        # Retrieve a broader pool to support metadata filtering
        search_k = min(max(top_k * 4, 15), self.count())
        scores, indices = self.index.search(query_np, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if category and category != "All" and chunk.get("category") != category:
                continue
            results.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "filename": chunk.get("filename", "Unknown"),
                "category": chunk.get("category", "General"),
                "course_code": chunk.get("course_code", ""),
                "department": chunk.get("department", ""),
                "page_number": chunk.get("page_number", 1),
                "similarity_score": round(float(score), 4),
            })
            if len(results) >= top_k:
                break
        return results

    def list_documents(self) -> List[Dict[str, Any]]:
        doc_stats = {}
        for c in self.chunks:
            fn = c.get("filename", "Unknown")
            if fn not in doc_stats:
                doc_stats[fn] = {
                    "filename": fn,
                    "category": c.get("category", "General"),
                    "course_code": c.get("course_code", ""),
                    "department": c.get("department", ""),
                    "chunk_count": 0,
                    "max_page": 1,
                }
            doc_stats[fn]["chunk_count"] += 1
            pg = c.get("page_number", 1)
            if pg > doc_stats[fn]["max_page"]:
                doc_stats[fn]["max_page"] = pg
        return list(doc_stats.values())

    def delete_document(self, filename: str) -> bool:
        remaining = [c for c in self.chunks if c.get("filename") != filename]
        if len(remaining) == len(self.chunks):
            return False
        # Rebuild FAISS index
        self._reset()
        if remaining:
            self.add_chunks(remaining)
        return True

    def clear(self) -> bool:
        self._reset()
        return True

    def count(self) -> int:
        return self.index.ntotal if self.index else 0


# ============================================================
# PINECONE STORE (REMOTE / HYBRID)
# ============================================================

class PineconeStore(BaseVectorStore):
    def __init__(self, api_key: str = "", index_name: str = "institutional-rag"):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY", "")
        self.index_name = index_name
        self.encoder = get_embedding_model()
        self.index = None
        self._local_fallback = None

        if self.api_key:
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=self.api_key)
                self.index = pc.Index(self.index_name)
            except Exception:
                # If Pinecone index not yet created or connection fails, use local fallback
                self.index = None

        if self.index is None:
            self._local_fallback = FAISSStore()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if self.index:
            try:
                texts = [c["text"] for c in chunks]
                embeddings = self.encoder.encode(texts, normalize_embeddings=True).tolist()
                vectors = []
                for c, emb in zip(chunks, embeddings):
                    vectors.append({
                        "id": c["id"],
                        "values": emb,
                        "metadata": {
                            "filename": c.get("filename", ""),
                            "category": c.get("category", "General"),
                            "course_code": c.get("course_code", ""),
                            "department": c.get("department", ""),
                            "page_number": c.get("page_number", 1),
                            "text": c["text"],
                        }
                    })
                self.index.upsert(vectors=vectors)
                return len(chunks)
            except Exception:
                pass
        return self._local_fallback.add_chunks(chunks)

    def search(self, query: str, top_k: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.index:
            try:
                query_emb = self.encoder.encode([query], normalize_embeddings=True).tolist()
                filter_dict = {"category": category} if category and category != "All" else None
                resp = self.index.query(vector=query_emb, top_k=top_k, include_metadata=True, filter=filter_dict)
                results = []
                for match in resp.get("matches", []):
                    meta = match.get("metadata", {})
                    results.append({
                        "id": match["id"],
                        "text": meta.get("text", ""),
                        "filename": meta.get("filename", "Unknown"),
                        "category": meta.get("category", "General"),
                        "course_code": meta.get("course_code", ""),
                        "department": meta.get("department", ""),
                        "page_number": meta.get("page_number", 1),
                        "similarity_score": round(float(match.get("score", 0.0)), 4),
                    })
                return results
            except Exception:
                pass
        return self._local_fallback.search(query, top_k=top_k, category=category)

    def list_documents(self) -> List[Dict[str, Any]]:
        return self._local_fallback.list_documents()

    def delete_document(self, filename: str) -> bool:
        return self._local_fallback.delete_document(filename)

    def clear(self) -> bool:
        return self._local_fallback.clear()

    def count(self) -> int:
        return self._local_fallback.count()


# ============================================================
# UNIFIED REPOSITORY MANAGER
# ============================================================

class KnowledgeRepository:
    """Unified facade managing whichever vector store is chosen by the user."""
    def __init__(self, store_type: str = "ChromaDB", pinecone_api_key: str = ""):
        self.store_type = store_type
        self.pinecone_api_key = pinecone_api_key
        self._init_store()

    def _init_store(self):
        if self.store_type == "ChromaDB":
            try:
                self.store = ChromaDBStore()
            except Exception:
                self.store = FAISSStore()
                self.store_type = "FAISS"
        elif self.store_type == "FAISS":
            self.store = FAISSStore()
        elif self.store_type == "Pinecone":
            self.store = PineconeStore(api_key=self.pinecone_api_key)
        else:
            self.store = ChromaDBStore()

    def switch_store(self, store_type: str, pinecone_api_key: str = ""):
        self.store_type = store_type
        self.pinecone_api_key = pinecone_api_key
        self._init_store()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        return self.store.add_chunks(chunks)

    def search(self, query: str, top_k: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.store.search(query, top_k=top_k, category=category)

    def list_documents(self) -> List[Dict[str, Any]]:
        return self.store.list_documents()

    def delete_document(self, filename: str) -> bool:
        return self.store.delete_document(filename)

    def clear(self) -> bool:
        return self.store.clear()

    def count(self) -> int:
        return self.store.count()

    def seed_sample_documents_if_empty(self) -> int:
        """Automatically indexes built-in institutional documents if repository is empty."""
        if self.count() > 0:
            return 0
        from document_processor import process_institutional_document
        sample_dir = os.path.join(BASE_DIR, "sample_documents")
        if not os.path.exists(sample_dir):
            return 0

        total_seeded = 0
        for fname in os.listdir(sample_dir):
            fpath = os.path.join(sample_dir, fname)
            if not os.path.isfile(fpath):
                continue
            with open(fpath, "rb") as f:
                content = f.read()

            category = "Curriculum & Syllabi" if "Syllabus" in fname or "Curriculum" in fname else "Academic Regulations"
            chunks = process_institutional_document(
                file_bytes=content,
                filename=fname,
                category=category,
                chunk_size=650,
                chunk_overlap=120
            )
            self.add_chunks(chunks)
            total_seeded += len(chunks)

        return total_seeded
