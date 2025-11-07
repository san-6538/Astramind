from rank_bm25 import BM25Okapi
import pickle
import os
import re
from typing import List, Dict, Any


class BM25Retriever:
    """
    🧠 Enhanced BM25 Retriever for AstraMind
    ---------------------------------------
    - Keyword-based retrieval for hybrid (dense + sparse) RAG
    - Supports persistence across restarts (via pickle)
    - Compatible with Pinecone hybrid search
    - Includes lightweight token cleaning and normalization
    """

    def __init__(self, storage_path: str = "bm25_index.pkl"):
        self.storage_path = storage_path
        self.documents: List[str] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25 = None
        self._load()

    # ------------------------------------------------------------
    # ✅ Text Cleaning & Tokenization
    # ------------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        """Clean text: remove symbols and extra spaces."""
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text.strip().lower())

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text safely (clean + lowercase split)."""
        cleaned = self._clean_text(text)
        return cleaned.split()

    # ------------------------------------------------------------
    # ✅ Add Documents
    # ------------------------------------------------------------
    def add_documents(self, docs: List[str]):
        """
        Add new documents to the BM25 index.
        Automatically skips duplicates and re-trains the index.
        """
        if not docs:
            print("⚠️ No documents provided for BM25 indexing.")
            return

        new_docs = [d for d in docs if d.strip() and d not in self.documents]
        if not new_docs:
            print("ℹ️ No new or unique documents to add.")
            return

        # Tokenize and update corpus
        self.documents.extend(new_docs)
        self.tokenized_docs.extend([self._tokenize(doc) for doc in new_docs])

        # Rebuild BM25 model
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self._save()
        print(f"✅ BM25 index updated with {len(new_docs)} new documents. Total: {len(self.documents)}")

    # ------------------------------------------------------------
    # ✅ Search / Retrieve
    # ------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents for a query using BM25 scores.
        Returns a list of dicts with {text, score}.
        """
        if not self.bm25 or not self.documents:
            print("⚠️ BM25 index is empty. Add documents first.")
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            print("⚠️ Empty query provided to BM25.")
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)
        results = [{"text": doc, "score": float(score)} for doc, score in ranked[:top_k]]

        return results

    # Alias for compatibility with hybrid_search
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Alias for retrieve() to maintain compatibility."""
        return self.retrieve(query, top_k)

    # ------------------------------------------------------------
    # ✅ Reset BM25 Index
    # ------------------------------------------------------------
    def reset(self):
        """Completely clears BM25 index and persistent file."""
        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None

        if os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
                print("🧹 Removed BM25 persistent file.")
            except Exception as e:
                print(f"⚠️ Failed to remove BM25 file: {e}")

        print("✅ BM25 index reset successful.")

    # ------------------------------------------------------------
    # ✅ Save & Load
    # ------------------------------------------------------------
    def _save(self):
        """Persist the BM25 index to disk."""
        try:
            with open(self.storage_path, "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "tokenized_docs": self.tokenized_docs
                }, f)
            print(f"💾 BM25 index saved ({len(self.documents)} docs).")
        except Exception as e:
            print(f"⚠️ Failed to save BM25 index: {e}")

    def _load(self):
        """Load saved BM25 documents from disk (if available)."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data.get("documents", [])
                    self.tokenized_docs = data.get("tokenized_docs", [])

                if self.documents:
                    self.bm25 = BM25Okapi(self.tokenized_docs)
                    print(f"✅ Loaded BM25 index with {len(self.documents)} documents.")
            except Exception as e:
                print(f"⚠️ Failed to load BM25 index: {e}")

    # ------------------------------------------------------------
    # ✅ Utility Functions
    # ------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Return quick stats for BM25 index."""
        return {
            "total_docs": len(self.documents),
            "is_initialized": self.bm25 is not None
        }

    def has_data(self) -> bool:
        """Check if BM25 index contains any data."""
        return bool(self.documents)


# ------------------------------------------------------------
# ✅ Global Instance
# ------------------------------------------------------------
bm25 = BM25Retriever()
