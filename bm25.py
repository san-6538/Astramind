from rank_bm25 import BM25Okapi
import pickle
import os

class BM25Retriever:
    """
    BM25 Keyword-based retriever with optional persistence and hybrid support.
    """

    def __init__(self, storage_path="bm25_index.pkl"):
        self.storage_path = storage_path
        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None
        self._load()

    def _tokenize(self, text: str):
        """Tokenize text safely (lowercase split)."""
        return text.lower().split()

    def add_documents(self, docs: list[str]):
        """Add new documents to BM25 index (duplicates ignored)."""
        if not docs:
            print("⚠️ No documents provided for BM25 indexing.")
            return

        new_docs = [d for d in docs if d not in self.documents]
        if not new_docs:
            print("ℹ️ No new documents to add to BM25 index.")
            return

        self.documents.extend(new_docs)
        self.tokenized_docs.extend([self._tokenize(doc) for doc in new_docs])
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self._save()
        print(f"✅ Added {len(new_docs)} documents to BM25 index.")

    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve top matches for a query."""
        if not self.bm25 or not self.documents:
            print("⚠️ BM25 is empty. Add documents first.")
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)

        results = [{"text": doc, "score": float(score)} for doc, score in ranked[:top_k]]
        return results

    # ✅ Alias for compatibility with hybrid_search
    def search(self, query: str, top_k: int = 5):
        """Alias for retrieve() to maintain backward compatibility."""
        return self.retrieve(query, top_k)

    def reset(self):
        """Clear BM25 index and storage."""
        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None

        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

        print("✅ BM25 reset successful.")

    def _save(self):
        """Save BM25 documents to disk."""
        try:
            with open(self.storage_path, "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "tokenized_docs": self.tokenized_docs
                }, f)
        except Exception as e:
            print(f"⚠️ Failed to save BM25 index: {e}")

    def _load(self):
        """Load saved BM25 documents (if any)."""
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

# Global instance
bm25 = BM25Retriever()
