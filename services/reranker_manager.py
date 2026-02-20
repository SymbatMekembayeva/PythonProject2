"""
Reranker Manager for RAG System
Handles reranking of retrieved documents using Cohere Rerank API
"""

from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RerankerManager:
    """Manages document reranking using Cohere Rerank API."""

    def __init__(
            self,
            api_key: str = None,
            model: str = "rerank-english-v3.0",
            enabled: bool = True
    ):
        """
        Initialize Reranker Manager.

        Args:
            api_key: Cohere API key (or set COHERE_API_KEY env var)
            model: Cohere rerank model to use
            enabled: Whether reranking is enabled
        """
        self.enabled = enabled
        self.model = model

        if not enabled:
            logger.info("Reranker is disabled")
            return

        # Import cohere only if needed
        try:
            import cohere
            import os

            # Get API key
            self.api_key = api_key or os.getenv('COHERE_API_KEY')

            if not self.api_key:
                logger.warning("No Cohere API key provided. Reranker will be disabled.")
                self.enabled = False
                return

            # Initialize Cohere client
            self.client = cohere.Client(self.api_key)
            logger.info(f"Reranker initialized with model: {model}")

        except ImportError:
            logger.error("Cohere package not installed. Run: pip install cohere")
            logger.error("Reranker will be disabled.")
            self.enabled = False
        except Exception as e:
            logger.error(f"Error initializing reranker: {e}")
            self.enabled = False

    def rerank(
            self,
            query: str,
            documents: List[Dict[str, Any]],
            top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: Search query
            documents: List of document dicts with 'text' and 'metadata'
            top_n: Number of top documents to return

        Returns:
            Reranked list of top_n documents with updated scores
        """
        # If reranker is disabled, return original documents
        if not self.enabled:
            logger.debug("Reranker disabled, returning original documents")
            return documents[:top_n]

        # If no documents or fewer than top_n, return as is
        if not documents or len(documents) <= top_n:
            return documents

        try:
            # Extract text content from documents
            doc_texts = [doc['text'] for doc in documents]

            logger.info(f"Reranking {len(documents)} documents, selecting top {top_n}")

            # Call Cohere Rerank API
            results = self.client.rerank(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=top_n,
                return_documents=False  # We already have the documents
            )

            # Map reranked results back to original documents
            reranked_docs = []
            for result in results.results:
                # Get original document by index
                original_doc = documents[result.index].copy()

                # Update with rerank score
                original_doc['rerank_score'] = result.relevance_score
                # Keep original similarity score for reference
                if 'score' in original_doc:
                    original_doc['original_score'] = original_doc['score']
                original_doc['score'] = result.relevance_score

                reranked_docs.append(original_doc)

            logger.info(f"Reranking complete. Returned {len(reranked_docs)} documents")

            # Log score changes for debugging
            if logger.isEnabledFor(logging.DEBUG):
                for i, doc in enumerate(reranked_docs[:3]):
                    orig_score = doc.get('original_score', 'N/A')
                    new_score = doc.get('rerank_score', 'N/A')
                    logger.debug(f"  Top {i + 1}: {orig_score} → {new_score}")

            return reranked_docs

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            logger.warning("Falling back to original ranking")
            return documents[:top_n]

    def is_enabled(self) -> bool:
        """Check if reranker is enabled and ready."""
        return self.enabled


# Example usage
if __name__ == "__main__":
    # Example documents
    sample_docs = [
        {
            "text": "RAG stands for Retrieval-Augmented Generation. It enhances LLMs.",
            "score": 0.85,
            "metadata": {"source": "doc1.pdf"}
        },
        {
            "text": "The weather is nice today. It's sunny and warm.",
            "score": 0.82,
            "metadata": {"source": "doc2.pdf"}
        },
        {
            "text": "RAG reduces hallucinations by using external knowledge sources.",
            "score": 0.80,
            "metadata": {"source": "doc3.pdf"}
        },
        {
            "text": "Machine learning is a subset of artificial intelligence.",
            "score": 0.78,
            "metadata": {"source": "doc4.pdf"}
        }
    ]

    # Initialize reranker (you'll need to set COHERE_API_KEY env var)
    reranker = RerankerManager(enabled=True)

    if reranker.is_enabled():
        query = "How does RAG prevent hallucinations?"

        print(f"Query: {query}\n")
        print("Original ranking:")
        for i, doc in enumerate(sample_docs, 1):
            print(f"{i}. (score: {doc['score']:.2f}) {doc['text'][:60]}...")

        # Rerank
        reranked = reranker.rerank(query, sample_docs, top_n=3)

        print("\nAfter reranking:")
        for i, doc in enumerate(reranked, 1):
            orig = doc.get('original_score', 'N/A')
            new = doc.get('rerank_score', 'N/A')
            print(f"{i}. (orig: {orig}, rerank: {new}) {doc['text'][:60]}...")
    else:
        print("Reranker is not enabled. Set COHERE_API_KEY environment variable.")