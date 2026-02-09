import uuid
from typing import List, Optional

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class QdrantManager:
    def __init__(self, qdrant_url: str = "http://localhost:6333", ollama_url: str = "http://localhost:11434"):
        """
        Initialize QdrantManager with local Qdrant and Ollama connections.

        Args:
            qdrant_url: URL for local Qdrant instance
            ollama_url: URL for local Ollama instance
        """
        self.client = QdrantClient(url=qdrant_url)
        self.ollama_url = ollama_url
        self.embedding_model = "nomic-embed-text:latest"

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector from Ollama's nomic-embed-text model.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        response = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={
                "model": self.embedding_model,
                "prompt": text
            }
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def create_collection(self, name: str, vector_size: int = 768):
        """
        Create a new collection in Qdrant.
        nomic-embed-text produces 768-dimensional vectors by default.

        Args:
            name: Name of the collection to create
            vector_size: Dimension of vectors (768 for nomic-embed-text)
        """
        try:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Collection '{name}' created successfully")
        except Exception as e:
            print(f"Error creating collection: {e}")

    def insert_point(self, embedding: List[float], collection_name: str, chunk_text: str,
                     metadata: Optional[dict] = None):
        """
        Insert a point (embedding + text) into a Qdrant collection.

        Args:
            embedding: The embedding vector
            collection_name: Name of the collection
            chunk_text: The original text chunk
            metadata: Optional additional metadata to store
        """
        point_id = str(uuid.uuid4())

        payload = {
            "text": chunk_text,
            **(metadata or {})
        }

        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            print(f"Point inserted with ID: {point_id}")
        except Exception as e:
            print(f"Error inserting point: {e}")

    def insert_points_batch(self, embeddings: List[List[float]], collection_name: str, chunk_texts: List[str],
                            metadata_list: Optional[List[dict]] = None):
        """
        Insert multiple points in batch for better performance.

        Args:
            embeddings: List of embedding vectors
            collection_name: Name of the collection
            chunk_texts: List of text chunks
            metadata_list: Optional list of metadata dicts
        """
        if len(embeddings) != len(chunk_texts):
            raise ValueError("Number of embeddings must match number of texts")

        points = []
        for i, (embedding, text) in enumerate(zip(embeddings, chunk_texts)):
            payload = {
                "text": text,
                **(metadata_list[i] if metadata_list and i < len(metadata_list) else {})
            }

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )
            points.append(point)

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"Batch inserted {len(points)} points")
        except Exception as e:
            print(f"Error inserting batch: {e}")

    def search_point(self, embedding: List[float], collection_name: str, top_k: int = 5,
                     score_threshold: float = 0.75) -> List[str]:
        """
        Search for similar points in the collection.

        Args:
            embedding: Query embedding vector
            collection_name: Name of the collection to search
            top_k: Number of top results to return
            score_threshold: Minimum similarity score (0-1 for cosine)

        Returns:
            List of text chunks that match the query
        """
        try:
            # Use query_points for newer qdrant-client versions
            results = self.client.query_points(
                collection_name=collection_name,
                query=embedding,
                limit=top_k,
                score_threshold=score_threshold
            )

            # Extract text from results
            texts = [hit.payload.get("text", "") for hit in results.points]
            return texts
        except AttributeError:
            # Fallback for older versions
            try:
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=embedding,
                    limit=top_k,
                    score_threshold=score_threshold
                )
                texts = [hit.payload.get("text", "") for hit in results]
                return texts
            except Exception as e:
                print(f"Error searching (fallback): {e}")
                return []
        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def search_by_text(self, query_text: str, collection_name: str, top_k: int = 5, score_threshold: float = 0.75) -> \
    List[dict]:
        """
        Search using text query (automatically converts to embedding).

        Args:
            query_text: Text query to search for
            collection_name: Name of the collection to search
            top_k: Number of top results to return
            score_threshold: Minimum similarity score

        Returns:
            List of dicts containing text and score
        """
        # Get embedding for query text
        query_embedding = self.get_embedding(query_text)

        try:
            # Try newer API first (query_points)
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=score_threshold
            )

            # Return text and scores
            return [
                {
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
                }
                for hit in results.points
            ]
        except AttributeError:
            # Fallback for older API (search)
            try:
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    score_threshold=score_threshold
                )

                return [
                    {
                        "text": hit.payload.get("text", ""),
                        "score": hit.score,
                        "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
                    }
                    for hit in results
                ]
            except Exception as e:
                print(f"Error searching (fallback): {e}")
                return []
        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def delete_collection(self, name: str):
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name=name)
            print(f"Collection '{name}' deleted successfully")
        except Exception as e:
            print(f"Error deleting collection: {e}")

    def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []

    def get_collection_info(self, name: str):
        """Get information about a collection."""
        try:
            return self.client.get_collection(collection_name=name)
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = QdrantManager()

    # Create a collection
    collection_name = "my_documents"
    manager.create_collection(collection_name)

    # Example text chunks
    chunks = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Neural networks are inspired by biological neurons."
    ]

    # Get embeddings and insert
    for chunk in chunks:
        embedding = manager.get_embedding(chunk)
        manager.insert_point(embedding, collection_name, chunk)

    # Search by text
    query = "What is machine learning?"
    results = manager.search_by_text(query, collection_name, top_k=2)

    print("\nSearch results:")
    for i, result in enumerate(results):
        print(f"{i + 1}. Score: {result['score']:.3f}")
        print(f"   Text: {result['text']}")