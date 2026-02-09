import ollama
from typing import List
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "nomic-embed-text"):
        """
        Initialize the Embedder with Ollama model.

        Args:
            model_name: Name of the embedding model
        """
        self.model_name = model_name
        self.embedding_dim = None
        self._verify_model()

    def _verify_model(self):
        """Verify that the model is available in Ollama."""
        try:
            models = ollama.list()
            available_models = [model['model'] for model in models['models']]

            if not any(self.model_name in model for model in available_models):
                print(f"Warning: Model '{self.model_name}' not found.")
                print(f"Run: ollama pull {self.model_name}")
            else:
                print(f"Model '{self.model_name}' is ready")
        except Exception as e:
            print(f"Warning: Could not verify model: {e}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = ollama.embeddings(
                model=self.model_name,
                prompt=text
            )

            embedding = response['embedding']

            if self.embedding_dim is None:
                self.embedding_dim = len(embedding)
                print(f"Embedding dimension: {self.embedding_dim}")

            return embedding

        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        total = len(texts)

        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 10 == 0:
                print(f"Processing: {i + 1}/{total}")

            embedding = self.embed_text(text)
            embeddings.append(embedding)

        if show_progress:
            print(f"Completed: {total}/{total}")

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self.embedding_dim is None:
            dummy_embedding = self.embed_text("test")
            self.embedding_dim = len(dummy_embedding)
        return self.embedding_dim

    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


# Usage with chat
def chat_with_ollama(model: str = "llama2"):
    """Simple chat function."""
    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'user', 'content': 'Explain quantum computing in simple terms'}
        ]
    )
    return response['message']['content']


# Example usage
if __name__ == "__main__":
    # Embeddings
    embedder = Embedder(model_name="nomic-embed-text")

    text = "Machine learning is fascinating"
    embedding = embedder.embed_text(text)
    print(f"Embedding length: {len(embedding)}")

    # Chat
    print("\n--- Chat Example ---")
    answer = chat_with_ollama()
    print(answer)

    # List available models
    print("\n--- Available Models ---")
    models = ollama.list()
    for model in models['models']:
        print(f"- {model['name']}")