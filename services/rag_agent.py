"""
RAG Agent for Question Answering
Uses Qdrant for retrieval and Gemini for generation.
"""

from typing import List, Dict, Any
from qdrant_manager import QdrantManager
from google import genai
import os


class RAGAgent:
    """RAG agent that answers questions using retrieved document context."""

    def __init__(
            self,
            collection_name: str = "pdf_documents",
            gemini_api_key: str = None,
            gemini_model: str = "gemini-2.5-flash",  # Changed to stable model
            qdrant_url: str = "http://localhost:6333",
            ollama_url: str = "http://localhost:11434",
            top_k: int = 3,
            score_threshold: float = 0.3
    ):
        """
        Initialize RAG Agent.

        Args:
            collection_name: Qdrant collection name
            gemini_api_key: Google Gemini API key (or set GEMINI_API_KEY env var)
            gemini_model: Gemini model name (gemini-1.5-flash or gemini-1.5-pro)
            qdrant_url: Qdrant server URL
            ollama_url: Ollama URL for embeddings
            top_k: Number of chunks to retrieve (3-5 recommended, lower = faster)
            score_threshold: Minimum similarity score for retrieval
        """
        self.collection_name = collection_name
        self.gemini_model = gemini_model
        self.top_k = top_k
        self.score_threshold = score_threshold

        # Initialize Gemini with new API
        print("Initializing RAG Agent...")
        api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable or pass gemini_api_key parameter.")

        self.client = genai.Client(api_key=api_key)

        # Initialize Qdrant manager
        self.qdrant_manager = QdrantManager(
            qdrant_url=qdrant_url,
            ollama_url=ollama_url  # Still used for embeddings
        )

        print(f"RAG Agent initialized with:")
        print(f"  - Collection: {collection_name}")
        print(f"  - LLM model: {gemini_model}")
        print(f"  - Retrieval: top_{top_k}, threshold={score_threshold}")

    def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        Uses top_k and score_threshold settings.

        Args:
            query: User query

        Returns:
            List of retrieved chunks with metadata
        """
        results = self.qdrant_manager.search_by_text(
            query_text=query,
            collection_name=self.collection_name,
            top_k=self.top_k,
            score_threshold=self.score_threshold
        )

        if results:
            print(f"✓ Found {len(results)} relevant chunks")
        else:
            print("✗ No relevant context found")

        return results

    def format_context(self, results: List[Dict[str, Any]], max_length: int = 3000) -> str:
        """
        Format retrieved results into context string.
        Limits total context length to prevent timeouts.

        Args:
            results: List of search results
            max_length: Maximum total characters for context

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found in the documents."

        context_parts = []
        total_length = 0

        for i, result in enumerate(results, 1):
            source = result['metadata'].get('source', 'Unknown')
            score = result['score']
            text = result['text']

            # Truncate individual chunks if too long
            if len(text) > 800:
                text = text[:800] + "..."

            chunk = f"[Source {i}: {source} (relevance: {score:.2f})]\n{text}"

            # Check if adding this chunk would exceed max_length
            if total_length + len(chunk) > max_length:
                # Truncate the last chunk to fit
                remaining = max_length - total_length
                if remaining > 200:  # Only add if we have reasonable space
                    chunk = chunk[:remaining] + "..."
                    context_parts.append(chunk)
                break

            context_parts.append(chunk)
            total_length += len(chunk)

        return "\n\n".join(context_parts)

    def generate_answer(
            self,
            query: str,
            context: str,
            stream: bool = True
    ) -> str:
        """
        Generate answer using Gemini based on query and context.

        Args:
            query: User query
            context: Retrieved context
            stream: Whether to stream the response

        Returns:
            Generated answer
        """
        # Create prompt
        prompt = self._create_prompt(query, context)

        # Call Gemini API
        try:
            print(f"\n🤖 Generating answer with {self.gemini_model}...")

            if stream:
                # Handle streaming response
                response = self.client.models.generate_content_stream(
                    model=self.gemini_model,
                    contents=prompt
                )
                answer = ""
                print("Answer: ", end="", flush=True)
                for chunk in response:
                    if chunk.text:
                        answer += chunk.text
                        print(chunk.text, end="", flush=True)
                print()  # New line after streaming

                if not answer:
                    return "Error: Received empty response from Gemini"
                return answer
            else:
                # Handle non-streaming response
                response = self.client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt
                )
                answer = response.text

                if not answer:
                    return "Error: Received empty response from Gemini"

                return answer

        except Exception as e:
            error_msg = f"Error generating answer with Gemini: {e}"
            print(f"\n❌ {error_msg}")
            return error_msg

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for LLM.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful assistant that answers questions based on the provided context from PDF documents.

Context from documents:
{context}

User question: {query}

Instructions:
- Answer the question based ONLY on the information provided in the context above
- If the context doesn't contain enough information to answer the question, say "I don't have enough information in the documents to answer this question."
- Be concise and direct in your answer
- Cite which source you're using when relevant (e.g., "According to Source 1...")
- Do not make up information that isn't in the context

Answer:"""

        return prompt

    def classify_query(self, query: str) -> bool:
        """
        Classify if a query needs document retrieval or can be answered directly.

        Args:
            query: User query

        Returns:
            True if RAG is needed, False if LLM can answer directly
        """
        # Keywords that indicate domain-specific questions needing RAG
        rag_keywords = [
            # ML/AI topics
            'machine learning', 'ml', 'deep learning', 'neural network', 'training',
            'model', 'algorithm', 'dataset', 'feature', 'prediction', 'classification',
            'regression', 'clustering', 'supervised', 'unsupervised', 'reinforcement',
            'accuracy', 'loss', 'optimization', 'gradient', 'backpropagation',

            # LLM topics
            'llm', 'large language model', 'transformer', 'attention', 'gpt',
            'bert', 'embedding', 'token', 'prompt', 'fine-tuning', 'pre-training',

            # RAG topics
            'rag', 'retrieval', 'augmented', 'vector', 'similarity', 'semantic search',
            'knowledge base', 'document', 'context', 'retrieval augmented',

            # Research/paper topics
            'paper', 'research', 'study', 'experiment', 'results', 'methodology',
            'approach', 'technique', 'framework', 'architecture',

            # Technical topics
            'implementation', 'performance', 'evaluation', 'benchmark', 'metric',
            'hyperparameter', 'activation', 'layer', 'attention mechanism'
        ]

        # Convert query to lowercase for matching
        query_lower = query.lower()

        # Check if any RAG keyword is in the query
        for keyword in rag_keywords:
            if keyword in query_lower:
                return True

        # Questions about "the document" or "the paper" need RAG
        document_phrases = ['the document', 'the paper', 'the article', 'the study',
                            'according to', 'in the', 'this research', 'the authors']
        for phrase in document_phrases:
            if phrase in query_lower:
                return True

        # Default to False - use LLM directly for general questions
        return False

    def direct_llm_answer(self, query: str, stream: bool = True) -> str:
        """
        Answer a question directly with Gemini without document retrieval.

        Args:
            query: User query
            stream: Whether to stream the response

        Returns:
            LLM's answer
        """
        print(f"\n💡 Answering directly (no document search needed)")

        # Simple prompt for general questions
        prompt = f"""You are a helpful AI assistant. Answer the following question directly and concisely.

Question: {query}

Answer:"""

        # Call Gemini
        try:
            print(f"\n🤖 Generating answer with {self.gemini_model}...")

            if stream:
                response = self.client.models.generate_content_stream(
                    model=self.gemini_model,
                    contents=prompt
                )
                answer = ""
                print("Answer: ", end="", flush=True)
                for chunk in response:
                    if chunk.text:
                        answer += chunk.text
                        print(chunk.text, end="", flush=True)
                print()
                return answer if answer else "Error: Empty response"
            else:
                response = self.client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt
                )
                return response.text if response.text else "Error: Empty response"

        except Exception as e:
            error_msg = f"Error generating answer with Gemini: {e}"
            print(f"\n❌ {error_msg}")
            return error_msg

    def rag_based_answer(
            self,
            query: str,
            stream: bool = True,
            show_context: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG (Retrieval Augmented Generation).
        Searches documents and generates answer based on retrieved context.

        Args:
            query: User question
            stream: Whether to stream the LLM response
            show_context: Whether to print retrieved context

        Returns:
            Dictionary with answer and metadata
        """
        print(f"\n🔍 Searching documents (top_{self.top_k}, threshold={self.score_threshold})...")

        # Step 1: Retrieve relevant context
        results = self.retrieve_context(query)

        if not results:
            print("\n⚠️  No relevant documents found. Falling back to direct LLM answer...")
            answer = self.direct_llm_answer(query, stream=stream)
            return {
                "query": query,
                "answer": answer,
                "sources": [],
                "num_sources": 0,
                "used_rag": False
            }

        # Step 2: Format context
        context = self.format_context(results)

        if show_context:
            print(f"\n📚 Retrieved Context:")
            print("-" * 80)
            print(context)
            print("-" * 80)

        # Step 3: Generate answer using context
        answer = self.generate_answer(query, context, stream=stream)

        # Extract sources
        sources = [
            {
                "source": r['metadata'].get('source', 'Unknown'),
                "score": r['score'],
                "text_preview": r['text'][:200] + "..." if len(r['text']) > 200 else r['text']
            }
            for r in results
        ]

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "num_sources": len(results),
            "used_rag": True
        }

    def answer(
            self,
            query: str,
            stream: bool = True,
            show_context: bool = False,
            force_rag: bool = False
    ) -> Dict[str, Any]:
        """
        Main method to answer a question using RAG or direct LLM.
        Automatically classifies the query and chooses the best approach.

        Args:
            query: User question
            stream: Whether to stream the LLM response
            show_context: Whether to print retrieved context
            force_rag: Force use of RAG even for simple questions

        Returns:
            Dictionary with answer and metadata
        """
        print(f"\n{'=' * 80}")
        print(f"📝 Question: {query}")
        print(f"{'=' * 80}")

        # Classify if RAG is needed
        needs_rag = force_rag or self.classify_query(query)

        if not needs_rag:
            # Answer directly without RAG
            answer = self.direct_llm_answer(query, stream=stream)
            return {
                "query": query,
                "answer": answer,
                "sources": [],
                "num_sources": 0,
                "used_rag": False
            }

        # Use RAG for domain-specific questions
        print(f"\n🎯 Domain-specific question detected - using RAG...")
        return self.rag_based_answer(query, stream=stream, show_context=show_context)

    def interactive_chat(self):
        """
        Interactive chat mode - keep asking questions until user quits.
        """
        print("\n" + "=" * 80)
        print("🤖 RAG AGENT - INTERACTIVE CHAT MODE")
        print("=" * 80)
        print("Ask questions about your documents!")
        print("Commands:")
        print("  - Type your question to get an answer")
        print("  - 'context' - Toggle showing retrieved context")
        print("  - 'force-rag' - Toggle forcing RAG for all questions")
        print("  - 'quit' or 'exit' - Exit chat")
        print("=" * 80)
        print("\n💡 Smart mode: I'll automatically decide when to search documents")
        print("   vs. when to answer directly based on your question.")

        show_context = False
        force_rag = False

        while True:
            try:
                user_input = input("\n💬 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == 'context':
                    show_context = not show_context
                    status = "ON" if show_context else "OFF"
                    print(f"✓ Context display is now {status}")
                    continue

                if user_input.lower() == 'force-rag':
                    force_rag = not force_rag
                    status = "ON" if force_rag else "OFF"
                    print(f"✓ Force RAG mode is now {status}")
                    if force_rag:
                        print("  All questions will search documents")
                    else:
                        print("  Smart classification enabled")
                    continue

                # Answer the question
                result = self.answer(
                    query=user_input,
                    stream=True,
                    show_context=show_context,
                    force_rag=force_rag
                )

                # Show sources if RAG was used
                if result.get('used_rag', False) and result['sources']:
                    print(f"\n📖 Sources used ({result['num_sources']}):")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"  {i}. {source['source']} (score: {source['score']:.2f})")
                elif not result.get('used_rag', False):
                    print("\n💡 (Answered directly without searching documents)")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    """Main function to run the RAG agent."""

    # Configuration
    COLLECTION_NAME = "pdf_documents"
    GEMINI_MODEL = "gemini-2.5-flash"  # Stable model that works
    QDRANT_URL = "http://localhost:6333"
    OLLAMA_URL = "http://localhost:11434"  # Still needed for embeddings

    # Get Gemini API key
    # Option 1: Set your API key directly here
    GEMINI_API_KEY = "AIzaSyBajNRSha0jpsooel1wzb0RxRzFfFIJMYY"  # Replace with your actual key

    # Option 2: Or use environment variable (more secure)
    # GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

    if not GEMINI_API_KEY:
        print("\n⚠️  GEMINI_API_KEY not set!")
        print("\nPlease add your API key in the code or set environment variable")
        print("Get an API key from: https://makersuite.google.com/app/apikey")
        return

    # Initialize agent
    try:
        agent = RAGAgent(
            collection_name=COLLECTION_NAME,
            gemini_api_key=GEMINI_API_KEY,
            gemini_model=GEMINI_MODEL,
            qdrant_url=QDRANT_URL,
            ollama_url=OLLAMA_URL,
            top_k=3,
            score_threshold=0.3
        )
    except Exception as e:
        print(f"\n❌ Error initializing RAG agent: {e}")
        return

    # Check if collection has documents
    try:
        collection_info = agent.qdrant_manager.get_collection_info(COLLECTION_NAME)
        if collection_info:
            count = collection_info.points_count
            print(f"\n✓ Collection '{COLLECTION_NAME}' has {count} documents")

            if count == 0:
                print("\n⚠️  Warning: The collection is empty!")
                print("Please run qdrant_populate.py first to add documents.")
                print()
                response = input("Continue anyway? (yes/no): ").strip().lower()
                if response != 'yes':
                    return
        else:
            print(f"\n⚠️  Collection '{COLLECTION_NAME}' does not exist!")
            print("Please run qdrant_populate.py first to create and populate the collection.")
            return
    except Exception as e:
        print(f"\n⚠️  Error checking collection: {e}")
        print("Please make sure Qdrant is running and the collection exists.")
        return

    # Menu
    print("\n" + "=" * 80)
    print("RAG AGENT - QUESTION ANSWERING SYSTEM")
    print("=" * 80)
    print("\nOptions:")
    print("1. Interactive chat mode")
    print("2. Test with sample questions")
    print("3. Single question")
    print("4. Exit")

    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()

            if choice == "1":
                agent.interactive_chat()

            elif choice == "2":
                # Sample questions
                sample_questions = [
                    "What is the main topic of the documents?",
                    "Can you summarize the key points?",
                    "What are the important findings?"
                ]

                print("\n" + "=" * 80)
                print("TESTING WITH SAMPLE QUESTIONS")
                print("=" * 80)

                for question in sample_questions:
                    result = agent.answer(question, stream=False, show_context=False)
                    print(f"\nAnswer: {result['answer']}")
                    print("-" * 80)

            elif choice == "3":
                question = input("\nEnter your question: ").strip()
                if question:
                    agent.answer(question, stream=True, show_context=True)

            elif choice == "4":
                print("\nExiting...")
                break

            else:
                print("Invalid choice. Please enter 1-4")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()