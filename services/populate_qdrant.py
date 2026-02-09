from pathlib import Path
from typing import List
import time
from embedding_manager import Embedder
from qdrant_manager import QdrantManager
from preprocessing import pdf2chunks


class QdrantPopulator:
    def __init__(
            self,
            data_folder: str = "./data",
            collection_name: str = "pdf_documents",
            chunk_size: int = 1000,
            chunk_overlap: int = 200
    ):
        """
        Initialize the Qdrant populator.

        Args:
            data_folder: Path to folder containing PDF files
            collection_name: Name of Qdrant collection
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.data_folder = Path(data_folder)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize managers
        self.embedder = Embedder(model_name="nomic-embed-text")
        self.qdrant_manager = QdrantManager()

        print(f"Initialized with data folder: {self.data_folder}")
        print(f"Collection name: {self.collection_name}")

    def get_pdf_files(self) -> List[Path]:
        """
        Get list of all PDF files in the data folder.

        Returns:
            List of Path objects for PDF files
        """
        if not self.data_folder.exists():
            print(f"Creating data folder: {self.data_folder}")
            self.data_folder.mkdir(parents=True, exist_ok=True)
            return []

        pdf_files = list(self.data_folder.glob("*.pdf"))
        print(f"\nFound {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file.name}")

        return pdf_files

    def process_single_pdf(self, pdf_path: Path, verbose: bool = True) -> int:
        """
        Process a single PDF file: chunk, embed, and insert into Qdrant.

        Args:
            pdf_path: Path to the PDF file
            verbose: Whether to print chunk contents

        Returns:
            Number of chunks processed
        """
        print(f"\n{'=' * 80}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'=' * 80}")

        # Step 1: Extract chunks from PDF
        print(f"\n[1/3] Extracting chunks from PDF...")
        try:
            chunks = pdf2chunks(
                pdf_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            print(f"Extracted {len(chunks)} chunks")
        except Exception as e:
            print(f"Error extracting chunks from {pdf_path.name}: {e}")
            return 0

        if not chunks:
            print(f"No content extracted from {pdf_path.name}")
            return 0

        # Print chunks if verbose
        if verbose:
            print(f"\n--- Chunks from {pdf_path.name} ---")
            for i, chunk in enumerate(chunks, 1):
                print(f"\nChunk {i}/{len(chunks)} ({len(chunk)} chars):")
                print("-" * 80)
                # Print first 200 characters of each chunk
                preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                print(preview)

        # Step 2: Generate embeddings
        print(f"\n[2/3] Generating embeddings for {len(chunks)} chunks...")
        start_time = time.time()
        try:
            embeddings = self.embedder.embed_texts(chunks, show_progress=True)
            elapsed = time.time() - start_time
            print(f"Embedding completed in {elapsed:.2f} seconds")
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return 0

        # Step 3: Insert into Qdrant
        print(f"\n[3/3] Inserting into Qdrant collection '{self.collection_name}'...")
        try:
            # Prepare metadata for each chunk
            metadata_list = [
                {
                    "source": pdf_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk)
                }
                for i, chunk in enumerate(chunks)
            ]

            self.qdrant_manager.insert_points_batch(
                embeddings=embeddings,
                collection_name=self.collection_name,
                chunk_texts=chunks,
                metadata_list=metadata_list
            )
            print(f"Successfully inserted {len(chunks)} chunks from {pdf_path.name}")
        except Exception as e:
            print(f"Error inserting into Qdrant: {e}")
            return 0

        return len(chunks)

    def populate(self, verbose: bool = True, recreate_collection: bool = False):
        """
        Main method to populate Qdrant with all PDFs in data folder.

        Args:
            verbose: Whether to print chunk contents
            recreate_collection: Whether to delete and recreate the collection
        """
        print("\n" + "=" * 80)
        print("STARTING QDRANT POPULATION")
        print("=" * 80)

        # Get all PDF files
        pdf_files = self.get_pdf_files()

        if not pdf_files:
            print("\nNo PDF files found in data folder!")
            print(f"Please add PDF files to: {self.data_folder.absolute()}")
            return

        # Setup collection
        if recreate_collection:
            print(f"\nDeleting existing collection '{self.collection_name}'...")
            self.qdrant_manager.delete_collection(self.collection_name)

        # Check if collection exists, create if not
        existing_collections = self.qdrant_manager.list_collections()
        if self.collection_name not in existing_collections:
            print(f"\nCreating collection '{self.collection_name}'...")
            embedding_dim = self.embedder.get_embedding_dimension()
            self.qdrant_manager.create_collection(
                name=self.collection_name,
                vector_size=embedding_dim
            )
        else:
            print(f"\nUsing existing collection '{self.collection_name}'")

        # Process each PDF
        total_chunks = 0
        successful_files = 0
        start_time = time.time()

        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[File {i}/{len(pdf_files)}]")
            chunks_processed = self.process_single_pdf(pdf_path, verbose=verbose)

            if chunks_processed > 0:
                total_chunks += chunks_processed
                successful_files += 1

        # Summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print("POPULATION COMPLETE")
        print("=" * 80)
        print(f"Total files processed: {successful_files}/{len(pdf_files)}")
        print(f"Total chunks inserted: {total_chunks}")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Average time per file: {elapsed / len(pdf_files):.2f} seconds")

        # Collection info
        collection_info = self.qdrant_manager.get_collection_info(self.collection_name)
        if collection_info:
            print(f"\nCollection '{self.collection_name}' now contains:")
            print(f"  - Points count: {collection_info.points_count}")
            print(f"  - Vector size: {collection_info.config.params.vectors.size}")

    def test_search(self, query: str, top_k: int = 5, score_threshold: float = 0.5):
        """
        Test search functionality with a query.

        Args:
            query: Search query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score
        """
        print("\n" + "=" * 80)
        print("TESTING SEARCH")
        print("=" * 80)
        print(f"Query: '{query}'")
        print(f"Top K: {top_k}")
        print(f"Score threshold: {score_threshold}")

        # Generate query embedding
        print("\nGenerating query embedding...")
        query_embedding = self.embedder.embed_text(query)

        # Search
        print(f"Searching in collection '{self.collection_name}'...")
        results = self.qdrant_manager.search_by_text(
            query_text=query,
            collection_name=self.collection_name,
            top_k=top_k,
            score_threshold=score_threshold
        )

        # Display results
        if not results:
            print("\nNo results found!")
            return

        print(f"\nFound {len(results)} results:")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n[Result {i}]")
            print(f"Score: {result['score']:.4f}")
            print(f"Source: {result['metadata'].get('source', 'Unknown')}")
            print(
                f"Chunk: {result['metadata'].get('chunk_index', '?') + 1}/{result['metadata'].get('total_chunks', '?')}")
            print(f"\nText preview:")
            print("-" * 80)
            # Show first 300 characters
            text_preview = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
            print(text_preview)
            print("-" * 80)

    def interactive_search(self):
        """
        Interactive search mode - keep asking for queries until user quits.
        """
        print("\n" + "=" * 80)
        print("INTERACTIVE SEARCH MODE")
        print("=" * 80)
        print("Enter your queries (or 'quit' to exit)")
        print()

        while True:
            try:
                query = input("\nEnter query: ").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    print("Exiting interactive search...")
                    break

                if not query:
                    print("Please enter a valid query")
                    continue

                self.test_search(query, top_k=3, score_threshold=0.3)

            except KeyboardInterrupt:
                print("\n\nExiting interactive search...")
                break
            except Exception as e:
                print(f"Error during search: {e}")


def main():
    """Main function to run the population and testing."""

    # Configuration
    DATA_FOLDER = "../data"
    COLLECTION_NAME = "pdf_documents"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Initialize populator
    populator = QdrantPopulator(
        data_folder=DATA_FOLDER,
        collection_name=COLLECTION_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # Menu
    print("\n" + "=" * 80)
    print("QDRANT PDF POPULATOR")
    print("=" * 80)
    print("\nOptions:")
    print("1. Populate Qdrant with PDFs (keep existing data)")
    print("2. Populate Qdrant with PDFs (recreate collection)")
    print("3. Test search with sample queries")
    print("4. Interactive search mode")
    print("5. Show collection info")
    print("6. Exit")

    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()

            if choice == "1":
                populator.populate(verbose=False, recreate_collection=False)

            elif choice == "2":
                confirm = input("This will delete all existing data. Continue? (yes/no): ").strip().lower()
                if confirm == "yes":
                    populator.populate(verbose=False, recreate_collection=True)
                else:
                    print("Operation cancelled")

            elif choice == "3":
                # Test with sample queries
                sample_queries = [
                    "What is machine learning?",
                    "Explain neural networks",
                    "How does artificial intelligence work?"
                ]

                for query in sample_queries:
                    populator.test_search(query, top_k=3, score_threshold=0.3)
                    print("\n" + "-" * 80 + "\n")

            elif choice == "4":
                populator.interactive_search()

            elif choice == "5":
                collection_info = populator.qdrant_manager.get_collection_info(COLLECTION_NAME)
                if collection_info:
                    print(f"\nCollection: {COLLECTION_NAME}")
                    print(f"Points count: {collection_info.points_count}")
                    print(f"Vector size: {collection_info.config.params.vectors.size}")
                    print(f"Distance metric: {collection_info.config.params.vectors.distance}")
                else:
                    print(f"\nCollection '{COLLECTION_NAME}' does not exist")

            elif choice == "6":
                print("\nExiting...")
                break

            else:
                print("Invalid choice. Please enter 1-6")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()