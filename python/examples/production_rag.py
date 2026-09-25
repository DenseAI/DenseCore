"""
Production-Ready RAG (Retrieval-Augmented Generation) Example

This example demonstrates a complete, production-ready RAG system using DenseCore with:
- Document ingestion and processing
- Vector store integration
- Semantic search
- Context-aware generation
- Error handling and fallbacks
- Performance monitoring

Requirements:
    pip install densecore[langchain]
    pip install faiss-cpu chromadb  # Or your preferred vector store
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# DenseCore imports
from densecore.integrations import DenseCoreLLM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG system"""

    model_path: Optional[str] = None
    hf_repo_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_docs: int = 3
    temperature: float = 0.3
    max_tokens: int = 512


class ProductionRAGSystem:
    """
    Production-ready RAG system with DenseCore.

    Features:
    - Robust document processing
    - Semantic search with vector database
    - Context-aware generation
    - Error handling and retries
    - Performance monitoring
    """

    def __init__(self, config: RAGConfig):
        """Initialize RAG system"""
        self.config = config

        logger.info("Initializing RAG system...")

        # Initialize LLM
        logger.info(f"Loading LLM: {config.hf_repo_id}")
        self.llm = DenseCoreLLM(
            model_path=config.model_path,
            hf_repo_id=config.hf_repo_id,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # Initialize embeddings
        logger.info(f"Loading embeddings: {config.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)

        # Initialize vector store (will be populated later)
        self.vectorstore = None
        self.retriever = None

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info("✓ RAG system initialized")

    def ingest_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Ingest and process documents into vector store.

        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
        """
        logger.info(f"Ingesting {len(documents)} documents...")
        start_time = time.time()

        # Create document objects
        docs = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(documents, metadatas or [{}] * len(documents))
        ]

        # Split into chunks
        logger.info("Splitting documents into chunks...")
        splits = self.text_splitter.split_documents(docs)
        logger.info(f"Created {len(splits)} chunks")

        # Create vector store
        logger.info("Creating vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=splits, embedding=self.embeddings, collection_name="densecore_rag"
        )

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config.top_k_docs})

        elapsed = time.time() - start_time
        logger.info(f"✓ Ingestion complete in {elapsed:.2f}s")

    def create_qa_chain(self, custom_prompt: Optional[str] = None):
        """
        Create a QA chain with retrieval.

        Args:
            custom_prompt: Optional custom prompt template

        Returns:
            Retrieval runnable
        """
        if self.retriever is None:
            raise ValueError("No documents ingested. Call ingest_documents() first.")

        # Default prompt template
        if custom_prompt is None:
            template = """Use the following context to answer the question at the end.
If you don't know the answer based on the context, just say you don't know, don't make up an answer.

Context:
{context}

Question: {question}

Answer:"""
        else:
            template = custom_prompt

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        qa_chain = create_retrieval_chain(self.retriever, combine_docs_chain)

        return qa_chain

    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system.

        Args:
            question: User question
            return_sources: Whether to return source documents

        Returns:
            Dictionary with answer and optional sources
        """
        logger.info(f"Processing query: {question}")
        start_time = time.time()

        try:
            # Create QA chain
            qa_chain = self.create_qa_chain()

            # Get answer
            result = qa_chain.invoke({"input": question})

            elapsed = time.time() - start_time
            logger.info(f"✓ Query processed in {elapsed:.2f}s")

            # Format response
            response = {
                "answer": result["answer"],
                "query_time": elapsed,
            }

            if return_sources:
                response["sources"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in result.get("context", [])
                ]

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "query_time": time.time() - start_time,
                "error": str(e),
            }

    def search_documents(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Search for relevant documents without generation.

        Args:
            query: Search query
            top_k: Number of documents to return (default: config.top_k_docs)

        Returns:
            List of relevant documents
        """
        if self.vectorstore is None:
            raise ValueError("No documents ingested.")

        k = top_k or self.config.top_k_docs
        docs = self.vectorstore.similarity_search(query, k=k)

        return docs


def example_basic_rag():
    """Example: Basic RAG usage"""
    print("\n" + "=" * 80)
    print("Example: Basic RAG System")
    print("=" * 80)

    # Sample documents about Python
    documents = [
        """Python is a high-level, interpreted programming language created by Guido van Rossum
        and first released in 1991. It emphasizes code readability with significant use of
        whitespace and a clear, expressive syntax.""",
        """Python supports multiple programming paradigms including procedural, object-oriented,
        and functional programming. It features dynamic typing, automatic memory management,
        and a comprehensive standard library.""",
        """Popular Python frameworks include Django and Flask for web development, NumPy and
        Pandas for data science, TensorFlow and PyTorch for machine learning, and pytest for testing.""",
        """Python's design philosophy emphasizes code readability and simplicity. The Zen of Python
        includes aphorisms such as 'Explicit is better than implicit' and 'Simple is better than complex'.""",
    ]

    # Initialize RAG system
    config = RAGConfig(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.3,
        max_tokens=200,
    )

    rag = ProductionRAGSystem(config)

    # Ingest documents
    rag.ingest_documents(documents)

    # Query the system
    questions = [
        "Who created Python?",
        "What programming paradigms does Python support?",
        "What is the Zen of Python?",
    ]

    for question in questions:
        print(f"\n\nQ: {question}")
        result = rag.query(question, return_sources=True)
        print(f"A: {result['answer']}")
        print(f"   (Query time: {result['query_time']:.2f}s)")

        if "sources" in result and result["sources"]:
            print(f"\n   Sources used: {len(result['sources'])} documents")


def example_advanced_rag():
    """Example: Advanced RAG with custom prompt and error handling"""
    print("\n" + "=" * 80)
    print("Example: Advanced RAG with Custom Prompt")
    print("=" * 80)

    # Technical documentation
    documents = [
        "DenseCore is a high-performance CPU inference engine for LLMs with HuggingFace integration.",
        "DenseCore supports GGUF format models and provides both synchronous and asynchronous APIs.",
        "Key features include streaming generation, quantization, and automatic model caching.",
        "DenseCore integrates with LangChain and LangGraph for building complex LLM applications.",
    ]

    config = RAGConfig(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.2,
        max_tokens=150,
        top_k_docs=2,
    )

    rag = ProductionRAGSystem(config)
    rag.ingest_documents(documents)

    # Custom prompt for technical Q&A
    custom_prompt = """You are a technical documentation assistant.
Use the following context to answer the question precisely and concisely.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Technical Answer:"""

    # Create custom QA chain
    qa_chain = rag.create_qa_chain(custom_prompt=custom_prompt)

    # Test queries
    result = qa_chain({"query": "What formats does DenseCore support?"})
    print("\nQ: What formats does DenseCore support?")
    print(f"A: {result['result']}")


def main():
    """Run all examples"""
    print("\n📚 DenseCore Production RAG Examples")
    print("=" * 80)

    example_basic_rag()
    example_advanced_rag()

    print("\n" + "=" * 80)
    print("✅ All RAG examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
