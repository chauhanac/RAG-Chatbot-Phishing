# ================================
# Imports
# ================================
import glob

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings


# ================================
# PDF Ingestor (Directory Support)
# ================================
class DataIngestor:
    """Loads raw files from a directory and splits into chunks"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        
        self.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
        )

    def load_documents_from_dir(self, directory: str):
        """Load all PDF files from a directory into LangChain Document objects"""

        file_paths = glob.glob(f"{directory}/*.pdf")

        if not file_paths:
            raise ValueError(f"No PDF files found in directory: {directory}")

        dir_loader=DirectoryLoader(
        directory, glob="**/*.pdf", ## Pattern to match files  
        loader_cls= PyMuPDFLoader, ##loader class to use
        show_progress=True
        )
        
        pdf_documents=dir_loader.load()
        print(f"Loaded {len(pdf_documents)} documents from {len(file_paths)} files in {directory}")
        return pdf_documents

    def ingest(self, directory: str):
        """Load & split all documents from a directory"""
        raw_docs = self.load_documents_from_dir(directory)
        split_docs = self.text_splitter.split_documents(raw_docs)
        print(f"Split into {len(split_docs)} chunks")

        return split_docs

# ================================
# Embedding Manager
# ================================
class EmbeddingManager:
    """Manages embedding models (HuggingFace, Ollama, etc.)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", provider: str = "huggingface"):
        self.model_name = model_name
        self.provider = provider
        self.embedding_function = self._load_embedding()

    def _load_embedding(self):
        if self.provider == "huggingface":
            return HuggingFaceEmbeddings(model_name=self.model_name)
        elif self.provider == "ollama":
            return OllamaEmbeddings(model=self.model_name)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def get_embedding_function(self):
        return self.embedding_function

# # ================================
# # Vector Store Wrapper (ChromaDB)
# # ================================
class VectorStore:
    """Manages ChromaDB vector store with embeddings."""

    def __init__(self, embedding_manager: EmbeddingManager, persist_directory: str = "./chroma_store"):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_manager.get_embedding_function()
        self.store = None

    def create(self, documents):
        print("Creating vector store...")
        self.store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory,
        )
        print("Vector store created")

    def get_retriever(self, k: int = 5):
        if not self.store:
            raise ValueError("Vector store not initialized. Call create() first.")
            
        return self.store.as_retriever(search_type="mmr", search_kwargs={"k": k})

# ================================
# Conversational RAG Chatbot
# ================================
class ConversationalRAGChatbot:
    """Context + history-aware chatbot using RAG + Ollama."""

    def __init__(self, vector_store: VectorStore, model_name: str = "llama3"):
        self.vector_store = vector_store
        self.llm = ChatOllama(model=model_name, temperature=0)

        # Retriever from vector store
        self.retriever = self.vector_store.get_retriever(k=5)

        # Use ChatMessageHistory for memory
        self.history = ChatMessageHistory()

        self.histories = {}

        # Conversational QA Chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=False,
            verbose=True,
        )

        # Runnable with history wrapper
        self.runnable = RunnableWithMessageHistory(
            self.qa_chain,
            lambda session_id: self.histories.setdefault(session_id, ChatMessageHistory()),
            input_messages_key="question",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def ask(self, query: str, session_id: str = "default"):
        """Ask a question and return the chatbot’s response."""
        response = self.runnable.invoke(
            {"question": query},
            config={"configurable": {"session_id": session_id}},
        )
        return response["answer"]

    def get_history(self, session_id: str = "default"):
        """Return formatted conversation history for display in UI."""
        if session_id not in self.histories:
            return []

        history = self.histories[session_id].messages
        formatted_history = []

        for msg in history:
            if msg.type == "human":
                formatted_history.append(f"🧑 **User:** {msg.content}")
            elif msg.type == "ai":
                formatted_history.append(f"🤖 **Bot:** {msg.content}")
            else:
                formatted_history.append(f"ℹ️ {msg.content}")

        return formatted_history


def get_chatbot():
    """
    Convenience function to build and return a ConversationalRAGChatbot.
    """
    pdf_directory = "Papers"
    embedding_manager = EmbeddingManager(provider = "huggingface") # options for embedding model ["huggingface", "ollama"]
    vector_store = VectorStore(embedding_manager)
    ingestor = DataIngestor()

    # Step 1: Ingest data
    docs = ingestor.ingest(pdf_directory)

    # Step 2: Create Vector Store
    vector_store.create(docs)

    # # Step 3: Create chatbot
    chatbot = ConversationalRAGChatbot(vector_store)

    return chatbot