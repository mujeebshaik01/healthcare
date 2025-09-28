import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Required libraries - install with pip
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.llms import Ollama  # or use OpenAI, HuggingFace, etc.
    from langchain.chains import RetrievalQA
    from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
    from langchain.prompts import PromptTemplate
    import PyPDF2
except ImportError:
    print("Please install required packages: pip install langchain chromadb pypdf2 sentence-transformers")

class HealthcareRAGSystem:
    def __init__(self, model_name="llama2", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize Healthcare RAG System
        
        Args:
            model_name: LLM model name for Ollama
            embedding_model: Embedding model for document vectors
        """
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None
        self.qa_chain = None
        
        # Initialize LLM (using Ollama - you can replace with OpenAI, etc.)
        try:
            self.llm = Ollama(model=model_name)
        except:
            print("Ollama not available. Please set up LLM connection.")
            self.llm = None
        
        # Healthcare-specific prompt template
        self.prompt_template = """You are a helpful and cautious healthcare assistant. 
        Use the following medical context to answer the question. If you're unsure or the information 
        is not in the context, say so. Always recommend consulting healthcare professionals.

        Context: {context}

        Question: {question}

        Answer: """
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
    def load_documents(self, file_paths: List[str]):
        """
        Load healthcare documents from various sources
        
        Args:
            file_paths: List of paths to documents (PDF, TXT, etc.)
        """
        documents = []
        
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Loaded PDF: {file_path}")
                except Exception as e:
                    print(f"Error loading PDF {file_path}: {e}")
            
            elif file_path.endswith('.txt'):
                try:
                    loader = TextLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Loaded TXT: {file_path}")
                except Exception as e:
                    print(f"Error loading TXT {file_path}: {e}")
        
        return documents
    
    def process_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """
        Split documents into chunks for processing
        
        Args:
            documents: List of loaded documents
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, documents, persist_directory="./chroma_db"):
        """
        Create vector store from document chunks
        
        Args:
            documents: List of document chunks
            persist_directory: Directory to store vectors
        """
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=persist_directory
        )
        print("Vector store created successfully")
    
    def load_vector_store(self, persist_directory="./chroma_db"):
        """
        Load existing vector store
        
        Args:
            persist_directory: Directory containing stored vectors
        """
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_model
        )
        print("Vector store loaded successfully")
    
    def setup_qa_chain(self, k=3):
        """
        Set up the QA retrieval chain
        
        Args:
            k: Number of documents to retrieve
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Load or create one first.")
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
        print("QA chain setup completed")
    
    def ask_question(self, question: str, medical_context: str = ""):
        """
        Ask a healthcare-related question to the RAG system
        
        Args:
            question: The question to ask
            medical_context: Additional medical context (optional)
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not setup. Call setup_qa_chain() first.")
        
        # Add medical context to question if provided
        if medical_context:
            full_question = f"Context: {medical_context}\nQuestion: {question}"
        else:
            full_question = question
        
        try:
            result = self.qa_chain({"query": full_question})
            
            # Add healthcare disclaimer
            disclaimer = "\n\n⚠️ **Important**: This is an AI assistant for informational purposes only. Always consult qualified healthcare professionals for medical advice."
            
            return {
                "answer": result["result"] + disclaimer,
                "source_documents": result["source_documents"],
                "confidence": self._calculate_confidence(result)
            }
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "source_documents": [],
                "confidence": 0.0
            }
    
    def _calculate_confidence(self, result):
        """
        Calculate confidence score for the answer (simplified version)
        """
        # This is a simplified confidence calculation
        # In practice, you might want more sophisticated methods
        answer_length = len(result["result"])
        source_count = len(result["source_documents"])
        
        # Basic confidence based on answer length and source count
        confidence = min(1.0, (answer_length / 500) * 0.3 + (source_count / 5) * 0.7)
        return round(confidence, 2)

# Example usage and demonstration
def create_sample_healthcare_data():
    """
    Create sample healthcare documents for testing
    """
    sample_data = {
        "common_medications.txt": """
        Common Medications and Their Uses:

        1. Aspirin - Used for pain relief, fever reduction, and as an anti-inflammatory.
        Dosage: Typically 325-650mg every 4-6 hours as needed.
        Precautions: Avoid in children, can cause stomach irritation.

        2. Metformin - Used for type 2 diabetes management.
        Dosage: Usually 500-1000mg twice daily with meals.
        Precautions: Monitor kidney function, can cause gastrointestinal issues.

        3. Lisinopril - Used for high blood pressure and heart failure.
        Dosage: 10-40mg once daily.
        Precautions: Monitor for cough, angioedema, and kidney function.

        4. Atorvastatin - Used to lower cholesterol levels.
        Dosage: 10-80mg once daily.
        Precautions: Monitor liver enzymes, can cause muscle pain.
        """,
        
        "symptoms_guide.txt": """
        Common Symptoms and When to Seek Help:

        Chest Pain: 
        - If severe, radiating to arm/jaw, with sweating - seek emergency care
        - If mild and brief, monitor and consult doctor

        Fever:
        - Adults: Seek help if above 103°F or lasting more than 3 days
        - Children: Seek help if above 100.4°F in infants under 3 months

        Headache:
        - Seek emergency care if sudden, severe, or with neurological symptoms
        - Common headaches can be managed with OTC pain relievers
        """,
        
        "first_aid_basics.txt": """
        Basic First Aid Procedures:

        CPR for Adults:
        - Check responsiveness
        - Call emergency services
        - 30 chest compressions followed by 2 rescue breaths
        - Continue until help arrives

        Choking:
        - For conscious adults: Perform Heimlich maneuver
        - For unconscious: Begin CPR

        Bleeding Control:
        - Apply direct pressure with clean cloth
        - Elevate injured area if possible
        - Seek medical help for severe bleeding
        """
    }
    
    # Create sample files
    for filename, content in sample_data.items():
        with open(filename, 'w') as f:
            f.write(content)
        print(f"Created sample file: {filename}")
    
    return list(sample_data.keys())

def main():
    """
    Main demonstration function
    """
    print("=== Healthcare RAG System Demo ===\n")
    
    # Create sample healthcare data
    print("Creating sample healthcare documents...")
    sample_files = create_sample_healthcare_data()
    
    # Initialize RAG system
    print("\nInitializing Healthcare RAG System...")
    healthcare_rag = HealthcareRAGSystem()
    
    # Load and process documents
    print("Loading and processing documents...")
    documents = healthcare_rag.load_documents(sample_files)
    chunks = healthcare_rag.process_documents(documents)
    
    # Create vector store
    print("Creating vector store...")
    healthcare_rag.create_vector_store(chunks)
    
    # Setup QA chain
    print("Setting up QA chain...")
    healthcare_rag.setup_qa_chain(k=2)
    
    # Sample healthcare questions
    questions = [
        "What is the typical dosage for Metformin?",
        "When should I seek help for chest pain?",
        "How do I perform CPR on an adult?",
        "What are the precautions for taking aspirin?",
        "What should I do if someone is choking?"
    ]
    
    print("\n" + "="*50)
    print("Testing Healthcare RAG System:")
    print("="*50)
    
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        print("-" * 40)
        
        result = healthcare_rag.ask_question(question)
        
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Sources: {len(result['source_documents'])} documents referenced")
        
        # Show source documents
        for j, doc in enumerate(result['source_documents'], 1):
            source_content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"  Source {j}: {source_content}")
    
    # Interactive mode
    print("\n" + "="*50)
    print("Interactive Mode - Ask your own healthcare questions")
    print("Type 'quit' to exit")
    print("="*50)
    
    while True:
        user_question = input("\nYour healthcare question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_question:
            result = healthcare_rag.ask_question(user_question)
            print(f"\nAnswer: {result['answer']}")
            print(f"Confidence: {result['confidence']}")

# Additional utility functions
class HealthcareDataProcessor:
    """Utility class for processing healthcare-specific data"""
    
    @staticmethod
    def validate_medical_question(question: str) -> Dict[str, Any]:
        """
        Basic validation for medical questions
        """
        warnings = []
        
        # Check for emergency keywords
        emergency_keywords = ['emergency', '911', 'dying', 'heart attack', 'stroke', 'severe pain']
        if any(keyword in question.lower() for keyword in emergency_keywords):
            warnings.append("This sounds like an emergency. Please contact emergency services immediately.")
        
        # Check for medication dosage questions
        if 'dosage' in question.lower() and 'doctor' not in question.lower():
            warnings.append("Medication dosages should always be confirmed by a healthcare professional.")
        
        return {
            "is_valid": len(warnings) == 0,
            "warnings": warnings,
            "requires_professional": any(warning for warning in warnings)
        }

if __name__ == "__main__":
    main()

