import os, pickle, pdfplumber, re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import time

start = time.time()

# Paths to saved files
TEXT_CHUNKS_PATH = "text_chunks.pkl"
EMBEDDINGS_PATH = "chunk_embeddings.pkl"
FAISS_INDEX_PATH = "faiss_index.index"

# Load a better embedding model
embedding_model = SentenceTransformer("all-mpnet-base-v2")

""" Check if preprocessed data exists """
if (
    os.path.exists(TEXT_CHUNKS_PATH)
    and os.path.exists(EMBEDDINGS_PATH)
    and os.path.exists(FAISS_INDEX_PATH)
):
    print("Loading preprocessed data...")
    with open(TEXT_CHUNKS_PATH, "rb") as f:
        text_chunks = pickle.load(f)
    with open(EMBEDDINGS_PATH, "rb") as f:
        chunk_embeddings = pickle.load(f)
    index = faiss.read_index(FAISS_INDEX_PATH)
else:

    """Extract PDF"""
    print("Extracting text from PDF...")

    def extract_text_from_pdf(pdf_path):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text

    pdf_text = extract_text_from_pdf("input/AI.pdf")

    """ Clean text """
    print("Cleaning text...")

    def clean_text(text):
        # Remove extra spaces and line breaks
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    pdf_text = clean_text(pdf_text)

    """ Split text into chunks """
    print("Splitting text into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Larger chunks for more context
        chunk_overlap=200,  # More overlap for better context continuity
    )

    text_chunks = text_splitter.split_text(pdf_text)

    """ Create Embeddings """
    print("Creating embeddings...")

    # Load a pre-trained embedding model
    # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for text chunks
    chunk_embeddings = embedding_model.encode(text_chunks)

    """ Store embeddings in vector database """
    print("Storing embeddings in vector database...")

    # Convert embeddings to a FAISS-compatible format
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings.astype(np.float32))

    """ Save preprocessed data """
    print("Saving preprocessed data...")
    with open(TEXT_CHUNKS_PATH, "wb") as f:
        pickle.dump(text_chunks, f)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(chunk_embeddings, f)
    faiss.write_index(index, FAISS_INDEX_PATH)


""" QA System """
print("Initializing QA system...")
# Load a better pre-trained QA model
qa_pipeline = pipeline(
    "question-answering",
    model="openlm-research/open_llama_7b",  # Publicly available model
    tokenizer="CodeLlamaTokenizer",
)


def answer_question(question):
    # Step 1: Find the most relevant text chunk
    question_embedding = embedding_model.encode([question])
    _, indices = index.search(question_embedding.astype(np.float32), k=1)
    relevant_chunk = text_chunks[indices[0][0]]

    # Step 2: Use the QA model to get an answer
    result = qa_pipeline(question=question, context=relevant_chunk)
    print(result)
    return result["answer"]


# QA Object
class QA:
    def __init__(self, question):
        self.question = question
        self.answer = answer_question(question).strip().replace("\n", " ")

    def __str__(self):
        return f"Question: {self.question}\nAnswer: {self.answer}"


# Example usage:
qa1 = QA("What is the main topic of the document?")
# qa2 = QA("What is the goal of AI?")
# qa3 = QA("What is the Turing test?")
# qa4 = QA("What is the difference between weak and strong AI?")
# qa5 = QA("What are some ethical concerns related to AI?")
# qa6 = QA("What are some applications of AI?")

print(qa1)
# print(qa2)
# print(qa3)
# print(qa4)
# print(qa5)
# print(qa6)

end = time.time()
print("Time taken: ", end - start)
