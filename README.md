# **QUESTION YOUR PDF**



## **PDF-based Question Answering System**
This repository contains a PDF-based Question Answering System that uses LangChain, Pinecone, Word2Vec, and Meta-Llama-3-8B-Instruct to provide accurate answers to questions based on the content of PDF documents.

## **Introduction**
The PDF-based Question Answering System is designed to take PDF documents as input, extract their content, and answer questions based on the extracted text. **The system leverages:**
LangChain for chaining various components.
Pinecone as the vector database for efficient similarity search.
Word2Vec for generating embeddings.
Meta-Llama-3-8B-Instruct for generating answers based on the most relevant text extracted from the Pinecone database.


## **Features**
Text Extraction: Extracts text from PDF documents.
Embedding Generation: Uses Word2Vec to generate embeddings for the extracted text.
Vector Search: Utilizes Pinecone for efficient similarity search.
Question Answering: Uses Meta-Llama-3-8B-Instruct to answer questions based on the relevant text.


## **Requirements**
Python 3.7+
LangChain
Pinecone
Gensim (for Word2Vec)
PyMuPDF (for PDF text extraction)
Transformers (for Meta-Llama-3-8B-Instruct)

## **Installation**
Clone the repository:

git clone https://github.com/Ch-Praneeth1/PDF-based-Question-Answering-System.git


## **Install the dependencies:**

pip install -r requirements.txt


## **Set up Pinecone:**

Sign up for Pinecone and get your API key.
Create a Pinecone index:

pinecone.init(api_key="YOUR_PINECONE_API_KEY")
pinecone.create_index(name="pdf-qa", dimension=300)

## **Contributing**
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.
