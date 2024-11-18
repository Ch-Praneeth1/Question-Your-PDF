from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
import os
from langchain.llms import huggingface_hub
import PyPDF2
import numpy as np
from langchain.llms import Anthropic
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
PINECONE_API = os.getenv("PINECONE_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = TOKEN
os.environ["PINECONE_API_KEY"] = PINECONE_API



def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def spilt_text_into_chuncks(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20
    )
    # print(dir(text_splitter))
    text_chuncks = text_splitter.split_text(data)
    return text_chuncks


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def doc_search_fun(text_chuncks):
    doc_search = PineconeVectorStore.from_texts(text_chuncks,embedding=embedding,index_name="test-index")
    return doc_search

def agent_fun(doc_search):
    llm = huggingface_hub.HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3-8B-Instruct",model_kwargs={'temperature':0.5})
    # llm = Anthropic()
    
    agent = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=doc_search.as_retriever())
    return agent
