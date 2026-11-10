import zipfile
import os
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

with open('../../config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
os.environ['PINECONE_API_KEY'] = config['PINECONE_API_KEY']
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']

# Busca Semantica
zip_file_path = '../../../documentos.zip'
extracted_folder_path = 'docs'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

documents = []
for filename in os.listdir(extracted_folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(extracted_folder_path, filename)
        loader = PyMuPDFLoader(file_path)
        documents.extend(loader.load())
