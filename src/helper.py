# from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Extract Data from PDF
def load_data(data_path):
    loader = DirectoryLoader(data_path,glob='*.pdf',loader_cls=PyPDFLoader)
    data = loader.load()
    return data

# Create Text Chunks
def text_split(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap= 50)
    text_chunks = splitter.split_documents(data)
    return text_chunks

# download embedding model
def download_huggingface_embedding():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    return embeddings