# push vector to vector DB,
from src.helper import load_data, text_split, download_huggingface_embedding
from langchain_community.vectorstores import Pinecone
# from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# print(PINECONE_API_KEY )

PINECONE_INDEX_NAME = "med-chat"

extracted_data = load_data(r"D:\Medical_Chatbot\data")
text_chunks = text_split(extracted_data)
embeddings = download_huggingface_embedding()

vectorstore_from_docs = PineconeVectorStore.from_documents(
        text_chunks,
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
print(vectorstore_from_docs)

# completed store index,
# now store vectors to vdb -> app. comp -> prompt