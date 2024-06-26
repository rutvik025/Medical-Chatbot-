from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embedding
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA # chat with LLMs
from dotenv import load_dotenv
from src.prompt import *
import os
# import streamlit as st

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

embeddings = download_huggingface_embedding()

index_name="med-chat"

#Loading the index
docsearch=Pinecone.from_existing_index(index_name, embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':500,
                          'temperature':0.7})

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

    
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)



# from flask import Flask, render_template, jsonify, request
# from src.helper import download_huggingface_embedding
# from langchain_community.vectorstores import Pinecone
# import pinecone
# from langchain.prompts import PromptTemplate
# from langchain_community.llms import CTransformers
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv
# from src.prompt import *
# import os
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
# import logging

# app = Flask(__name__)

# # Load environment variables
# load_dotenv()

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# if not PINECONE_API_KEY:
#     raise ValueError("Pinecone API key not found. Please set the PINECONE_API_KEY environment variable.")

# # Initialize Pinecone
# pinecone.init(api_key=PINECONE_API_KEY)

# # Initialize embeddings
# embeddings = download_huggingface_embedding()

# # Define index name
# index_name = "medical-chatbot"

# # Load existing Pinecone index
# docsearch = Pinecone.from_existing_index(index_name, embeddings)

# # Define the prompt template
# PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# # Set chain type parameters
# chain_type_kwargs = {"prompt": PROMPT}

# # Configure the LLM
# llm = CTransformers(
#     model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
#     model_type="llama",
#     config={'max_new_tokens': 256, 'temperature': 0.7}
# )

# # Setup the RetrievalQA chain
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=docsearch.as_retriever(search_kwargs={'k': 5}),
#     return_source_documents=True,
#     chain_type_kwargs=chain_type_kwargs
# )

# @app.route("/")
# def index():
#     return render_template('chat.html')

# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     try:
#         msg = request.form["msg"]
#         input_query = msg.strip()
#         if not input_query:
#             return jsonify({"error": "Empty query"}), 400
        
#         logging.info(f"Received query: {input_query}")
#         result = qa({"query": input_query})
#         response = result["result"]
#         logging.info(f"Response: {response}")
#         return jsonify({"response": response})
#     except Exception as e:
#         logging.error(f"Error processing query: {e}")
#         return jsonify({"error": "Internal server error"}), 500

# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     app.run(host="0.0.0.0", port=8080, debug=True)
