import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Load the GROQ API key
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize Streamlit app
st.title("FUSIONAL AI Freedom Of Thinking")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the user requirement.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to create the vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        embeddings = SentenceTransformerEmbeddings()
        loader = PyPDFDirectoryLoader("./data")  # Data Ingestion
        docs = loader.load()  # Document Loading
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        final_documents = text_splitter.split_documents(docs[:20])  # Splitting
        vectors = FAISS.from_documents(final_documents, embeddings)  # Vector embeddings
        st.session_state.vectors = vectors

# Button to trigger document embedding
if st.button("Embed Documents"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

# Input for user's query
prompt1 = st.text_input("Ask whatever you want: ")

if prompt1 and "vectors" in st.session_state:
    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the retriever
    retriever = st.session_state.vectors.as_retriever()
    
    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Time the response
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    response_time = time.process_time() - start
    st.write(f"Response time: {response_time} seconds")
    
    # Output the answer
    st.write("Answer:", response['answer'])

    # Document similarity search results
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

