import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain_community.document_loaders import PyPDFLoader
import os
import requests
import tempfile

# Function to load text from uploaded PDF files
def load_text_from_pdf(uploaded_files):
    all_text = ""
    for uploaded_file in uploaded_files:
        # Save the uploaded PDF file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Use the saved file path to load PDF
        pdf_loader = PyPDFLoader(tmp_file_path)
        documents = pdf_loader.load()
        for doc in documents:
            all_text += doc.page_content

        # Delete the temporary file
        os.unlink(tmp_file_path)

    return all_text

# Model and tokenizer
model_name = 'google-bert/bert-large-cased-whole-word-masking-finetuned-squad'#"deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Streamlit app
#st.title('Sayakbotix: Instant AI chatbot:')
st.title(':red[Saybotix : ]:blue[ Instant AI Chatbot] :sunglasses:')
st.sidebar.image('download.png', use_column_width=True)

#st.image('download.png', use_column_width=True)

# File upload
uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)

if uploaded_files is not None:
    # Load text from uploaded PDF files
    try:
        all_text = load_text_from_pdf(uploaded_files)
    except Exception as e:
        st.error(f"Error loading PDF files: {str(e)}")
        st.stop()

    # Input question
    question = st.text_input('Enter your question:')
    
    if st.button('Search'):
        if question.strip() == '':
            st.warning('Please enter a question.')
        else:
            QA_input = {
                'question': question,
                'context': all_text
            }
            try:
                res = nlp(QA_input)
                st.write("Answer : ", res["answer"])
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")


#vector databese are -- pinecone vector database,chroma db, Chromadb is a vector database for building AI applications with embeddings. pinecone is a cloud based vector database . for similarity search we use cosine similarity.
# FAISS- Facebook AI similarity search is a memeory databaseused for similarity search. vector database store the embeddings / vector 