# Question-Answering-RAG-system

**Introduction :**

Imagine you have a lot of documents, like books, pdf files or research papers, and you need to find specific information quickly based on user questions. But reading through all of the documents takes too much time and effort. There are multiple AI tools those are build on LLM(chatgpt,bard etc), but they can’t be answered based on our document. They are trained on large amount of data, but they can’t answered our queries like information about any organization or information about any events or anything. How can we make it easier to find what we need in these documents without spending hours reading them? In that case, RAG is coming into the picture. RAG can handle that kind of problem efficiently. It reads through the documents and finds the answers for us based on queries. For implement RAG based question answering system I use ‘google-bert/bert-large-cased-whole-word-masking-finetuned-squad’ model.

**Model details:**

BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. 

**Library used:**

* **Streamlit:** It is used for deployment.
* **Langchain:** It is used to build Large Language model.
* **Tranasformer:** It is used for Large language model.

**Workflow :**

*	**Document Upload:** First, upload the documents. I deploy it using Streamlit. There is an option to load the files. These could be books, articles, or any text you need information from. And the documents are converted into embedding and store the vectors into vector DB.
*	
*	**Preprocessing:** Split the documents into chunks and preprocess them. And the chunks are converted into embedding and store the vectors into vector DB.
  
* **Question Input:** Next, you ask a question about the information you're looking for inside the interface. For example, Here I upload lie detection research paper, I can ask the question like “What is lie detection?”. This question converts to word embedding.
  
* **Document Reading:** The RAG system reads through all the uploaded documents to find the answer to your question. The whole document is converted into embedding vector and stored into vector database.
  
* **Answer Retrieval:** It retrieves the answer from vector database using cosine similarity. It searches linearly and using cosine similarity rag retrieve the best answer. It searches through the text to find the most relevant information that answers your question.
  
*	**Answer Generation:** Finally, the RAG system gives you the answer to your question. It might show you a sentence or a paragraph from one of the documents that contains the information you're looking for.
  
*	**Deployment:** Deploy the RAG based question answer system using streamlit for better user experience.

![image](https://github.com/bittu5555/Question-Answering-RAG-system/assets/106305917/65268d56-aa2c-4d98-b51e-c1515d83b1ee)


**Result**

![image](https://github.com/bittu5555/Question-Answering-RAG-system/assets/106305917/f3978ba6-4d6f-4702-98b0-c8c9e023324f)

