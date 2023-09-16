from flask import Flask, request, jsonify, render_template
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os
from flask_ngrok import run_with_ngrok
from flask_cors import CORS

from flask import render_template

app = Flask(__name__)

run_with_ngrok(app)

@app.route('/', methods=['GET'])
def home_page():
    return render_template('index.html')


def load_page_data(webpage_link):
    # Load the web page content
    loader = WebBaseLoader(webpage_link)
    return loader.load()

def process_page_content(page_data):
    # Process the web page content
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(page_data)
    return all_splits

def create_vectorstore(all_splits):
    # Create a vectorstore from processed documents
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    return vectorstore

def create_qa_chain(vectorstore, user_question):
    # Create a QA chain for question answering
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use five sentences maximum and keep the answer as concise as possible.
    Always say "Thanks for asking!" at the end of the answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # Get the answer to the user's question
    result = qa_chain({"query": user_question})
    answer = result["result"]
    return answer


@app.route('/api/question_answer', methods=['GET', 'POST'])
def question_answer():
    try:
        data = request.json
        webpage_link = data['webpageLink']
        user_question = data['userQuestion']

        print(webpage_link)
        print(user_question)

        page_data = load_page_data(webpage_link)
        all_splits = process_page_content(page_data)
        vectorstore = create_vectorstore(all_splits)
        answer = create_qa_chain(vectorstore, user_question)

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run()
