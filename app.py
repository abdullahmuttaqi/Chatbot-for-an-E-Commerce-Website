import os
import re
import requests
from flask import Flask, render_template, request, redirect
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from flask_ngrok import run_with_ngrok
from dotenv import load_dotenv
import webbrowser

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")  # Ensure your API key is stored in .env

# Flask App
app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when the app runs
vectorstore = None
conversation_chain = None
chat_history = []  # Initialize chat history as empty

def get_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def process_documents():
    global vectorstore, conversation_chain
    # Read text from website.txt
    raw_text = get_text_from_file('website.txt')
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)

def extract_links(text):
    # Regex to find URLs
    url_pattern = r'(https?://[^\s]+)'
    links = re.findall(url_pattern, text)
    return links

@app.route('/process', methods=['POST'])
def process_documents_route():
    process_documents()
    return redirect('/chat')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global conversation_chain, chat_history

    if request.method == 'POST':
        user_question = request.form['user_question']
        response = conversation_chain({'question': user_question})
        
        # Extract the answer from the response
        answer = response['answer']  # Adjust this line based on your actual response structure
        
        # Extract links from the answer
        links = extract_links(answer)
        
        # Store both user and bot messages along with links
        chat_history.append({'user': user_question, 'bot': answer, 'links': links})  

    return render_template('chat.html', chat_history=chat_history)

if __name__ == '__main__':
    # Process documents before starting the server
    process_documents()
    webbrowser.open_new('http://127.0.0.1:5000/chat')
    app.run()
