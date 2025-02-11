from flask import Flask, render_template
from flask import request, jsonify, abort
from langchain.prompts import PromptTemplate
from langchain.llms import Cohere
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
import os

app = Flask(__name__)

load_dotenv()

# Initialize LangChain with Cohere
llm = Cohere(model="command", temperature=0.7, cohere_api_key=os.environ.get("COHERE_API_KEY"))
prompt = PromptTemplate(
    input_variables=["user_input"],
    template="Answer the query politely and informatively:\n\nUser: {user_input}\nAI:"
)
def load_db():
    try:
        embeddings = CohereEmbeddings(cohere_api_key=os.environ.get("COHERE_API_KEY"))
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
        qa = RetrievalQA.from_chain_type(
            llm=Cohere(),
            chain_type="stuff", #refine here does not work.. it gives error in refine.py file
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        return qa
    except Exception as e:
        print("Error:", e)

qa = load_db()

def answer_from_knowledgebase(message):
    print(qa)
    res = qa({"query": message})
    return res['result']

def search_knowledgebase(message):

    res = qa({"query": message})
    sources = ""
    for count, source in enumerate(res['source_documents'],1): #Source_documents are never returning. 
        #tried my db and the db provided in example.. also searched a lot and a lot of ppl are facing the same issue
        sources += "Source " + str(count) + "\n"
        sources += source.page_content + "\n" #sources did not work
    return res['result']

def answer_as_chatbot(message):
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    formatted_prompt = prompt.format(user_input=user_message)

    # Generate response using Cohere's model
    response = llm(formatted_prompt)

    return response

@app.route('/kbanswer', methods=['POST'])
def kbanswer():
    message = request.json['message']

    # Generate a response
    response_message = answer_from_knowledgebase(message)
    
    # Return the response as JSON
    return jsonify({'message': response_message})

@app.route('/search', methods=['POST'])
def search():    
    message = request.json['message']

    response_message = search_knowledgebase(message)
    return jsonify({'message': response_message})

@app.route('/answer', methods=['POST'])
def answer():
    message = request.json['message']

    # Generate a response
    response_message = answer_as_chatbot(message)
    
    # Return the response as JSON
    return jsonify({'message': response_message})

@app.route("/")
def index():
    return render_template("index.html", title="")

if __name__ == "__main__":
    app.run()