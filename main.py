import os
from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from PyPDF2 import PdfReader
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
#from langchain.vectorstores import FAISS 
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
#from langchain.llms import OpenAI
#from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

#from openai_client import OpenAIClient

from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()

if not os.path.exists("./db"):

    print("CREANDO DB")
    doc_reader = PdfReader('./documento.pdf')

    raw_text = ''
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 300,
        chunk_overlap  = 30, 
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    vectorstore = FAISS.from_texts(texts, embeddings)

    vectorstore.save_local("./db")

else:
    print("LOADING DB")
    vectorstore = FAISS.load_local("./db", embeddings)


'''
doc_reader = PdfReader('./documento.pdf')

raw_text = ''
for i, page in enumerate(doc_reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200, 
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_texts(texts, embeddings)
'''

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":1})
#retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})

template = """Usa el siguiente contexto para responder las preguntas.
La respuesta debe estar en 3ra persona.
Si en la pregunta se menciona a una persona, el nombre también debe estar en la respuesta.
Responer siempre en una sola oración.
Agrega un emoji que resuma la respuesta.
La respuesta debe estar siempre en el mismo idioma de la pregunta.
{context}
Pregunta: {question}
Respuesta:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

#llm=OpenAI(model="gpt-4-1106-preview"), 
rqa = RetrievalQA.from_chain_type(
    llm=OpenAI(), 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

app = FastAPI()

@app.get("/response")
async def get_response(user_name: str, query: str):
    res = {
        "user_name": user_name,
        "query": query
    }

    res = rqa(query)['result']

    return res

@app.post("/response2")
async def get_response2(request: Request):
    data = await request.json()
    print(data["query"])
    print(data["user_name"])
    res = data
    return res
