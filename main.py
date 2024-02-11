import os
from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

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

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":4})
#retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})

template = """Usa el siguiente contexto para responder las preguntas.
Pensemos paso a paso siguiendo las siguientes Instrucciones:
- Se debe responder en el mismo idioma de la pregunta.
- Agrega un emoji que resuma la respuesta.
- La respuesta debe estar en 3ra persona.
- Si en la pregunta se menciona a una persona, el nombre también debe estar en la respuesta.
- Responer siempre en una sola oración.

{context}

Pregunta: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

app = FastAPI()

@app.get("/response")
async def get_response(user_name: str, query: str):
    res = {
        "user_name": user_name,
        "query": query
    }

    res = chain.invoke(query)

    return res

@app.post("/response2")
async def get_response2(request: Request):
    data = await request.json()
    print(data["query"])
    print(data["user_name"])
    query = data["query"]
    res = chain.invoke(query)
    return res
