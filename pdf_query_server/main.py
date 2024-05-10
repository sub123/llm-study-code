from fastapi import FastAPI, HTTPException

from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter
from input_output import InferenceInput, InferenceOutput
from contextlib import asynccontextmanager
from chat_qa_model import chat_qa_class

chat_qa = chat_qa_class("llama2")
chat_qa.load_model()


app = FastAPI()

@app.post("/chatqa")
async def completion(input_data: InferenceInput):
    print(input_data)
    res = ''
    res = chat_qa.run_inference(user_input=input_data.user_input)

    return InferenceOutput(inference_output=res)