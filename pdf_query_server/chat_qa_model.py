from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from input_output import InferenceInput, InferenceOutput
from operator import itemgetter

template = """
Answer the question based on the context below. If you can't
answer the question, reply "I don't know"

Context: {context}

Question: {question}
"""

class chat_qa_class:
    def __init__(self, model) -> None:
        self.model_name = model

    def load_model(self):
        model = Ollama(model = self.model_name)
        embeddings = OllamaEmbeddings(model = self.model_name)
        parser = StrOutputParser()
        prompt = PromptTemplate.from_template(template)

        loader = PyPDFLoader("user_manual.pdf")
        pages = loader.load_and_split()
        vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        self.chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
            }
            | prompt
            | model
            | parser
        )

    def run_inference(self, query):
        return self.chain.invoke({"question" : query})