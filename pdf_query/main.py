from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter

MODEL = "llama2"

model = Ollama(model = MODEL)
#print(model.invoke("Tell me a joke"))
embeddings = OllamaEmbeddings(model = MODEL)
parser = StrOutputParser()

chain = model | parser
print(chain.invoke("Tell me a joke"))

loader = PyPDFLoader("user_manual.pdf")
pages = loader.load_and_split()
#print(pages)

template = """
Answer the question based on the context below. If you can't
answer the question, reply "I don't know"

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)
#print(prompt.format(context = "My name is Subham", question="What is my name?"))

chain = prompt | model | parser
print(chain.invoke(
    {
        "context" : "My name is Subham",
        "question" : "What is my name?"
    }
))

vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = vectorstore.as_retriever()
retriever.invoke("Vacuum cleaner")

print("##########################################")

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | parser
)

print(chain.invoke({"question" : "Who is the manufacturer of stick vacuum cleaner device?"}))