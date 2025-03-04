import warnings
import getpass
import os


from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured.partition.pdf import partition_pdf
from unstructured_client.models.errors import SDKError
from langchain_community.embeddings import HuggingFaceEmbeddings




from unstructured.chunking.title import chunk_by_title
#from unstructured.partition.md import partition_md
#from unstructured.partition.pptx import partition_pptx
from unstructured.staging.base import dict_to_elements
from IPython.display import Image
from io import StringIO
from lxml import etree
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import panel as pn 


import chromadb

from utils import utils
from utils import upld_file


warnings.filterwarnings('ignore')
utils = utils()
open_key = utils.get_openapi_key()
api_key = utils.get_api_key(
)
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
#api_key = utils.get_api_url()

s = UnstructuredClient(
    api_key
    #url_key = utils.get_api_url(),
)

#Image(filename="physicstest1.pdf",height=400,width=400)
#Image(filename='',height=400,width=400)
filename = "dataprocessing/donut_paper.pdf"

with open(filename, "rb") as f:
    files=shared.Files(
        content=f.read(),
        file_name=filename,
    )
    
#req = shared.PartitionParameters(
#    files = files,
#    strategy = "hi_res",
#    hi_res_model_name="yolox",
#    skip_infer_table_types=[],
#)


try:
    elements = partition_pdf("dataprocessing/donut_paper.pdf",
                         strategy='hi_res',
                         infer_table_structure=True,
           )
    #pdf_elements = dict_to_elements(elements)
    pdf_elements = elements
except SDKError as e:
    print(e)
#pdf_elements[0].to_dict()
tables = [el for el in elements if el.category == "Table"]
tabel_html =tables[0].metadata.text_as_html
parser = etree.XMLParser(remove_blank_text=True)
file_obj = StringIO(tabel_html)
tree = etree.parse(file_obj, parser)
print(etree.tostring(tree, pretty_print=True).decode())


elements = chunk_by_title(pdf_elements)

document = []
for element in elements:
    metadata = element.metadata.to_dict()
    del metadata["languages"]
    metadata["source"] = metadata["filename"]
    document.append(Document(page_content=element.text, metadata=metadata))
#model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(document,embeddings)
retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs={"k":6}
)
template = """You are an AI assistant for answering questions about the document. You are given the following exracted parts of a long document
and a question. Provide a conversational answer. If you don't know the answer, just say "Hmm?". Don't try to make up an answer.
Question:{question}
=========
{context}
=========
Answer in Markdown:"""
prompt = PromptTemplate(template=template, input_variables=["question","context"])

llm = OpenAI(temperature=0)

doc_chain = load_qa_with_sources_chain(llm)
question_generator_chain = LLMChain(llm=llm, prompt=prompt)
qa_chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator_chain,
    combine_docs_chain=doc_chain,
)
qa_chain.invoke({
    "question": "Could you summerize this paper?",
    "chat_history": []
})["answer"]

filter_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":1,"filter":{"source":filename}}
)
filter_chain = ConversationalRetrievalChain(
    retriever = filter_retriever,
    question_generator = question_generator_chain,
    combine_docs_chain=doc_chain,
    
)
filter_chain.invoke(
    {
        "question": "What is important about donut?",
        "chat_history": [],
        "filter": filter,
    }
)["answer"]

#pn.extenstion()
#upld_widget = upld_file()
#pn.Row(upld_widget.widget_file_upload)