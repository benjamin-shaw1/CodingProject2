import warnings
warnings.filterwarnings('ignore')

from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError

from unstructured.chunking.title import chunk_by_title
#from unstructured.partition.md import partition_md
#from unstructured.partition.pptx import partition_pptx
from unstructured.staging.base import dict_to_elements
from IPython.display import Image
from io import StringIO
from lxml import etree
from langchain_community.vectorstores import chroma
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

utils = utils()

api_key = utils.get_api_key(
)
#api_key = utils.get_api_url()

s = UnstructuredClient(
    api_key = utils.get_api_key()
    #url_key = utils.get_api_url(),
)
Image(filename="physicstest1.pdf",height=400,width=400)
#Image(filename='',height=400,width=400)
filename = "physicstest1.pdf"

with open(filename, "rb") as f:
    files=shared.Files(
        content=f.read(),
        file_name=filename,
    )
    
req = shared.PartitionParamerters(
    files = files,
    strategy = "hi_res",
    hi_res_model_name="yolox",
    pdf_inter_table_structure = True,
    skip_infer_table_types=[],
)

try:
    resp = s.general.partition(req)
    pdf_elements = dict_to_elements(resp.elements)
except SDKError as e:
    print(e)
pdf_elements[0].to_dict()
tables = [el for el in pdf_elements if el.category == "Table"]
tabel_html =tables[0].metadata.text_as_html
parser = etree.XMLParser(remove_blank_text=True)
file_obj = StringIO(tabel_html)
tree = etree.parse(file_obj, parser)
print(etree.tostring(tree, pretty_print=True).decode())
Image(filename="physicstest1.pdf", height=400, width=400)
reference_title = [
    el for el in pdf_elements
    if el.text == "Reference"
    and el.category =="Title"
][0]

reference_title.to_dict()

reference_id = reference_title.id

for element in pdf_elements:
    if element.metadata.parent_id == reference_id:
        print(element)
        break
    
pdf_elements = [el for el in pdf_elements if el.metadata.parent_id]

Image(filename="physicstest1.pdf", height=400, width=400)

headers = [el for el in pdf_elements if el.category == "Header"]

headers[1].to_dict()

pdf_elements = [el for el in pdf_elements if el.category != "Header"]

elements = chunk_by_title(pdf_elements)

document = []
for element in elements:
    metadata = element.metadata.to_dict()
    del metadata["languages"]
    metadata["source"] = metadata["filename"]
    document.append(Document(page_content=element.text, metadata=metadata))
embeddings = OpenAIEmbeddings()
vectorstore = chroma.from_documents(document,embeddings)
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

doc_chain = load_qa_with_sources_chain(llm, chain_types="map_reduce")
question_generator_chain = LLMChain(llm=llm, prompt=prompt)
qa_chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator_chain,
    combine_docs_chain=doc_chain,
)
qa_chain.invoke({
    "question": "what is the magnitude of the resultant?",
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
        "question": "How big is the force being applied at a 25 degree angle?",
        "chat_history": [],
        "filter": filter,
    }
)["answer"]

#pn.extenstion()
#upld_widget = upld_file()
#pn.Row(upld_widget.widget_file_upload)