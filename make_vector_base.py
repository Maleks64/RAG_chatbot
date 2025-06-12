from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


loader = TextLoader("boc_site_all_pages_formatted.md",encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print("done")

embedding = HuggingFaceEmbeddings(model_name='Qwen/Qwen3-Embedding-8B')

vectorstore = FAISS.from_documents(chunks, embedding)

vectorstore.save_local("faiss_vector_base")
