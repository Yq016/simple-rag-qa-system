from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import os

docs = []

data_dir = "data"

for file in os.listdir(data_dir):

    path = os.path.join(data_dir, file)

    if file.endswith(".txt"):
        loader = TextLoader(path)
        docs.extend(loader.load())

    elif file.endswith(".pdf"):
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

print("Loaded documents:", len(docs))

# 切分文本
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = text_splitter.split_documents(docs)

print("Split chunks:", len(texts))

# embedding
embedding = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 建立向量库
vectorstore = FAISS.from_documents(texts, embedding)

vectorstore.save_local("faiss_index")

print("Vector index built successfully!")