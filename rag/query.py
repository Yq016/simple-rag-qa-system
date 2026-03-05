from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# embedding
embedding = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 加载向量数据库
vectorstore = FAISS.load_local(
    "faiss_index",
    embedding,
    allow_dangerous_deserialization=True
)

# LLM
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2-1.5B-Instruct",
    max_new_tokens=150
)

print("RAG Knowledge Assistant (type 'exit' to quit)")

while True:

    question = input("\nQuestion: ")

    if question.lower() == "exit":
        break

    docs = vectorstore.similarity_search(question, k=2)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful AI assistant called RAG Knowledge Assistant.

Use the context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    result = generator(prompt)

    answer = result[0]["generated_text"]
    answer = answer.split("Answer:")[-1].strip()

    print("\nAnswer:", answer)