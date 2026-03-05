import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

st.title("🤖 RAG Knowledge Assistant")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# embedding
embedding = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"  # 直接调用
)

# vector db
vectorstore = FAISS.load_local(
    "faiss_index",
    embedding,
    allow_dangerous_deserialization=True
)

# LLM
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2-1.5B-Instruct",
    max_new_tokens=150,
    temperature=0.7
)

# 显示历史聊天
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 用户输入
if prompt := st.chat_input("Ask something about your knowledge base..."):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.write(prompt)

    # RAG检索
    docs = vectorstore.similarity_search(prompt, k=2)

    context = "\n".join([doc.page_content for doc in docs])

    rag_prompt = f"""
You are a helpful AI assistant called RAG Knowledge Assistant.

Use the context to answer the question.

Context:
{context}

Question:
{prompt}

Answer:
"""

    result = generator(rag_prompt)

    answer = result[0]["generated_text"]
    answer = answer.split("Answer:")[-1].strip()

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )