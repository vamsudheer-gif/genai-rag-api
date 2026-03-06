import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.title("GenAI RAG Chat")

question = st.text_input("Ask a question about the documents")

if st.button("Ask"):

    response = requests.post(
        API_URL,
        json={"question": question}
    )

    data = response.json()

    st.subheader("Top Retrieved Chunks")

    for chunk in data["top_chunks"]:
        st.write("Source:", chunk["source"])
        st.write(chunk["text"])
        st.write("---")