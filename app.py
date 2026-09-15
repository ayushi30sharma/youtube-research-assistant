import streamlit as st
from agent import agent

st.title("🎥 YouTube Research Assistant")

st.write("Ask a question about the YouTube video.")

query = st.text_input("Enter your question:")

if query:

    state = {
        "query": query,
        "source": "",
        "context": [],
        "answer": ""
    }

    result = agent.invoke(state)

    st.write("### Answer")
    st.write(result["answer"])

    st.write("### Source")
    
    if result["source"] == "youtube":
        st.write("🎥 YouTube Video")
    else:
        st.write("🌐 Web Search")