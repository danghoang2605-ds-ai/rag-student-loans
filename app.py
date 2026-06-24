import streamlit as st
from rag import ask

st.title("UGA Student Loans Assistant")
st.write("Ask me anything about student loans, FAFSA, or financial aid!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask(prompt)
                answer = result["result"]
                sources = result["source_documents"]
                st.write(answer)
                with st.expander("Sources"):
                    for doc in sources:
                        st.write(f"- {doc.metadata.get('source', 'unknown')}")
            except Exception as e:
                answer = f"Sorry, I ran into an error answering that: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})