import os

import streamlit as st
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores.neo4j_vector import Neo4jVector
from streamlit.logger import get_logger
from chains import (
    load_embedding_model,
    load_llm,
)

# load api key lib
from dotenv import load_dotenv

load_dotenv(".env")


url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)


embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})

# Streamlit UI
styl = f"""
<style>
    .stChatFloatingInputContainer {{
        bottom: 20px;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)
# create separate functions for the chat history and the chat input
def display_chat_history():
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = []
    if "generated_output" not in st.session_state:
        st.session_state["generated_output"] = []
    if st.session_state[f"generated_output"]:
        size = len(st.session_state[f"generated_output"])
        # Display only the last three exchanges
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input"][i])
            with st.chat_message("assistant"):
                st.write(st.session_state[f"generated_output"][i])
        with st.container():
            st.write("&nbsp;")
def chat_input(qa):
    query = st.chat_input("Ask questions about related your upload pdf files")
    if query:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            qa.run(query, callbacks=[stream_handler])
            # add the query to user input
            st.session_state[f"user_input"].append(query)
            # add the answer to generated output
            text = stream_handler.text
            st.session_state[f"generated_output"].append(text)


def main():
    st.header("📄Chat with your pdf files with history")

    # upload a your pdf file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # Store the chunks part in db (vector)
        vectorstore = Neo4jVector.from_texts(
            chunks,
            url=url,
            username=username,
            password=password,
            embedding=embeddings,
            index_name="pdf_bot",
            node_label="PdfBotChunk",
            pre_delete_collection=True,  # Delete existing PDF data
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )

        # # Accept user questions/query
        # query = st.text_input("Ask questions about your PDF file")

        # if query:
        #     stream_handler = StreamHandler(st.empty())
        #     qa.run(query, callbacks=[stream_handler])

        display_chat_history()
        chat_input(qa)


if __name__ == "__main__":
    main()
