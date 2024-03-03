import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


def get_text_from_pdf(pdf_files):
    """
    pdf_files: List of pdf files
    return: String of all the text in all the pdf files
    """
    text = ""
    for pdf in pdf_files:
        file = PdfReader(pdf)
        for page in file.pages:
            text += page.extract_text()
    return text


def chunk_text(raw_text):
    """
    raw_text: String of all the text in all the pdf files
    return: list of chunks of text
    """
    splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    chunks = splitter.split_text(raw_text)
    return chunks


def get_vectorstore(chunks):
    """
    chunks: list of chunks of text
    return: Vectorstore with chunks embedded
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """
    vectorstore: Vectorstore with chunks embedded
    return: Conversation chain
    """
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain


def handle_user_input(user_query):
    """
    user_query: String of user input
    return: String of bot response
    """
    response = st.session_state.conversation({'question': user_query})
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chat", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_query = st.text_input("Ask a question about your data:")
    if user_query:
        handle_user_input(user_query)

    with st.sidebar:
        st.subheader("Your Documents")
        docs = st.file_uploader("Upload your PDFs and Click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..!"):
                pdf_content = get_text_from_pdf(docs)
                chunks = chunk_text(pdf_content)
                vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()