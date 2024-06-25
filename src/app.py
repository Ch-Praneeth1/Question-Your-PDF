import streamlit as st
import helper.helper_functions as helper_functions
import numpy as np
from langchain_core.messages import AIMessage,HumanMessage

st.set_page_config(page_title="Question Your PDF", page_icon="📑")
st.title("Question Your PDF")
uploaded_file = st.file_uploader(label="Upload your PDF..",type='pdf')
if uploaded_file is not None:
    data = helper_functions.extract_text_from_pdf(uploaded_file)

    text_chuncks = helper_functions.spilt_text_into_chuncks(data)
    # st.write(text_chuncks)


    doc_search = helper_functions.doc_search_fun(text_chuncks)

    agent = helper_functions.agent_fun(doc_search)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[]
    user_question = st.chat_input(placeholder="Question your PDF..")

    if user_question!='' and user_question!=None:
        ans = agent.invoke(user_question)
        helpful_answer = ans['result'].split("Helpful Answer:")[1].strip()
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=helpful_answer))
        for message in st.session_state.chat_history:
            if isinstance(message,HumanMessage):

                with st.chat_message("Human"):
                    st.write(message.content)

            if isinstance(message,AIMessage):

                with st.chat_message("AI"):
                    st.write(message.content)