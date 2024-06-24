import streamlit as st
import helper.helper_functions as helper_functions
import numpy as np

st.title("PDF-based Question Answering System.")
uploaded_file = st.file_uploader(label="Upload your PDF..",type='pdf')
if uploaded_file is not None:
    data = helper_functions.extract_text_from_pdf(uploaded_file)

    text_chuncks = helper_functions.spilt_text_into_chuncks(data)
    # st.write(text_chuncks)


    doc_search = helper_functions.doc_search_fun(text_chuncks)

    agent = helper_functions.agent_fun(doc_search)

    user_question = st.chat_input(placeholder="Question your PDF..")

    if user_question!='':
        with st.chat_message("Human"):
            st.write(user_question)

        with st.chat_message("AI"):
            ans = agent.invoke(user_question)
            helpful_answer = ans['result'].split("Helpful Answer:")[1].strip()
            # print(helpful_answer)
            st.write(helpful_answer)