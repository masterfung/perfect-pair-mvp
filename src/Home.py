import streamlit as st


#Config
st.set_page_config(layout="wide", page_icon="💬", page_title="Perfect Pair 🤖")


#Title
st.markdown(
    """
    <h2 style='text-align: center;'>Perfect Pair</h1>
    """,
    unsafe_allow_html=True,)

st.markdown("---")


#Description
st.markdown(
    """ 
    <h5 style='text-align:center;'>Perfect Pair, an intelligent chatbot created by combining 
    the strengths of Langchain and Streamlit to help find bottle possibilities for sales reps. 
    I use large language models to provide
    context-sensitive interactions. My goal is to help you better understand your data.
    I support PDF, TXT, and CSV</h5>
    """,
    unsafe_allow_html=True)
st.markdown("---")




