import streamlit as st

st.title("HTML TEST")

st.markdown(
    """
    <div style="background:#10243d;padding:30px;border-radius:15px;color:white;">
        <h1>SeizureSense AI</h1>
        <p>HTML IS RENDERING CORRECTLY</p>
    </div>
    """,
    unsafe_allow_html=True
)
