import streamlit as st
from analysis import load_and_analyze_data
from evaluate import evaluate_model
from memory import setup_memory
from retriever import get_custom_retriever
from prompt_engineering import generate_response
import matplotlib.pyplot as plt

st.title("📊 InsightForge: Business Intelligence Assistant")

df, stats, plots = load_and_analyze_data("data/sales_data.csv")

st.subheader("Summary Statistics")
st.write(stats)

st.subheader("Visualizations")
for fig in plots:
    st.pyplot(fig)

st.subheader("Ask a Business Question")
query = st.text_input("Your question:")
if query:
    retriever = get_custom_retriever(df)
    memory = setup_memory()
    response = generate_response(query, retriever, memory)
    st.write(response)

st.subheader("Model Evaluation")
if st.button("Run Evaluation"):
    eval_result = evaluate_model()
    st.write(eval_result)
