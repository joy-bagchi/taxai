import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import os

# Set your OpenAI API key securely
# os.environ["OPENAI_API_KEY"] = "sk-...your-key..."

st.set_page_config(page_title="Sales Tax Analyzer", layout="wide")
st.title("🧾 Sales Tax Data Analyzer with LLM + Pandas Agent")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your sales tax CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV loaded successfully!")
    st.write("Preview of data:")
    st.dataframe(df.head())

    # Initialize the LLM and agent
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        agent_type="openai-tools",
        allowed_tools=["pandas"],
        allow_dangerous_code=True
    )

    st.markdown("### Ask a question about your data:")
    user_input = st.text_area("Your question", placeholder="e.g., Which state had the highest local tax rate in Q1 2024?")

    if st.button("Submit") and user_input:
        with st.spinner("Analyzing with agent..."):
            try:
                response = agent.run(user_input)
                st.markdown("### 🧠 Assistant Response")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("Please upload a CSV file to get started.")
