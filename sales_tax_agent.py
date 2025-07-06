from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import os

# 1. Load your structured sales tax data
# Replace this with your actual data loading step
data = {
    "state": ["CA", "TX", "NY", "CA", "TX", "NY"],
    "month": ["2024-01", "2024-01", "2024-01", "2024-02", "2024-02", "2024-02"],
    "total_sales": [100000, 85000, 120000, 110000, 83000, 115000],
    "state_tax_rate": [0.06, 0.0625, 0.04, 0.06, 0.0625, 0.04],
    "local_tax_rate": [0.025, 0.0125, 0.045, 0.0275, 0.015, 0.045],
    "collected_tax": [8250, 6375, 10200, 9625, 6450, 10350]
}
df = pd.DataFrame(data)

# 2. Initialize LLM (use your actual OpenAI API key via environment variable)
os.environ["OPENAI_API_KEY"] = "sk-...your-key..."
llm = ChatOpenAI(temperature=0, model="gpt-4o")

# 3. Create the Pandas Agent with the DataFrame
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    agent_type="openai-tools"
)

# 4. Prompt examples for interaction
def query_agent(prompt):
    print("\n> User:", prompt)
    response = agent.run(prompt)
    print("\n> Assistant:\n", response)

# 5. Example questions
query_agent("Which state had the highest average local tax rate?")
query_agent("Compare total sales and collected tax between CA and TX for each month.")
query_agent("Find any months where the collected tax seems too low based on total sales and tax rates.")
