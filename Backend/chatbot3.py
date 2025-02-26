
import os
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_experimental.tools.python.tool import PythonREPLTool
from Backend.load_datasets import load_Dataset
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()
hugging_face_api_key = os.getenv("hugging_face_api_key")
df = load_Dataset()

llm = HuggingFaceEndpoint(
    temperature=0.1,
    repo_id="tiiuae/falcon-7b-instruct",
)
tools = [PythonREPLTool()]

# Create the agent with error handling
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,  # Critical for recovering from errors
    allow_dangerous_code=True,
    tools=tools,
    prefix = """You are an AI assistant tasked with analyzing the Titanic dataset using Python. 
You have access to a Python REPL tool to execute code on the dataset, which is loaded as a pandas DataFrame named 'df'.

To answer questions:
1. Think step by step, explaining your reasoning.
2. Use the Python REPL tool to calculate the answer if needed.
3. Provide the final answer **exactly** as valid JSON, fenced by triple backticks (```), with one field called "response" containing the answer as a string.

For example:
Question: How many passengers survived?
Thought: I need to count survivors. The 'Survived' column uses 1 for survived and 0 for not survived. I’ll use the Python REPL to sum this column.
Action: Use Python REPL to run `df['Survived'].sum()`
Observation: The result is 342.
Thought: So, 342 passengers survived.
Final Answer:
{{"response": Observation}}
**Do not include extra text like 'The final answer is' or use single backticks (`) in the Final Answer section.** Only provide the JSON between triple backticks.

Now, answer this question:
Question: {input}
Thought:
"""
)



# Query function
def query_titanic(query):
    try:
        response = agent.run({'input': query})
        import json
        import re

        # Try to find JSON in triple backticks first (preferred format)
        json_match = re.search(r'Final Answer:\s*```\s*(\{.*?\})\s*```', response, re.DOTALL | re.IGNORECASE)
        if json_match:
            response_json = json.loads(json_match.group(1))
            return response_json["response"]

        # Fallback: Extract number from "The final answer is `X`"
        fallback_match = re.search(r'Final Answer:.*?`([\d.]+)`', response, re.DOTALL | re.IGNORECASE)
        if fallback_match:
            return fallback_match.group(1)

        return "Error: Could not parse the final answer"
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
#     class TitanicResponse(BaseModel):
#     response: str = Field(..., description="A response to the question about the Titanic dataset.")

# output_parser = PydanticOutputParser(pydantic_object=TitanicResponse)

# # Revised prompt template
# template = """
# You are given a dataset about the Titanic.
# Question: {query}
# Your answer must be in valid JSON fenced by triple backticks. Do not output any additional text.
# The JSON must have exactly one field:
# "response": a string containing your answer.
# Example output:
# """
# prompt = PromptTemplate(template=template, input_variables=["query"])
# chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)



#     prefix="""You are given a dataset about the Titanic.
# Question: {input}
# Your answer must be in valid JSON fenced by triple backticks. Do not output any additional text.
# The JSON must have exactly one field:
# "response": a string containing your answer.
# Example output:
# """