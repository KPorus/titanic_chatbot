import os
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
from langchain_huggingface import HuggingFaceEndpoint
from langchain_experimental.tools.python.tool import PythonREPLTool
from Backend.load_datasets import load_Dataset
import json, re
import base64
import matplotlib.pyplot as plt

# Example helper function to generate a plot and return a base64 encoded string.
# You can adjust this function or have the agent generate plots dynamically.
def save_plot_and_return_base64():
    plt.figure()
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    plt.title("Example Plot")
    plot_path = "plot.png"
    plt.savefig(plot_path)
    plt.close()
    with open(plot_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded

load_dotenv()
hugging_face_api_key = os.getenv("hugging_face_api_key")
df = load_Dataset()

llm = HuggingFaceEndpoint(
    temperature=0.1,
    repo_id="tiiuae/falcon-7b-instruct",
)
tools = [PythonREPLTool()]

# Updated prompt: the agent is now instructed to output a JSON object with two keys:
# "response" for text and "plot" for the base64 image (or an empty string if no visualization).
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,  # This will allow the agent to try again on parsing errors.
    allow_dangerous_code=True,
    tools=tools,
    prefix = """You are an AI assistant for data scientists and data analysts working exclusively on the Titanic dataset. 
You have access to a Python REPL tool to execute code on the dataset, which is loaded as a pandas DataFrame named 'df'. 
If appropriate, you can generate visualizations. When a visualization is needed, generate the plot using Python code,
save the plot (for example, as a PNG), and then encode the image in base64 format so that it can be rendered on the frontend.

When answering questions:
1. Think step by step internally.
2. Use the Python REPL tool to compute any required values or generate plots if needed.
3. **Output exactly one valid JSON object (and nothing else) enclosed in triple backticks.**
   The JSON object must have exactly two keys:
   - "response": a string containing your textual answer.
   - "plot": a string containing the base64 encoded image of the plot if a visualization is generated, or an empty string ("") if no visualization is needed.
Do not include any extra text, commentary, or nested JSON.

For example:
Question: How many passengers survived?
Thought: I will use the Python REPL to compute the survivors from the 'Survived' column.
Observation: The result is 342.
Final Answer:
{{"response": "342 passengers survived.", "plot": ""}}
Now, answer this question:
Question: {input}
Thought:
"""
)

# Modified query function that extracts a JSON block with two keys.
def query_titanic(query):
    try:
        response = agent.run({'input': query})
        
        # Attempt to extract the JSON block enclosed in triple backticks.
        json_match = re.search(r'```(\{.*?\})```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                result = json.loads(json_str)
                text_response = result.get("response", "Error: 'response' key not found")
                plot_data = result.get("plot", "")
                return {"response": text_response, "plot": plot_data}
            except Exception as e:
                return {"response": f"Error: Could not parse JSON. Details: {str(e)}", "plot": ""}
        
        # Fallback: search for any JSON-like substring.
        json_candidates = re.findall(r'(\{.*\})', response)
        if json_candidates:
            try:
                result = json.loads(json_candidates[-1])
                text_response = result.get("response", "Error: 'response' key not found")
                plot_data = result.get("plot", "")
                return {"response": text_response, "plot": plot_data}
            except Exception:
                return {"response": "Error: Could not parse JSON from fallback extraction", "plot": ""}
        return {"response": "Error: Could not parse the final answer", "plot": ""}
    except Exception as e:
        return {"response": f"An error occurred: {str(e)}", "plot": ""}


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