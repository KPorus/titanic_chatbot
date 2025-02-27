import os
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
from langchain_huggingface import HuggingFaceEndpoint
from langchain_experimental.tools.python.tool import PythonREPLTool
from Backend.load_datasets import load_Dataset
import re
import json

load_dotenv()
hugging_face_api_key = os.getenv("hugging_face_api_key")
df = load_Dataset()

llm = HuggingFaceEndpoint(
    temperature=0.7,
    # repo_id="gpt2",
    # repo_id="tiiuae/falcon-7b-instruct",
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
)

prefix = """You are an AI assistant analyzing the Titanic dataset using Python. 
The dataset is a pandas DataFrame named 'df'. Use the Python REPL tool to execute code and answer questions.

Follow this exact format for your responses:
1. [THOUGHT]: Explain your reasoning step-by-step.
2. [ACTION]: Use 'python_repl_ast' and provide the Python code to execute in a single line (no backticks needed here).
3. Repeat steps 1-3 only if necessary to refine your answer.
4. [FINAL ANSWER]: Output ONLY the JSON string ```{{"response": "your answer"}}``` with no extra text.

**Rules**:
- Use the simplest, most direct Python code possible via the REPL tool.
- Do NOT repeat actions unnecessarily; analyze the observation before proceeding.
- NEVER include labels like 'Observation' or 'Thought' in the Python code itself—they are for structure only.
- Ensure the [FINAL ANSWER] is always a valid JSON string in triple backticks.

**Examples**:
1. Question: "How many passengers survived?"
   [THOUGHT]: I need to count survivors. The 'Survived' column uses 1 for survived, so I’ll sum it.
   [ACTION]: python_repl_ast
   [ACTION INPUT]: df['Survived'].value_count()[1]
   [THOUGHT]: The sum 342 is the number of survivors.
   [FINAL ANSWER]: ```{{"response": "342"}}```

2. Question: "How many passengers embarked at each port?"
   [THOUGHT]: I’ll count passengers per embarked port using value_counts() on 'Embarked'.
   [ACTION]: python_repl_ast
   [ACTION INPUT]: df['Embarked'].value_counts()
   [THOUGHT]: The counts are S: 644, C: 168, Q: 77.
   [FINAL ANSWER]: ```{{"response": "S: 644, C: 168, Q: 77"}}```
Now, answer this question in the same format:
Question: {input}
[THOUGHT]:
"""

tools = [PythonREPLTool()]
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True, 
    allow_dangerous_code=True,
    tools=tools,
    prefix=prefix
)


def query_titanic(query, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = agent.run({'input': query})
            print(f"Attempt {attempt + 1}: {response}")
            final_answer_match = re.search(r'\[FINAL ANSWER\]\s*(.*?)\s*(?=\[|$)', response, re.DOTALL)
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()
                json_match = re.search(r'```(\{.*?\})```', final_answer, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))["response"]
            print(f"Attempt {attempt + 1} failed to parse, retrying...")
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Error after {max_retries} attempts: {str(e)}"
    return "Error: Could not parse response after retries"

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