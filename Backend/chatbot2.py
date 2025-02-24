import os
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
hugging_face_api_key = os.getenv("hugging_face_api_key")

# Set up LangChain agent using Hugging Face model (gpt2 or any relevant model)
llm = HuggingFaceEndpoint(
    repo_id="gpt2",  # You can use a more suitable model like GPT-3 or GPT-4 if desired
    temperature=0.5,
    huggingfacehub_api_token=hugging_face_api_key
)

# Define a tool that handles Titanic-related queries (this can be generic for all queries)
tools = [
    Tool(
        name="Titanic Query Tool",
        func=lambda query: f"Agent's answer for Titanic query: {query}",  # Processing query with model
        description="A tool that processes Titanic dataset-related questions."
    )
]

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Unified query function that sends Titanic-related questions to the agent
def query_titanic(question):
    """
    Uses the agent to answer Titanic dataset-related questions.
    """
    # Check if the question is related to Titanic dataset (optional)
    relevant_keywords = ['fare', 'class', 'survival', 'age', 'embarked', 'sex', 'ticket']
    if any(keyword in question.lower() for keyword in relevant_keywords):
        print(f"Sending query via agent: {question}")
        try:
            # Process the query through the agent
            response = agent.invoke([question], handle_parsing_errors=True)
            print(f"Response from agent: {response}")
            return response
        except Exception as e:
            print(f"Error during agent processing: {e}")
            return str(e)
    else:
        return "Sorry, I can only answer questions related to the Titanic dataset."

# Example usage
print(query_titanic("What’s the most common ticket fare paid?"))
print(query_titanic("Compare survival rates between different classes."))
