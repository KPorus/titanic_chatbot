import os
from Backend.reponse import respond_to_greeting
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from Backend.load_datasets import load_Dataset
from Backend.helper_fuc import (
    generate_histogram,
    query_correlation,
    calculate_survival_rate_by_class,
    get_class_distribution,
    get_embarked_counts,
    get_num_died,
    get_num_survived,
    get_overall_survival_rate,
    survival_rate_by_embarked,
    survival_rate_by_gender,
    get_most_common_fare
)


# Load environment variables and dataset
load_dotenv()
hugging_face_api_key = os.getenv("hugging_face_api_key")
df = load_Dataset()
print("Dataset loaded successfully.")

# --- Main query function ---
def query_titanic_dataset(query):
    """
    Delegates the query to the correct backend logic based on the user's question.
    First checks for greetings, then for Titanic-specific queries.
    """
    query = query.lower()

    if "correlation" in query:
        return query_correlation()

    if "percentage" in query and "male" in query:
        male_percentage = (df['Sex'] == 'male').mean() * 100
        return f"{male_percentage:.2f}% of passengers were male."
    
    if "percentage" in query and "female" in query:
        female_percentage = (df['Sex'] == 'female').mean() * 100
        return f"{female_percentage:.2f}% of passengers were female."

    if "histogram" in query:
        if "age" in query:
            return generate_histogram("Age", "Histogram of Passenger Ages", "Age")
        elif "fare" in query:
            return generate_histogram("Fare", "Histogram of Ticket Fares", "Fare")
        elif "sex" in query:
            return generate_histogram("Sex", "Histogram of Gender", "Sex")
        elif "embarked" in query:
            return generate_histogram("Embarked", "Histogram of Embarkation", "Embarked")
        elif "survived" in query or "survival rate" in query:
            return generate_histogram("Survived", "Histogram of Survival Rate", "Survival Rate")

    if "average" in query and "ticket fare" in query:
        avg_fare = df['Fare'].mean()
        return f"The average ticket fare was ${avg_fare:.2f}."
    
    if "most common fare" in query or "common ticket fare" in query:
        return get_most_common_fare()

    if "embarked" in query:
        return get_embarked_counts()

    if "survival rate" in query and "class" in query:
        return calculate_survival_rate_by_class()
    
    if "most of" in query and "class" in query or "percentage of" in query and "passengers" in query:
        return get_class_distribution()

    if "overall survival rate" in query or ("survival rate" in query and "overall" in query):
        return get_overall_survival_rate()

    if "survive" in query and "how many" in query:
        return get_num_survived()

    if "died" in query and "how many" in query:
        return get_num_died()

    if "survival rate" in query and "gender" in query:
        return survival_rate_by_gender()

    if "survival rate" in query and "embarked" in query:
        return survival_rate_by_embarked()

    return "I didn't quite understand the question. Can you rephrase it?"

# --- Set up LangChain agent ---
llm = HuggingFaceEndpoint(
    repo_id="gpt2",
    temperature=0.5,
    huggingfacehub_api_token=hugging_face_api_key
)

tools = [
    Tool(
        name="Titanic Dataset Query",
        func=query_titanic_dataset,
        description="Answer questions about the Titanic dataset including statistics and visualizations."
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Final query function
def query_titanic(question):
    """
    Uses the custom query function for Titanic data.
    Since our queries are well-defined, we bypass the agent chain for these.
    """
    lowered = question.lower()
    if any(kw in lowered for kw in ["titanic", "male", "age", "fare", "embarked", "sex", "survival rate", "survived", "died", "class distribution", "correlation"]):
        return query_titanic_dataset(question)
    elif any(kw in lowered for kw in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
        return respond_to_greeting(question)
    else:
        print(f"Sending query via agent: {question}")
        try:
            response = agent.invoke([question], handle_parsing_errors=True)
            print(f"Response from agent: {response}")
            return response
        except Exception as e:
            print(f"Error during agent processing: {e}")
            return str(e)
