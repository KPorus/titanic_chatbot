"""
This is the file of Hybrid apporch where i run the model with my predefine function 

"""

# import os
# from dotenv import load_dotenv
# from langchain.agents import create_pandas_dataframe_agent
# from langchain.agents import AgentType
# from langchain_huggingface import HuggingFaceEndpoint
# from Backend.load_datasets import load_Dataset
# from Backend.reponse import respond_to_greeting

# # Load environment variables and dataset
# load_dotenv()
# hugging_face_api_key = os.getenv("hugging_face_api_key")
# df = load_Dataset()
# print("Dataset loaded successfully.")

# # Use a more instruction-following model with adjusted generation parameters.
# llm = HuggingFaceEndpoint(
#     repo_id="tiiuae/falcon-7b-instruct",  # try a model known to follow instructions
#     temperature=0.1,
#     huggingfacehub_api_token=hugging_face_api_key,
#     max_new_tokens=256
# )

# # Create the pandas dataframe agent (note: custom prompt or handle_parsing_errors are no longer accepted)
# agent = create_pandas_dataframe_agent(
#     llm,
#     df,
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     allow_dangerous_code=True,handle_parsing_errors=True
# )

# def query_titanic(question):
#     """
#     Routes queries to the pandas dataframe agent or a greeting responder.
#     """
#     lowered = question.lower()
#     if any(kw in lowered for kw in [
#             "titanic", "male", "age", "fare", "embarked", "sex",
#             "survival rate", "survived", "died", "class distribution", "correlation"]):
#         try:
#             return agent.invoke([question])
#         except Exception as e:
#             print("Error during agent.invoke:", e)
#             return str(e)
#     elif any(kw in lowered for kw in [
#             'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
#         return respond_to_greeting(question)
#     else:
#         try:
#             return agent.invoke([question])
#         except Exception as e:
#             print("Error during agent.invoke:", e)
#             return str(e)

# # Example queries
# print(agent.invoke("How many rows are in the Titanic dataset?",handle_parsing_errors=True))
# print(agent.invoke("What is the average age of passengers?"))
# print(agent.invoke("How many passengers survived?"))



# import os
# from Backend.reponse import respond_to_greeting
# from langchain.agents import initialize_agent, Tool, AgentType
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from dotenv import load_dotenv
# from Backend.load_datasets import load_Dataset
# from Backend.helper_fuc import (
#     generate_histogram,
#     query_correlation,
#     calculate_survival_rate_by_class,
#     get_class_distribution,
#     get_embarked_counts,
#     get_num_died,
#     get_num_survived,
#     get_overall_survival_rate,
#     survival_rate_by_embarked,
#     survival_rate_by_gender,
#     get_most_common_fare
# )


# # Load environment variables and dataset
# load_dotenv()
# hugging_face_api_key = os.getenv("hugging_face_api_key")
# df = load_Dataset()
# print("Dataset loaded successfully.")

# # --- Main query function ---
# # def query_titanic_dataset(query):
# #     """
# #     Delegates the query to the correct backend logic based on the user's question.
# #     First checks for greetings, then for Titanic-specific queries.
# #     """
# #     query = query.lower()

# #     if "correlation" in query:
# #         return query_correlation()

# #     if "percentage" in query and "male" in query:
# #         male_percentage = (df['Sex'] == 'male').mean() * 100
# #         return f"{male_percentage:.2f}% of passengers were male."
    
# #     if "percentage" in query and "female" in query:
# #         female_percentage = (df['Sex'] == 'female').mean() * 100
# #         return f"{female_percentage:.2f}% of passengers were female."

# #     if "histogram" in query:
# #         if "age" in query:
# #             return generate_histogram("Age", "Histogram of Passenger Ages", "Age")
# #         elif "fare" in query:
# #             return generate_histogram("Fare", "Histogram of Ticket Fares", "Fare")
# #         elif "sex" in query:
# #             return generate_histogram("Sex", "Histogram of Gender", "Sex")
# #         elif "embarked" in query:
# #             return generate_histogram("Embarked", "Histogram of Embarkation", "Embarked")
# #         elif "survived" in query or "survival rate" in query:
# #             return generate_histogram("Survived", "Histogram of Survival Rate", "Survival Rate")

# #     if "average" in query and "ticket fare" in query:
# #         avg_fare = df['Fare'].mean()
# #         return f"The average ticket fare was ${avg_fare:.2f}."
    
# #     if "most common fare" in query or "common ticket fare" in query:
# #         return get_most_common_fare()

# #     if "embarked" in query:
# #         return get_embarked_counts()

# #     if "survival rate" in query and "class" in query:
# #         return calculate_survival_rate_by_class()
    
# #     if "most of" in query and "class" in query or "percentage of" in query and "passengers" in query:
# #         return get_class_distribution()

# #     if "overall survival rate" in query or ("survival rate" in query and "overall" in query):
# #         return get_overall_survival_rate()

# #     if "survive" in query and "how many" in query:
# #         return get_num_survived()

# #     if "died" in query and "how many" in query:
# #         return get_num_died()

# #     if "survival rate" in query and "gender" in query:
# #         return survival_rate_by_gender()

# #     if "survival rate" in query and "embarked" in query:
# #         return survival_rate_by_embarked()

# #     return "I didn't quite understand the question. Can you rephrase it?"

# # --- Set up LangChain agent ---
# llm = HuggingFaceEndpoint(
#     repo_id="EleutherAI/gpt-neo-2.7B",
#     # repo_id="gpt2",
#     temperature=0.5,
#     huggingfacehub_api_token=hugging_face_api_key,
#     # max_new_tokens=64
# )

# # tools = [
# #     Tool(
# #         name="Titanic Dataset Query",
# #         func=query_titanic_dataset,
# #         description="Answer questions about the Titanic dataset including statistics and visualizations."
# #     )
# # ]

# # agent = initialize_agent(
# #     tools,
# #     llm,
# #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# #     verbose=True,
# #     handle_parsing_errors=True
# # )
# # Reduce the prompt size by including fewer rows from the dataframe (if supported)
# agent = create_pandas_dataframe_agent(
#    llm, 
#    df, 
#    verbose=True, 
#    agent_type=AgentType.OPENAI_FUNCTIONS,
#    allow_dangerous_code=True,
# )

# # Final query function
# def query_titanic(question):
#     """
#     Uses the pandas dataframe agent for Titanic data.
#     If the query is a greeting, it delegates to respond_to_greeting.
#     Otherwise, it sends the query to the agent.
#     """
#     lowered = question.lower()
#     if any(kw in lowered for kw in [
#             "titanic", "male", "age", "fare", "embarked", "sex",
#             "survival rate", "survived", "died", "class distribution", "correlation"]):
#         return agent.invoke(question)
#     elif any(kw in lowered for kw in [
#             'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
#         return respond_to_greeting(question)
#     else:
#         print(f"Sending query via agent: {question}")
#         try:
#             response = agent.invoke([question])
#             print(f"Response from agent: {response}")
#             return response
#         except Exception as e:
#             print(f"Error during agent processing: {e}")
#             return str(e)

# # Example queries
# # response = agent.invoke("How many rows are in the Titanic dataset?")
# # print(response)

# # response = agent.invoke("What is the average age of passengers?")
# # print(response)

# response = query_titanic("How many passengers survived?")
# print(response)