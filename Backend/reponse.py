from datetime import datetime

def respond_to_greeting(query):
    """This function responds to various greetings based on the time of day."""
    query = query.lower()
    # print("inside the greeting function: ",query)
    # Check for common greetings
    if any(greeting in query for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
        current_hour = datetime.now().hour
        # print("Current hour: ", current_hour)
        if current_hour < 12:
            return "Good morning! How can I help you today?"
        elif current_hour < 18:
            return "Good afternoon! What can I do for you?"
        else:
            return "Good evening! How's your day going?"

    return "Hello! How can I assist you?"