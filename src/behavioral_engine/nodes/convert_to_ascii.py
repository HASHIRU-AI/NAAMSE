import unidecode

def convert_to_ascii(state):
    """
    This function takes the translated conversation history and converts it to ASCII.
    """
    ascii_conversation_history = []
    for message in state["translated_conversation_history"]["messages"]:
        role = message["role"]
        content = message["content"]
        ascii_content = unidecode.unidecode(content)
        ascii_conversation_history.append({"role": role, "content": ascii_content})
        
    
    print(f"--- [Behavior Engine] Converting to ASCII ---")
    print(f"Converted Conversation History: {ascii_conversation_history}")
    return {
        "sanitized_conversation_history": {"messages": ascii_conversation_history}
    }