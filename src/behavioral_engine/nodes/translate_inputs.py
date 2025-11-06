from deep_translator import GoogleTranslator

def translate_inputs(state):
    """
    This function takes in the decoded conversation history and translates them to english.
    """
    translator = GoogleTranslator(source='auto', target='en')
    translated_conversation_history = []
    for message in state["decoded_conversation_history"]:
        role = message["role"]
        content = message["content"]
        translated_content = translator.translate(content)
        translated_conversation_history.append({"role": role, "content": translated_content})
    
    
    
    print(f"--- [Behavior Engine] Translating Inputs to English ---")
    print(f"Translated Conversation History: {translated_conversation_history}")
    return {
        "translated_conversation_history": translated_conversation_history
    }