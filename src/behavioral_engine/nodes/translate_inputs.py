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
        
        chunk_size = 4999  # Google Translate API limit
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
        
        translated_content = ""
        for chunk in chunks:
            for attempt in range(3):
                try:
                    translated_content += translator.translate(chunk)
                    break
                except Exception as e:
                    print(f"--- [Behavior Engine] Translation attempt {attempt + 1} failed: {e} ---")
                finally:
                    if attempt == 2:
                        translated_content += chunk  # Fallback to original chunk if translation fails
        translated_conversation_history.append({"role": role, "content": translated_content})
    
    
    
    print(f"--- [Behavior Engine] Translating Inputs to English ---")
    print(f"Translated Conversation History: {translated_conversation_history}")
    return {
        "translated_conversation_history": translated_conversation_history
    }