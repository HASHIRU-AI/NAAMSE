import ftfy
import unicodedata
    
import base64
import urllib.parse
import html
import re
from typing import Union, List, Dict, Any
def extract_text_from_content(content: Union[str, List[Dict[str, Any]], Dict[str, Any]]) -> str:
    """
    Extracts text from content block. Handles:
    1. String (Standard OpenAI)
    2. Dictionary (A2A / Rich content)
    3. List (Multimodal / Complex)
    """
    # Case 1: Simple String
    if isinstance(content, str):
        return content

    # Case 2: Dictionary (A2A / Rich content)
    if isinstance(content, dict):
        # We just grab the text field and ignore the signature/extras
        return str(content.get("text", ""))

    # Case 3: List (The standard OpenAI "multimodal" format)
    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    chunks.append(str(part["text"]))
                elif "text" in part and isinstance(part["text"], str):
                    chunks.append(str(part["text"]))
            elif isinstance(part, str):
                chunks.append(part)
        return " ".join(chunks)

    # Case 4: Invalid input
    raise ValueError(f"Content must be a string, dict, or list. Got: {type(content)}")
    
def decode_inputs(state):
    """
    This function takes in the conversation history. It fixes the encoding errors and then decodes them. 
    """
    decoded_conversation_history = []
    for message in state["conversation_history"]["messages"]:
        role = message["role"]
        content = message["content"]
        text = extract_text_from_content(content)
        
        decoded_text = text
        for _ in range(3):
            encoding_type, new_content = detect_and_decode(decoded_text)
            if encoding_type == 'none' or new_content == decoded_text:
                break
            decoded_text = new_content
    
        fixed_text = ftfy.fix_text(decoded_text)
        normalized_content = unicodedata.normalize("NFKC", fixed_text)
        decoded_conversation_history.append({"role": role, "content": normalized_content})

    print(f"--- [Behavior Engine] Decoding Inputs ---")
    print(f"Decoded Conversation History: {decoded_conversation_history}")
    return {
        "decoded_conversation_history": {"messages": decoded_conversation_history}
    }
    


def detect_and_decode(input_string):
    """Attempt to detect and decode common encodings"""
    
    # Try Base64
    if re.match(r'^[A-Za-z0-9+/]+={0,2}$', input_string):
        try:
            decoded = base64.b64decode(input_string).decode('utf-8')
            return ('base64', decoded)
        except:
            pass
    
    # Try URL encoding (if contains %)
    if '%' in input_string:
        try:
            decoded = urllib.parse.unquote(input_string)
            if decoded != input_string:
                return ('url', decoded)
        except:
            pass
    
    # Try HTML entities
    if '&' in input_string and ';' in input_string:
        decoded = html.unescape(input_string)
        if decoded != input_string:
            return ('html', decoded)
    
    # Try hex
    if re.match(r'^[0-9a-fA-F]+$', input_string) and len(input_string) % 2 == 0:
        try:
            decoded = bytes.fromhex(input_string).decode('utf-8')
            return ('hex', decoded)
        except:
            pass
    
    return ('none', input_string)
