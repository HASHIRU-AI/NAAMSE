import ftfy
import unicodedata
    
import base64
import urllib.parse
import html
import re
def decode_inputs(state):
    """
    This function takes in the conversation history. It fixes the encoding errors and then decodes them. 
    """
    # TODO: Implement decoding logic
    decoded_conversation_history = []
    for message in state["conversation_history"]:
        role = message["role"]
        content = message["content"]
        decoded_content= content
        for _ in range(3):
            encoding_type, new_content = detect_and_decode(decoded_content)
            if encoding_type == 'none' or new_content == decoded_content:
                break
            decoded_content = new_content
        
        fixed_content = ftfy.fix_text(decoded_content)
        normalized_content = unicodedata.normalize("NFKC", fixed_content)
        decoded_conversation_history.append({"role": role, "content": normalized_content})

    print(f"--- [Behavior Engine] Decoding Inputs ---")
    print(f"Decoded Conversation History: {decoded_conversation_history}")
    return {
        "decoded_conversation_history": decoded_conversation_history
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