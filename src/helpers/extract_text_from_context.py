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