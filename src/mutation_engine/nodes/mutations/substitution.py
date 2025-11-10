from langchain.tools import tool

@tool
def substitution(word_to_substitute: str) -> str:
    """Substitutes the provided word with a synonym."""
    # For demonstration, we'll use a simple hardcoded synonym dictionary
    return f"substitution_{word_to_substitute}"

system_prompt = """
Your job is to take in a prompt and mutate it by substituting words with their synonyms.
You select words in the prompt to substitute.
You will be provided with a tool that performs the substitution for a given word.
Use the tool as needed to perform substitutions in the prompt until the prompt is different enough from the original.
You must use the tool to perform substitutions.
"""