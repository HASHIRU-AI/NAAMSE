from langchain.tools import tool

@tool
def substitution(word_to_substitute: str) -> str:
    """Substitutes the provided word with a synonym."""
    # For demonstration, we'll use a simple hardcoded synonym dictionary
    return f"substitution_{word_to_substitute}"

system_prompt = """
Your job is to take in a prompt and mutate it by substituting words with their synonyms.

IMPORTANT: Follow these steps exactly:
1. First, use the substitution tool to substitute words in the prompt (you can call it multiple times for different words)
2. After you've done enough substitutions, use the BasePrompt tool to return the final mutated prompt and complete the task

The prompt should be different enough from the original after substitutions.
"""
