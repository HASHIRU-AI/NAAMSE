import random
from langchain.tools import tool

from src.mutation_engine.mutation_workflow_state import BasePrompt


def unicode_wrap(prompt: str) -> str:
    """Substitutes characters in the prompt with similar-looking Unicode homoglyphs."""
    homoglyphs = {
        'a': ['A', 'À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'à', 'á', 'â', 'ã', 'ä', 'å', 'ɑ', 'Α', 'α', 'а', 'Ꭺ', 'Ａ', 'ａ'],
        'b': ['B', 'ß', 'ʙ', 'Β', 'β', 'В', 'Ь', 'Ᏼ', 'ᛒ', 'Ｂ', 'ｂ'],
        'c': ['C', 'ϲ', 'Ϲ', 'С', 'с', 'Ꮯ', 'Ⅽ', 'ⅽ', '𐐠', 'Ｃ', 'ｃ'],
        'd': ['D', 'Ď', 'ď', 'Đ', 'đ', 'ԁ', 'ժ', 'Ꭰ', 'ḍ', 'Ⅾ', 'ⅾ', 'Ｄ', 'ｄ'],
        'e': ['E', 'È', 'É', 'Ê', 'Ë', 'é', 'ê', 'ë', 'Ē', 'ē', 'Ĕ', 'ĕ', 'Ė', 'ė', 'Ę', 'Ě', 'ě', 'Ε', 'Е', 'е', 'Ꭼ', 'Ｅ', 'ｅ'],
        'f': ['F', 'Ϝ', 'Ｆ', 'ｆ'],
        'g': ['G', 'ɡ', 'ɢ', 'Ԍ', 'ն', 'Ꮐ', 'Ｇ', 'ｇ'],
        'h': ['H', 'ʜ', 'Η', 'Н', 'һ', 'Ꮋ', 'Ｈ', 'ｈ'],
        'i': ['I', 'l', 'ɩ', 'Ι', 'І', 'і', 'ا', 'Ꭵ', 'ᛁ', 'Ⅰ', 'ⅰ', 'Ｉ', 'ｉ'],
        'j': ['J', 'ϳ', 'Ј', 'ј', 'յ', 'Ꭻ', 'Ｊ', 'ｊ'],
        'k': ['K', 'Κ', 'κ', 'К', 'Ꮶ', 'ᛕ', 'K', 'Ｋ', 'ｋ'],
        'l': ['L', 'ʟ', 'ι', 'ا', 'Ꮮ', 'Ⅼ', 'ⅼ', 'Ｌ', 'ｌ'],
        'm': ['M', 'Μ', 'Ϻ', 'М', 'Ꮇ', 'ᛖ', 'Ⅿ', 'ⅿ', 'Ｍ', 'ｍ'],
        'n': ['N', 'ɴ', 'Ν', 'Ｎ', 'ｎ'],
        'o': ['O', 'ο', 'О', 'о', 'Օ', 'Ｏ', 'ｏ'],
        'p': ['P', 'Ρ', 'ρ', 'Р', 'р', 'Ꮲ', 'Ｐ', 'ｐ'],
        'q': ['Q', 'Ⴍ', 'Ⴓ', 'Ｑ', 'ｑ'],
        'r': ['R', 'Ŕ', 'Ř', 'Ṛ', 'Ŗ', 'ℛ', 'ℜ', 'ℝ', '𝐑', '𝑅', '𝒜', 'ℛ', 'ℜ', 'ℝ', '𝐑', '𝑅', '𝒓', '𝓡', '𝔯', '𝕣', '𝖗', '𝗋', '𝘳', '𝙧', '𝚁', '𝛒', '𝜚', '𝝔', '𝞌', 'ʀ', 'Ի', 'Ꮢ', 'ᚱ', 'Ｒ', 'ｒ'],
        's': ['S', 'Ѕ', 'ѕ', 'Տ', 'Ⴝ', 'Ꮪ', 'Ｓ', 'ｓ'],
        't': ['T', 'Τ', 'τ', 'Т', 'Ꭲ', 'Ｔ', 'ｔ'],
        'u': ['U', 'μ', 'υ', 'Ա', 'Ս', '⋃', 'Ｕ', 'ｕ'],
        'v': ['V', 'ν', 'Ѵ', 'ѵ', 'Ꮩ', 'Ⅴ', 'ⅴ', 'Ｖ', 'ｖ'],
        'w': ['W', 'ѡ', 'Ꮃ', 'Ｗ', 'ｗ'],
        'x': ['X', 'Χ', 'χ', 'Х', 'х', 'Ⅹ', 'ⅹ', 'Ｘ', 'ｘ'],
        'y': ['Y', 'ʏ', 'Υ', 'γ', 'у', 'Ү', 'Ｙ', 'ｙ'],
        'z': ['Z', 'Ζ', 'Ꮓ', 'Ｚ', 'ｚ'],
        '0': ['O', 'o', 'Ο', 'ο', 'О', 'о', 'Օ', 'Ｏ', 'ｏ'],
        '1': ['I', 'l', 'ا', '１'],
        '2': ['２'],
        '3': ['３'],
        '4': ['４'],
        '5': ['５'],
        '6': ['６'],
        '7': ['７'],
        '8': ['８'],
        '9': ['９'],
    }
    mutated = []
    for char in prompt:
        lower_char = char.lower()
        if lower_char in homoglyphs and random.random() < 0.4:  # 40% chance to replace
            # Choose a homoglyph that's different from the original
            candidates = [h for h in homoglyphs[lower_char] if h != char]
            if candidates:
                homoglyph = random.choice(candidates)
                mutated.append(homoglyph)
            else:
                mutated.append(char)
        else:
            mutated.append(char)
    return ''.join(mutated)


@tool
def unicode_mutation(prompt: BasePrompt) -> BasePrompt:
    """Mutates the prompt by replacing characters with similar-looking Unicode homoglyphs."""
    final_prompts = []
    for p in prompt["prompt"]:
        mutated_prompt = unicode_wrap(p)
        final_prompts.append(mutated_prompt)
    return BasePrompt(prompt=final_prompts)


system_prompt = """
Your job is to take in a prompt and mutate it by replacing characters with similar-looking Unicode homoglyphs.

If there are other Unicode characters available beyond the standard homoglyphs, introduce more randomness in the replacements to create more varied and unpredictable mutations.

IMPORTANT: Follow these steps exactly:
1. First, invoke the unicode_mutation tool to mutate the prompt
2. The tool will return a mutated prompt
3. Then, use the BasePrompt tool to return the final mutated prompt and complete the task

Do NOT call unicode_mutation multiple times. Call it once, get the result, then use BasePrompt to return it.
"""