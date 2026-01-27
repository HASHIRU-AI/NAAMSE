"""
This script tests the behavioral subgraph by running it on a sample of harmful and harmless prompts
from the WildGuardMix dataset. It evaluates the final scores assigned by the behavior engine and saves
the results to a JSON file for further analysis.

It randomly selects 10 harmful prompts with harmful responses and 10 harmless prompts with harmless responses.
It then evaluates them using the behavior engine and saves the results, including prompts, responses, types,
and scores, to a JSON file.

It also tests harmful prompts with harmless responses and harmless prompts with harmful responses to check for false positives and negatives.
and includes them in the output.
"""

import random
import csv
import os
from datasets import load_dataset
from src.behavioral_engine.behavior_engine_workflow import behavior_engine_graph
from dotenv import load_dotenv
import json

load_dotenv()  # Load environment variables from .env file

# Set seed for reproducibility
SEED = 42
random.seed(SEED)

# Load dataset
dataset = load_dataset("allenai/wildguardmix", "wildguardtest")['test']
print(dataset)

# Filter harmful and harmless prompts based on adversarial
harmful_prompts = [item for item in dataset if item['prompt_harm_label'] == 'harmful' and item['response_harm_label'] == 'harmful']
harmless_prompts = [item for item in dataset if item['prompt_harm_label'] == 'unharmful' and item['response_harm_label'] == 'unharmful']
harmless_prompt_harmful_response = [item for item in dataset if item['prompt_harm_label'] == 'unharmful' and item['response_harm_label'] == 'harmful']
harmful_prompt_harmless_response = [item for item in dataset if item['prompt_harm_label'] == 'harmful' and item['response_harm_label'] == 'unharmful']

# Sample 5 each
harmful_sample = random.sample(harmful_prompts, min(10, len(harmful_prompts)))
harmless_sample = random.sample(harmless_prompts, min(10, len(harmless_prompts)))
harmless_prompt_harmful_response_sample = random.sample(harmless_prompt_harmful_response, min(10, len(harmless_prompt_harmful_response)))
harmful_prompt_harmless_response_sample = random.sample(harmful_prompt_harmless_response, min(10, len(harmful_prompt_harmless_response)))

# Run behavior engine on harmful prompts
results = []
for item in harmful_sample:
    prompt = item['prompt']
    response = item['response']
    
    conv = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }
    result = behavior_engine_graph.invoke({"conversation_history": conv})
    score = result['final_score']
    
    results.append({
        'prompt': prompt,
        'response': response,
        'type': 'harmful',
        'score': score
    })

for item in harmful_prompt_harmless_response_sample:
    prompt = item['prompt']
    response = item['response']
    
    conv = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }
    result = behavior_engine_graph.invoke({"conversation_history": conv})
    score = result['final_score']
    
    results.append({
        'prompt': prompt,
        'response': response,
        'type': 'harmful_prompt_harmless_response',
        'score': score
    })

# Run behavior engine on harmless prompts
for item in harmless_sample:
    prompt = item['prompt']
    response = item['response']
    
    conv = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }
    result = behavior_engine_graph.invoke({"conversation_history": conv, "is_score_flipped": True})
    score = result['final_score']
    
    results.append({
        'prompt': prompt,
        'response': response,
        'type': 'harmless',
        'score': score
    })
    
# Run behavior engine on harmless prompts with harmful responses (false positives)
for item in harmless_prompt_harmful_response_sample:
    prompt = item['prompt']
    response = item['response']
    
    conv = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }
    result = behavior_engine_graph.invoke({"conversation_history": conv, "is_score_flipped": True})
    score = result['final_score']
    
    results.append({
        'prompt': prompt,
        'response': response,
        'type': 'harmless_prompt_harmful_response',
        'score': score
    })


json_file = 'behavioral_subgraph_test_scoring.json'

# Create list of result objects
output_data = []
for result_item in results:
    output_data.append({
        'prompt': result_item['prompt'],
        'response': result_item['response'],
        'type': result_item['type'],
        'score': result_item['score']
    })

# Write to JSON file
with open(json_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Results saved to {json_file}")