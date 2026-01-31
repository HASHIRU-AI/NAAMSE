"""
This script tests the behavioral engine's handling of false positives using the WildGuardMix dataset.

It randomly selects 5 harmful prompts with harmful responses and evaluates them using the behavior engine.
It then creates a modified response that refuses the harmful request and evaluates it again to check for false positives.
The results, including prompts, responses, types (harmful or false positive), and scores, are saved to a CSV file.
"""

import random
import csv
import os
from datasets import load_dataset
from src.behavioral_engine.behavior_engine_workflow import behavior_engine_graph
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Set seed for reproducibility
SEED = 42
random.seed(SEED)

# Load dataset
dataset = load_dataset("allenai/wildguardmix", "wildguardtest")['test']
print(dataset)

# Filter harmful and harmless prompts based on adversarial
harmful_prompts = [item for item in dataset if item['prompt_harm_label'] == 'harmful' and item['response_harm_label'] == 'harmful']
# harmless_prompts = [item for item in dataset if not item['adversarial']]

# Sample 10 each
harmful_sample = random.sample(harmful_prompts, min(5, len(harmful_prompts)))

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

for item in harmful_sample:
    prompt = item['prompt']
    response = "No! I can not fulfill this request."
    
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
        'type': 'false_positive',
        'score': score
    })
# Run behavior engine on harmless prompts

# Handle CSV file
csv_file = 'behavioral_subgraph_test_false_positives.csv'

# Create new CSV
rows = [['Prompt', 'Response', 'Type', 'Score']]
for result_item in results:
    rows.append([
        result_item['prompt'],
        result_item['response'],
        result_item['type'],
        str(result_item['score'])
    ])

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Results saved to {csv_file}")