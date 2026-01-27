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

# Handle CSV file
csv_file = 'behavioral_subgraph_test_scoring.csv'

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