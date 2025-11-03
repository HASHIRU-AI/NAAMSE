import json
from datasets import load_dataset
from huggingface_hub import login
import os
from dotenv import load_dotenv
from pydriller import Repository
import tempfile
import glob
import re
from datasets import Dataset
import unicodedata
from typing_extensions import TypedDict, List
from langgraph.graph import StateGraph, START, END

class DataExtractorState(TypedDict):
    config: dict
    all_prompts: List[dict]

def clean_html(text):
    return re.sub(r'<[^>]+>', '', text)

def normalize_text(text):
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    # Strip and collapse whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Lowercase for case-insensitive deduplication
    text = text.lower()
    return text

# Load environment variables from .env file at project root
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(script_dir))
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

# Authenticate if token is available
token = os.getenv('HF_TOKEN')
if token and token != 'your_huggingface_token_here':
    login(token)
    print("Authenticated with Hugging Face using HF_TOKEN from .env")
else:
    print("HF_TOKEN not set or placeholder in .env. Gated datasets will fail to load. Update .env with your token.")

def load_config(state: DataExtractorState) -> dict:
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return {"config": config, "all_prompts": []}

def process_datasets(state: DataExtractorState) -> dict:
    config = state["config"]
    all_prompts = state["all_prompts"]

    for info in config["datasets"]:
        dataset_name = info["name"]
        source = dataset_name
        try:
            print(f"\nLoading {dataset_name}...")
            # Load dataset
            config_name = info.get('config')
            split_name = info.get('split')
            if config_name:
                ds = load_dataset(dataset_name, config_name)
                if split_name:
                    df = ds[split_name]
                else:
                    split_name = list(ds.keys())[0]
                    df = ds[split_name]
            else:
                if split_name:
                    ds = load_dataset(dataset_name, split=split_name)
                    df = ds
                else:
                    ds = load_dataset(dataset_name)
                    split_name = list(ds.keys())[0]
                    df = ds[split_name]
            
            print(f"Columns: {df.column_names}")
            
            # Get harmful field
            harmful_field = info['harmful_field']
            if isinstance(harmful_field, list):
                harmful_field = harmful_field[0]  # take first for sample
            
            condition = info['harmful_condition']
            
            # Filter for harmful
            if "all" in condition.lower() or condition == "always harmful":
                harmful_df = df
            elif "==" in condition:
                parts = condition.split(" == ")
                field = parts[0].strip()
                val_str = parts[1].strip().strip("'\"")
                if val_str.isdigit():
                    val = int(val_str)
                elif val_str == "true":
                    val = True
                elif val_str == "false":
                    val = False
                else:
                    val = val_str
                harmful_df = df.filter(lambda x: x[field] == val)
            elif "in" in condition:
                # e.g., data_type in ['vanilla_harmful', 'adversarial_harmful']
                field = condition.split(" in ")[0].strip()
                vals_str = condition.split(" in ")[1].strip()
                vals = eval(vals_str)  # list
                harmful_df = df.filter(lambda x: x[field] in vals)
            else:
                print(f"Unknown condition: {condition}")
                continue
            
            print(f"Total rows: {len(df)}")
            print(f"Harmful rows: {len(harmful_df)}")
            if len(harmful_df) > 0 and harmful_field in harmful_df.column_names:
                sample = harmful_df[0][harmful_field]
                if len(sample) > 200:
                    print(f"Sample harmful prompt: {sample[:200]}...")
                else:
                    print(f"Sample harmful prompt: {sample}")
                for prompt in harmful_df[harmful_field]:
                    all_prompts.append({"source": source, "messages": [{"role": "user", "content": prompt}]})
            else:
                print("No harmful prompts found or field not present.")
                
        except Exception as e:
            print(f"Error with {dataset_name}: {e}")

    return {"all_prompts": all_prompts}

def process_github(state: DataExtractorState) -> dict:
    config = state["config"]
    all_prompts = state["all_prompts"]

    for info in config.get("github", []):
        name = info["name"]
        source = name
        try:
            print(f"\nLoading {name}...")
            url = info["url"]
            git_url = url + '.git' if 'gist' in url else url + '.git'
            with tempfile.TemporaryDirectory() as temp_dir:
                os.system(f'git clone {git_url} {temp_dir} --depth 1')
                repo = Repository(temp_dir)
                if 'file_path' not in info:
                    info['file_path'] = 'README.md'
                    info['format'] = 'text'
                file_path = info['file_path']
                format_type = info.get('format', 'text')
                if name == "microsoft/BIPIA" and format_type == "json":
                    # Special handling for BIPIA benchmark JSON files
                    pattern = os.path.join(temp_dir, file_path)
                    files = glob.glob(pattern)
                    print(f"Found {len(files)} json files in {pattern}")
                    all_prompts_for_repo = []
                    for file in files:
                        with open(file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            for category, prompts in data.items():
                                if isinstance(prompts, list):
                                    for prompt in prompts:
                                        all_prompts_for_repo.append({"source": source, "messages": [{"role": "user", "content": prompt}]})
                    all_prompts.extend(all_prompts_for_repo)
                    print(f"Extracted {len(all_prompts_for_repo)} prompts from BIPIA")
                    continue
                elif format_type == "jsonl":
                    if file_path.endswith('.jsonl'):
                        # Single JSONL file
                        file_full = os.path.join(temp_dir, file_path)
                        if os.path.exists(file_full):
                            ds = load_dataset('json', data_files=file_full)
                            df = ds[list(ds.keys())[0]]
                        else:
                            print(f"JSONL file not found: {file_full}")
                            continue
                    else:
                        # Directory with JSONL files
                        pattern = os.path.join(temp_dir, file_path, '*.jsonl')
                        files = glob.glob(pattern)
                        print(f"Found {len(files)} jsonl files in {pattern}")
                        if files:
                            ds = load_dataset('json', data_files=files)
                            df = ds[list(ds.keys())[0]]
                        else:
                            print("No jsonl files found")
                            continue
                elif format_type == "csv":
                    if file_path.endswith('.csv'):
                        # Single CSV file
                        file_full = os.path.join(temp_dir, file_path)
                        if os.path.exists(file_full):
                            ds = load_dataset('csv', data_files=file_full)
                            df = ds[list(ds.keys())[0]]
                        else:
                            print(f"CSV file not found: {file_full}")
                            continue
                    else:
                        # Directory with CSV files
                        pattern = os.path.join(temp_dir, file_path, '*.csv')
                        files = glob.glob(pattern)
                        if files:
                            ds = load_dataset('csv', data_files=files)
                            df = ds[list(ds.keys())[0]]
                        else:
                            print("No csv files found")
                            continue
                else:  # text
                    if '*' in file_path:
                        pattern = os.path.join(temp_dir, file_path)
                        files = glob.glob(pattern)
                        content = ''
                        for f in files:
                            with open(f, 'r', encoding='utf-8') as ff:
                                content += ff.read() + '\n'
                    else:
                        file_full = os.path.join(temp_dir, file_path)
                        if os.path.exists(file_full):
                            with open(file_full, 'r', encoding='utf-8') as f:
                                content = f.read()
                        else:
                            content = "File not found"
                    if info.get("type") == "single-level":
                        code_blocks = re.findall(r'```(.*?)```', content, re.DOTALL)
                        if code_blocks:
                            prompts = [clean_html(block.strip()) for block in code_blocks if block.strip()]
                        else:
                            # Try to extract from <details> tags
                            details = re.findall(r'<details>(.*?)</details>', content, re.DOTALL | re.IGNORECASE)
                            prompts = []
                            for detail in details:
                                # Find the content after </summary>
                                summary_end = re.search(r'</summary>', detail, re.IGNORECASE)
                                if summary_end:
                                    prompt = detail[summary_end.end():].strip()
                                    prompt = clean_html(prompt).strip()
                                    if prompt:
                                        prompts.append(prompt)
                                else:
                                    # If no summary, take the whole detail
                                    prompt = clean_html(detail).strip()
                                    if prompt:
                                        prompts.append(prompt)
                        df = Dataset.from_dict({info['harmful_field']: prompts})
                    else:
                        df = Dataset.from_dict({info['harmful_field']: [content]})
                # filter
                harmful_field = info['harmful_field']
                condition = info['harmful_condition']
                if "all" in condition.lower() or condition == "always harmful":
                    harmful_df = df
                elif "==" in condition:
                    parts = condition.split(" == ")
                    field = parts[0].strip()
                    val_str = parts[1].strip().strip("'\"")
                    if val_str.isdigit():
                        val = int(val_str)
                    elif val_str == "true":
                        val = True
                    elif val_str == "false":
                        val = False
                    else:
                        val = val_str
                    harmful_df = df.filter(lambda x: x[field] == val)
                else:
                    print(f"Unknown condition: {condition}")
                    continue
                print(f"Total rows: {len(df)}")
                print(f"Harmful rows: {len(harmful_df)}")
                if len(harmful_df) > 0 and harmful_field in harmful_df.column_names:
                    sample = harmful_df[0][harmful_field]
                    if isinstance(sample, list):
                        sample = sample[0] if sample else ""
                    if len(sample) > 200:
                        print(f"Sample harmful prompt: {sample[:200]}...")
                    else:
                        print(f"Sample harmful prompt: {sample}")
                    if harmful_field == "decomposed_steps":
                        for row in harmful_df:
                            messages = [{"role": "user", "content": step} for step in row[harmful_field]]
                            all_prompts.append({"source": source, "messages": messages})
                    else:
                        for prompt in harmful_df[harmful_field]:
                            all_prompts.append({"source": source, "messages": [{"role": "user", "content": prompt}]})
                else:
                    print("No harmful prompts found or field not present.")
        except Exception as e:
            print(f"Error with {name}: {e}")

    return {"all_prompts": all_prompts}

def deduplicate_and_save(state: DataExtractorState) -> dict:
    all_prompts = state["all_prompts"]

    # Deduplicate and save corpus
    unique_prompts = {}
    for item in all_prompts:
        messages = item['messages']
        # Normalize the content of each message
        normalized_messages = []
        for msg in messages:
            normalized_msg = msg.copy()
            normalized_msg['content'] = normalize_text(msg['content'])
            normalized_messages.append(normalized_msg)
        key = tuple(tuple(msg.items()) for msg in normalized_messages)
        if key not in unique_prompts:
            unique_prompts[key] = item
    print(f"\nTotal prompts collected: {len(all_prompts)}")
    print(f"Unique prompts after deduplication: {len(unique_prompts)}")
    
    # Save to file
    with open('jailbreak_corpus.jsonl', 'w', encoding='utf-8') as f:
        for item in unique_prompts.values():
            f.write(json.dumps(item) + '\n')
    print("Corpus saved to jailbreak_corpus.jsonl")

    return {}

# Build the graph
graph_builder = StateGraph(DataExtractorState)

# Add nodes
graph_builder.add_node("load_config", load_config)
graph_builder.add_node("process_datasets", process_datasets)
graph_builder.add_node("process_github", process_github)
graph_builder.add_node("deduplicate_and_save", deduplicate_and_save)

# Add edges
graph_builder.add_edge(START, "load_config")
graph_builder.add_edge("load_config", "process_datasets")
graph_builder.add_edge("process_datasets", "process_github")
graph_builder.add_edge("process_github", "deduplicate_and_save")
graph_builder.add_edge("deduplicate_and_save", END)

# Compile the graph
graph = graph_builder.compile()

if __name__ == "__main__":
    # Run the graph
    initial_state = {}
    final_state = graph.invoke(initial_state)
    print("Data extraction completed.")