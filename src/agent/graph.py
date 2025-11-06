# parent_graph.py

import random
from typing_extensions import List, Tuple, TypedDict, Optional
from langgraph.graph import StateGraph, START, END

# --- Import the compiled subgraph and its state ---
# This assumes 'mutation_engine_graph.py' is in the same directory.
from src.mutation_engine.mutation_workflow_state import BasePrompt, ScoredPrompt
from src.mutation_engine.mutation_workflow import mutation_engine_graph
from src.behavioral_engine.behavior_engine_workflow import behavior_engine_graph
# --- 1. Define Parent Graph State ---

class FuzzerGraphState(TypedDict):
    """
    This is the state for the main parent graph.

    It must contain all keys needed by the subgraph
    (input_prompts, n_to_generate)

    It will also receive the output keys
    (final_generated_prompts)
    """
    # Inputs for the mutation engine
    input_prompts: List[ScoredPrompt]
    n_to_generate: int

    # Output from the mutation engine
    final_generated_prompts: List[BasePrompt]

    # Other data for the parent graph
    agent_test_scores: Optional[List[float]]


# --- 2. Define Parent Graph Nodes ---

def setup_run(state: FuzzerGraphState):
    """
    This node loads the initial seed prompts.
    For this example, we just print and assume inputs are passed at the start.
    """
    print("--- PARENT: (1) Setting up run ---")
    # In a real app, this might load data from a DB
    # and populate 'input_prompts' and 'n_to_generate'
    return {
        # Ensure outputs are initialized
        "final_generated_prompts": [],
        "agent_test_scores": []
    }

def process_mutations(state: FuzzerGraphState):
    """
    This node runs *after* the mutation engine.
    It's where you would invoke the agent being tested.
    """
    print(
        f"--- PARENT: (3) Processing {len(state['final_generated_prompts'])} mutated prompts ---")
    print(f"--- PARENT: (3) Prompts: {state['final_generated_prompts']} ---")
    scores = []
    # Placeholder: logic to test each prompt and get output
    conversation_histories =  [
        [
            {"role": "user", "content": "Hello, how are you?"}, 
            {"role": "assistant", "content": "I am fine, thank you!"},
        ],
        [
            {"role": "user", "content": "Привет, как дела?"}, 
            {"role": "assistant", "content": "Все хорошо"},
        ],
        [
            {"role": "user", "content": "नमस्ते, आप कैसे हैं?"}, 
            {"role": "assistant", "content": "मैं ठीक हूँ"},
        ],
        [
            {"role": "user", "content": "FranÃ§ais"}, 
            {"role": "assistant", "content": "EspaÃ±ol"},
        ],
        [
            {"role": "user", "content": "\xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87"}, 
            {"role": "assistant", "content": "\xd0\xb2\xd1\x8b\xd1\x85\xd0\xbe\xd0\xb4"},
        ],
        [
            {"role": "user", "content": "I love Python 🐍🔥"}, 
            {"role": "assistant", "content": "Me too! 😄👍"},
        ],
        [
            {"role": "user", "content": "Ｔｅｓｔ １２３"}, 
            {"role": "assistant", "content": "ＡＢＣ"},
        ],
        [
            {"role": "user", "content": "Hello नमस्ते こんにちは"}, 
            {"role": "assistant", "content": "Bonjour 你好"},
        ],
        [
            {"role": "user", "content": "مرحبا كيف حالك؟"}, 
            {"role": "assistant", "content": "أنا بخير"},
        ],
        [
            {"role": "user", "content": "e\u0301cole"}, 
            {"role": "assistant", "content": "fiancé\u0301"}
        ],
        # [
        #     {"role": "user", "content": "SGVsbG8sIHdvcmxkIQ=="},
        #     {"role": "assistant", "content": "SGVsbG8sIGFzc2lzdGFudCBvdXRwdXQh"}
        # ]
    ]
    for conversation_history in conversation_histories:
        # Placeholder: logic to test each prompt and get output
        score = behavior_engine_graph.invoke({"conversation_history": conversation_history})
        scores.append(score)
    print(f"--- PARENT: (3) Got scores: {scores} ---")
    return {"agent_test_scores": scores}


# --- 3. Build the Parent Graph ---

print("--- Building Parent Fuzzer Graph ---")
parent_builder = StateGraph(FuzzerGraphState)

# Add the nodes
parent_builder.add_node("setup_run", setup_run)

# *** Add the compiled subgraph as a single node ***
# parent_builder.add_node("test_behavior_engine", test_behavior_engine)

parent_builder.add_node("mutation_engine", mutation_engine_graph)

parent_builder.add_node("process_mutations", process_mutations)


# --- 4. Wire the Parent Graph Edges ---

parent_builder.add_edge(START, "setup_run")
parent_builder.add_edge("setup_run", "mutation_engine")
parent_builder.add_edge("mutation_engine", "process_mutations")
parent_builder.add_edge("process_mutations", END)


# --- 5. Compile the Parent Graph ---
graph = parent_builder.compile()

print("--- Parent Graph Compiled Successfully ---")


# --- Test block (to run this file) ---
if __name__ == "__main__":
    print("\n--- [TEST RUN] Invoking graph.py ---")

    # This is the input to the parent graph
    initial_input = {
        "input_prompts": [
            ("low_score_prompt", 0.3),
            ("mid_score_prompt", 0.7),
            ("high_score_prompt", 0.9)
        ],
        "n_to_generate": 3
    }

    final_state = graph.invoke(initial_input)

    print("\n--- [TEST RUN] Parent Graph Run Complete ---")
    print("\nFinal State:")
    print(final_state)
