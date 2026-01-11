# NAAMSE

[![CI](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml)

NAAMSE (Neural Adversarial Agent Mutation-based Security Evaluator) is an automated security fuzzing framework for LLM-based agents that uses evolutionary algorithms to discover vulnerabilities. Built on LangGraph and compliant with the AgentBeats A2A protocol, NAAMSE acts as a "green agent" that evaluates target "purple agents" by iteratively generating adversarial prompts through intelligent mutations, invoking the target agent, and scoring responses for security violations like jailbreaks, prompt injections, and PII leakage. The system employs a mutation engine with LLM-powered prompt transformations, a behavioral scoring engine using mixture-of-experts evaluation, and a clustering engine that organizes attack vectors by type. Over multiple iterations, high-scoring prompts (those that successfully exploit vulnerabilities) are selected as parents for the next generation, creating an evolutionary pressure toward more effective attacks. The framework outputs comprehensive PDF reports with vulnerability analysis, attack effectiveness metrics, and cluster-based categorization of discovered exploits, making it a practical tool for red-teaming and hardening LLM agents before deployment.

This template demonstrates a simple application implemented using [LangGraph](https://github.com/langchain-ai/langgraph), designed for showing how to get started with [LangGraph Server](https://langchain-ai.github.io/langgraph/concepts/langgraph_server/#langgraph-server) and using [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/), a visual debugging IDE.

<div align="center">
  <img src="./static/studio_ui.png" alt="Graph view in LangGraph studio UI" width="75%" />
</div>

The core logic defined in `src/agent/graph.py`, showcases an single-step application that responds with a fixed string and the configuration provided.

You can extend this graph to orchestrate more complex agentic workflows that can be visualized and debugged in LangGraph Studio.

## Top level Workflow
```mermaid
graph TD
    A[Start: Data Clusters] --> B(Select Initial Seed Prompts);
    B --> C(Mutation Engine);
    
    C -- "Generates n prompts" --> D{Loop: For each i in n};
    D -- "Next prompt" --> E(Invoke Agent Under Test);
    E --> F[Agent Output];
    F --> G(Behavioral Engine);
    G --> H[Output Score 0-100%];

    %% --- New Steps Added ---
    H --> H1{Score good AND Prompt new?};
    H1 -- "Yes" --> H2(Clustering Engine: Persist Prompt);
    H2 --> I(Record Scores);
    H1 -- "No" --> I(Record Scores);
    %% --- End of New Steps ---
    
    I --> D;
    
    D -- "n prompts tested" --> J{Check Exit Condition?};
    
    J -- "Avg Score < 90% AND Iterations < Limit" --> K(Select Parents based on Score Threshold);
    K --> C;
    
    J -- "Avg Score > 90% OR Iterations Reached" --> L(Pass Data to Report Generation Engine);
    L --> M[Generated Report];
    M --> Z[END];
```

## LangGraph Configuration
```json
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent/graph.py:graph",
    "fuzzer_iteration_subgraph": "./src/agent/fuzzer_iteration_subgraph.py:graph",
    "mutation_engine": "./src/mutation_engine/mutation_workflow.py:mutation_engine_graph",
    "mutation_action_subgraph": "./src/mutation_engine/nodes/run_mutation_action_subgraph.py:mutation_action_subgraph",
    "behavior_engine": "./src/behavioral_engine/behavior_engine_workflow.py:behavior_engine_graph",
    "invoke_agent": "./src/invoke_agent/invoke_agent_workflow.py:invoke_agent_graph"
  },
  "env": ".env",
  "image_distro": "wolfi"
}
```

## Mutation Agent Workflow
```mermaid
graph TD
    %% 1. Entry point
    A[START] --> B(calculate_probabilities);
    
    %% 2. The main loop starts here
    B --> C(select_prompt_by_probability);
    C --> D(decide_action_by_score);
    
    %% 3. Conditional branching based on state
    D -- "action == 'explore'" --> E(run_explore_action);
    D -- "action == 'similar'" --> F(run_similar_action);
    D -- "action == 'mutate'" --> G(run_mutation_workflow);
    
    %% 4. The mutation workflow is a subgraph/nested agent
    subgraph "Mutation Agent"
        G --> G_1(select_mutation_type);
        G_1 --> G_2(invoke_llm_with_tools_and_kb);
        G_2 --> G_End[Return Mutated Prompt];
    end
    
    %% 5. Collectors: All branches lead to the update node
    E --> H(add_prompt_to_output_list);
    F --> H;
    G_End --> H;
    
    %% 6. Conditional loop (edge)
    H --> I{check_loop_condition};
    I -- "i < n" --> C;
    I -- "i == n" --> Z[END];
```

## Behavioral Engine Workflow
```mermaid
graph TD

    %% Inputs
    X1["Conversation History – list of strings"] --> A
    X2["Input Prompt"] --> A

    %% Top-level Behavioral Engine
    A["Behavioral Engine"] --> B

    %% 3.1 Sanitization Layer (with sublayers)
    subgraph B["Sanitization Layer"]
        direction TB
        B1["Translation Layer<br/>Translate non-English text to English"]
        B2["Decode Encoded Data<br/>Decode Base64 or similar for further checks"]
        B3["Replace Unicode & Masked Characters<br/>(e.g., '@' → 'a')"]
    end

    %% Decision Node (Logic Diamond)
    B --> B4{Enough Data collected?}
    B4 -- Yes --> B6[Custom LLM finetuned on Input and output score]
    B4 -- No --> B5[Custom score]

    %% Connections from Decision Node
    B5 --> C
    B5 --> D
    B5 --> E

    %% 3.2 Factual Correctness
    C["Factual Correctness score<br/>Score factual accuracy (0–100) via semantic similarity"] --> W

    %% 3.3 Sensitive Data Detector
    D["Sensitive Data Score (0-100)<br/>Detect if AI output has PII"] --> W

    %% 3.4 LLM-based Score (with sublayers)
    subgraph E["LLM-based MOE Score (0-100)"]
        direction TB
        E1["Custom System Prompt:<br/>Define criteria for LLM-based scoring"]
        E1 --> E2["Agent 1"]
        E1 --> E3["Agent 2"]
        E1 --> E4["Agent N"]

        E2 --> E5["weighted aggregate score"]
        E3 --> E5
        E4 --> E5
    end
    E --> W["Combined weighted score"]
    B6 -->Z
    W --> Z
    %% 3.5 Fine-tuned Model (Weighted Final Score)
    Z["FinalScore (0–100)"]
    %% Output
```

## Clustering Engine Workflow
```mermaid
graph TD
    Start([Start]) --> load_data[1. load_data]

    %% --- Data Embedding Stage ---
    subgraph A["Embedding Generation"]
        load_data --> check_embed_cache{"embeddings.npy exists?"}
        check_embed_cache -- Yes --> load_embeds[Load cached embeddings]
        check_embed_cache -- No --> gen_embeds[Generate new embeddings]
        gen_embeds --> save_embeds[Save embeddings.npy]
        load_embeds --> optimize_kmeans[2. optimize_kmeans]
        save_embeds --> optimize_kmeans
    end

    %% --- Optimization Stage ---
    subgraph B["Parameter Optimization"]
        optimize_kmeans --> check_params_cache{"kmeans_params.json exists?"}
        check_params_cache -- Yes --> load_params[Load cached K-means params]
        check_params_cache -- No --> run_optuna[Run Optuna optimization]
        run_optuna --> save_params[Save kmeans_params.json]
        load_params --> cluster_data["3. cluster_data<br/>Perform initial K-means"]
        save_params --> cluster_data
    end

    %% --- Clustering Stage ---
    subgraph C["Recursive Clustering"]
        cluster_data --> check_cluster_cache{"final_clusters.pkl exists?"}
        check_cluster_cache -- Yes --> load_clusters[Load cached final clusters]
        check_cluster_cache -- No --> run_hierarchical[Run recursive clustering]
        run_hierarchical -- "Stop: cluster size <= 5" --> save_clusters[Save final_clusters.pkl]
        run_hierarchical -- "Stop: cannot split further" --> save_clusters
        run_hierarchical -- "Recursive Step" --> run_hierarchical
        load_clusters --> label_clusters[5. label_clusters]
        save_clusters --> label_clusters
    end

    %% --- Labeling Stage ---
    subgraph D["Cluster Labeling"]
        label_clusters --> check_llm_toggle{"use_llm_labeling == True?"}
        check_llm_toggle -- No --> generic_labels[Use generic labels]
        check_llm_toggle -- Yes --> check_label_cache{"cluster_labels.json exists?"}
        check_label_cache -- Yes --> load_labels[Load existing labels]
        check_label_cache -- No --> new_labels[Start empty labels]
        load_labels --> run_llm[Loop: Label remaining clusters w/ LLM]
        new_labels --> run_llm
        run_llm --> save_label_progress[Save progress to cluster_labels.json]
    end

    %% --- Saving Stage ---
    subgraph E["Output Saving"]
        generic_labels --> save_results["6. save_results<br/>Save all outputs"]
        save_label_progress --> save_results
        save_results --> Stop([END])
    end
```

## Getting Started

1. Install dependencies, along with the [LangGraph CLI](https://langchain-ai.github.io/langgraph/concepts/langgraph_cli/), which will be used to run the server.

```bash
cd path/to/your/app
pip install -e . "langgraph-cli[inmem]"
```

2. (Optional) Customize the code and project as needed. Create a `.env` file if you need to use secrets.

```bash
cp .env.example .env
```

If you want to enable LangSmith tracing, add your LangSmith API key to the `.env` file.

```text
# .env
LANGSMITH_API_KEY=lsv2...
```

3. Start the LangGraph Server.

```shell
# To start the LangGraph Server:
langgraph dev

# To run the A2A agent (if needed for testing):
uv run .\util\a2a_agent.py
```

For more information on getting started with LangGraph Server, [see here](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/).

## How to customize

1. **Define runtime context**: Modify the `Context` class in the `graph.py` file to expose the arguments you want to configure per assistant. For example, in a chatbot application you may want to define a dynamic system prompt or LLM to use. For more information on runtime context in LangGraph, [see here](https://langchain-ai.github.io/langgraph/agents/context/?h=context#static-runtime-context).

2. **Extend the graph**: The core logic of the application is defined in [graph.py](./src/agent/graph.py). You can modify this file to add new nodes, edges, or change the flow of information.

## Development

While iterating on your graph in LangGraph Studio, you can edit past state and rerun your app from previous states to debug specific nodes. Local changes will be automatically applied via hot reload.

Follow-up requests extend the same thread. You can create an entirely new thread, clearing previous history, using the `+` button in the top right.

For more advanced features and examples, refer to the [LangGraph documentation](https://langchain-ai.github.io/langgraph/). These resources can help you adapt this template for your specific use case and build more sophisticated conversational agents.

LangGraph Studio also integrates with [LangSmith](https://smith.langchain.com/) for more in-depth tracing and collaboration with teammates, allowing you to analyze and optimize your chatbot's performance.


# NAAMSE Green Agent

**A2A-compatible green agent for AgentBeats platform that runs LLM security fuzzing.**

NAAMSE (Neural Adversarial Agent Mutation-based Security Evaluator) is an A2A green agent that evaluates purple agents by running adversarial prompt mutations and scoring their responses to find vulnerabilities.

## Project Structure

```
src/
├─ agentbeats/           # A2A Green Agent Implementation
│  ├─ agent.py           # NAAMSE agent logic
│  ├─ executor.py        # A2A request handling
│  ├─ server.py          # Server entry point
│  ├─ models.py          # Pydantic models (EvalRequest, NAAMSEConfig)
│  └─ test_green_agent.py  # Test script
├─ agent/                # NAAMSE Fuzzer Graph (LangGraph)
│  ├─ graph.py           # Main fuzzer workflow
│  └─ ...
├─ mutation_engine/      # Prompt mutation subgraph
├─ behavioral_engine/    # Response scoring subgraph
├─ cluster_engine/       # Clustering and data management
└─ invoke_agent/         # Agent invocation subgraph
```

## Getting Started

### Prerequisites

1. Python 3.10+ with virtual environment
2. Google API key for Gemini models

### Installation

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies (if needed)
pip install -e .

# Set environment variables
$env:GOOGLE_API_KEY = "your-gemini-api-key"
```

## Running the Agent

Start the NAAMSE green agent server:

```powershell
.venv\Scripts\python.exe -m src.agentbeats.server --host 127.0.0.1 --port 9009
```

The agent will be available at:
- **Server:** http://127.0.0.1:9009
- **Agent Card:** http://127.0.0.1:9009/.well-known/agent-card.json

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Host to bind the server |
| `--port` | `9009` | Port to bind the server |
| `--card-url` | `http://{host}:{port}` | URL to advertise in the agent card |

## Testing

### 1. Start a Target Agent

Run the test purple agent (the agent to be evaluated):

```powershell
.venv\Scripts\python.exe util/a2a_agent.py --host 127.0.0.1 --port 5000
```

### 2. Run the Test Script

```powershell
.venv\Scripts\python.exe src/agentbeats/test_green_agent.py --target http://localhost:5000 --iterations 3 --mutations 2
```

### Test Options

| Flag | Default | Description |
|------|---------|-------------|
| `--target` | `http://localhost:5000` | Target agent URL to fuzz |
| `--green-agent` | `http://localhost:9009` | Green agent URL |
| `--iterations` | `3` | Number of fuzzer iterations |
| `--mutations` | `2` | Mutations per iteration |

## Request Format

The agent follows the [AgentBeats](https://agentbeats.dev) standard `EvalRequest` format:

```json
{
  "participants": {
    "agent": "http://localhost:5000"
  },
  "config": {
    "iterations_limit": 3,
    "mutations_per_iteration": 2,
    "score_threshold": 50.0,
    "initial_prompts": [
      {"prompt": ["your seed prompt"], "score": 0.0}
    ]
  }
}
```

**Required fields:**
- `participants.agent` - URL of the target agent to evaluate

**Optional config fields (with defaults):**
- `iterations_limit` (3) - Number of fuzzer iterations
- `mutations_per_iteration` (2) - Mutations per iteration
- `score_threshold` (50.0) - Score threshold for prompt selection
- `initial_prompts` (null) - Seed prompts to start fuzzing

## Architecture

Based on the [green-agent-template](https://github.com/RDI-Foundation/green-agent-template), NAAMSE implements:

- **Agent class** ([agent.py](src/agentbeats/agent.py)) - Core NAAMSE evaluation logic
- **Executor class** ([executor.py](src/agentbeats/executor.py)) - A2A protocol handling
- **Server** ([server.py](src/agentbeats/server.py)) - HTTP server with agent card
- **Models** ([models.py](src/agentbeats/models.py)) - Request/response schemas

The agent delegates fuzzing to the NAAMSE LangGraph workflow defined in [src/agent/graph.py](src/agent/graph.py).

---

# NAAMSE Fuzzer Workflow Documentation

## LangGraph Configuration
