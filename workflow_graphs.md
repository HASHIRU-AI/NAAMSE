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