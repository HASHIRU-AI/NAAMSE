# NAAMSE

<!-- [![CI](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml) -->

**Neural Adversarial Agent Mutation-based Security Evaluator**

NAAMSE (Neural Adversarial Agent Mutation-based Security Evaluator) is an automated security fuzzing framework for LLM-based agents that uses evolutionary algorithms to discover vulnerabilities. Built on LangGraph and compliant with the AgentBeats A2A protocol, NAAMSE acts as a "green agent" that evaluates target "purple agents" by iteratively generating adversarial prompts through intelligent mutations, invoking the target agent, and scoring responses for security violations like jailbreaks, prompt injections, and PII leakage. The system employs a mutation engine with LLM-powered prompt transformations, a behavioral scoring engine using mixture-of-experts evaluation, and a clustering engine that organizes attack vectors by type. Over multiple iterations, high-scoring prompts (those that successfully exploit vulnerabilities) are selected as parents for the next generation, creating an evolutionary pressure toward more effective attacks. The framework outputs comprehensive PDF reports with vulnerability analysis, attack effectiveness metrics, and cluster-based categorization of discovered exploits, making it a practical tool for red-teaming and hardening LLM agents before deployment.

## Links

- **Green Agent Docker Image**: [https://github.com/HASHIRU-AI/NAAMSE/pkgs/container/naamse-naamse-green-agent](https://github.com/HASHIRU-AI/NAAMSE/pkgs/container/naamse-naamse-green-agent)
- **Green Agent Agentbeats Page**: [https://agentbeats.dev/helloparthshah/naamse-neural-adversarial-agent-mutation-based-security-evaluator](https://agentbeats.dev/helloparthshah/naamse-neural-adversarial-agent-mutation-based-security-evaluator)
- **Example Purple Agent Docker Image**: [https://github.com/HASHIRU-AI/NAAMSE/pkgs/container/naamse-naamse-purple-agent](https://github.com/HASHIRU-AI/NAAMSE/pkgs/container/
naamse-naamse-purple-agent)
- **Purple agent Agentbeats Page**: [https://agentbeats.dev/helloparthshah/naamse-purpleagent](https://agentbeats.dev/helloparthshah/naamse-purpleagent)
- **Leaderboard Repository**: [https://github.com/HASHIRU-AI/naamse-leaderboard/tree/main](https://github.com/HASHIRU-AI/naamse-leaderboard/tree/main)
- **Live Leaderboard**: [https://agentbeats.dev/helloparthshah/naamse-neural-adversarial-agent-mutation-based-security-evaluator](https://agentbeats.dev/helloparthshah/naamse-neural-adversarial-agent-mutation-based-security-evaluator)

<!-- <div align="center">
  <img src="./static/studio_ui.png" alt="Graph view in LangGraph studio UI" width="75%" />
</div> -->

## Quick Start

### Docker

The easiest way to run NAAMSE is using the pre-built Docker image:

```bash
# Pull the Docker image
docker pull YOUR_USERNAME/naamse-green-agent:latest

# Run the green agent
# .env expects GOOGLE_API_KEY to be set at least. 
# look at .env.example for more information.
docker run -p 9009:9009 \
   --env-file .env \
  YOUR_USERNAME/naamse-green-agent:latest
```

The agent will be available at:
- **Server**: http://localhost:9009
- **Agent Card**: http://localhost:9009/.well-known/agent-card.json

#### Building the Docker Image Locally

```bash
# Clone the repository
git clone https://github.com/HASHIRU-AI/NAAMSE.git

# Build the image
docker build -f ./scenarios/naamse/Dockerfile.naamse-green-agent -t naamse-naamse-green-agent .
# Run the container
# .env expects GOOGLE_API_KEY to be set at least. 
# look at .env.example for more information.
docker run -p 9009:9009 \
  --env-file .env \
  naamse-green-agent
```

## Request Format

The agent follows the [AgentBeats](https://agentbeats.dev) standard `EvalRequest` format:

```json
{
  "participants": {
    "agent": "http://localhost:5000"
  },
  "config": {
    "iterations_limit": 7,
    "mutations_per_iteration": 4,
    "score_threshold": 50.0,
  }
}
```

**Required fields:**

- `participants.agent` - URL of the target agent to evaluate
- `iterations_limit` (7) - Number of fuzzer iterations
- `mutations_per_iteration` (4) - Mutations per iteration
- `score_threshold` (50.0) - Score threshold for prompt selection

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

<!-- ## Leaderboard

View the latest evaluation results and agent rankings on our [live leaderboard](https://naamse-leaderboard.example.com).

The leaderboard tracks:
- Overall security scores across all evaluated agents
- Vulnerability breakdowns by category (jailbreaks, prompt injections, PII leakage)
- Historical performance trends
- Attack vector effectiveness metrics -->
