# NAAMSE

<!-- [![CI](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml) -->

**Neural Adversarial Agent Mutation-based Security Evaluator**

NAAMSE (Neural Adversarial Agent Mutation-based Security Evaluator) is an automated security fuzzing framework for LLM-based agents that uses evolutionary algorithms to discover vulnerabilities. Built on LangGraph and compliant with the AgentBeats A2A protocol, NAAMSE acts as a "green agent" that evaluates target "purple agents" by iteratively generating adversarial prompts through intelligent mutations, invoking the target agent, and scoring responses for security violations like jailbreaks, prompt injections, and PII leakage. The system employs a mutation engine with LLM-powered prompt transformations, a behavioral scoring engine using mixture-of-experts evaluation, and a clustering engine that organizes attack vectors by type using separate SQLite databases for adversarial and benign prompt corpora. Over multiple iterations, high-scoring prompts (those that successfully exploit vulnerabilities) are selected as parents for the next generation, creating an evolutionary pressure toward more effective attacks. The framework outputs comprehensive PDF reports with vulnerability analysis, attack effectiveness metrics, and cluster-based categorization of discovered exploits, making it a practical tool for red-teaming and hardening LLM agents before deployment.

## Links

- **Green Agent Docker Image**: [https://github.com/HASHIRU-AI/NAAMSE/pkgs/container/naamse-naamse-green-agent](https://github.com/HASHIRU-AI/NAAMSE/pkgs/container/naamse-naamse-green-agent)
- **Green Agent Agentbeats Page**: [https://agentbeats.dev/helloparthshah/naamse-neural-adversarial-agent-mutation-based-security-evaluator](https://agentbeats.dev/helloparthshah/naamse-neural-adversarial-agent-mutation-based-security-evaluator)
- **Example Purple Agent Docker Image**: [https://github.com/HASHIRU-AI/NAAMSE/pkgs/container/naamse-naamse-purple-agent](https://github.com/HASHIRU-AI/NAAMSE/pkgs/container/naamse-naamse-purple-agent)
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
docker pull ghcr.io/hashiru-ai/naamse-naamse-green-agent:latest

# Run the green agent
# .env expects GOOGLE_API_KEY to be set at least. 
# look at .env.example for more information.
docker run -p 9009:9009 \
   --env-file .env \
  ghcr.io/hashiru-ai/naamse-naamse-green-agent:latest
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

## Local Development Setup

For local development without Docker:

1. **Prerequisites**: Ensure Python 3.10+ is installed.

2. **Install uv** (Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Clone the repository**:
   ```bash
   git clone https://github.com/HASHIRU-AI/NAAMSE.git
   cd NAAMSE
   ```

4. **Install dependencies**:
   ```bash
   uv sync
   ```

5. **Activate the virtual environment**:
   ```bash
   # Linux/macOS
   source .venv/bin/activate

   # Windows
   .venv\Scripts\activate
   ```

6. **Set up environment variables**:
   - Copy `.env.example` to `.env`
   - Set required variables (e.g., `GOOGLE_API_KEY`). Refer to `.env.example` for details.

7. **Run the server**:
   ```bash
   python -m src.agentbeats.server --port 9009
   ```
   The server will start on `http://localhost:9009`.

8. **Test the agent** (in another terminal, with the virtual environment activated):
   ```bash
   python src/agentbeats/test_green_agent.py --target http://localhost:5000 --green-agent http://localhost:9009
   ```
   - `--target`: URL of the target agent to evaluate
   - `--green-agent`: URL of the running green agent

   This will send a test evaluation request and stream the results.

   You can also use `langgraph dev` for an interactive testing experience.

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
  }
}
```

**Required fields:**

- `participants.agent` - URL of the target agent to evaluate
- `iterations_limit` (7) - Number of fuzzer iterations
- `mutations_per_iteration` (4) - Mutations per iteration

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
│  ├─ graph.py           # Main fuzzer workflow with parallel iteration workers
│  └─ ...
├─ mutation_engine/      # Prompt mutation subgraph
├─ behavioral_engine/    # Response scoring subgraph (PII detection, jailbreak scoring)
├─ cluster_engine/       # Clustering and data management
│  ├─ data_access/
│  │  ├─ adversarial/    # SQLite database with 128K+ adversarial jailbreak prompts
│  │  ├─ benign/         # SQLite database with 50K+ benign security testing prompts
│  │  ├─ sqlite_source.py # Database access layer with embedding-based similarity search
│  │  └─ ...
│  └─ ...
└─ invoke_agent/         # Agent invocation subgraph
```

## Database Structure

NAAMSE uses separate SQLite databases for different types of prompts:

- **Adversarial Database** (`src/cluster_engine/data_access/adversarial/naamse.db`): 128,000+ jailbreak and adversarial prompts organized into hierarchical clusters by attack type (DAN prompts, uncensored personas, encoding attacks, etc.)
- **Benign Database** (`src/cluster_engine/data_access/benign/naamse_benign.db`): 50,000+ benign prompts for security testing, including legitimate user queries that help validate scoring accuracy

Both databases include:
- Prompt text and metadata
- Hierarchical clustering information
- Sentence embeddings for similarity search
- Cluster labels and categorization

## Generate PDF report

After getting the JSON report from the green agent.
You can generate the pdf by running the `util/generate_pdf_report.py` by using the following command.

```shell
# Linux/macOS
python ./util/generate_pdf_report.py --input_json=<input_json> --output_pdf=<output_pdf>

# Windows
python .\util\generate_pdf_report.py --input_json=<input_json> --output_pdf=<output_pdf>
```

<!-- ## Leaderboard

View the latest evaluation results and agent rankings on our [live leaderboard](https://naamse-leaderboard.example.com).

The leaderboard tracks:
- Overall security scores across all evaluated agents
- Vulnerability breakdowns by category (jailbreaks, prompt injections, PII leakage)
- Historical performance trends
- Attack vector effectiveness metrics -->
