# NAAMSE Green Agent

A2A-compatible agent that runs the NAAMSE fuzzer against target LLM agents.

## Prerequisites

1. Set your Google API key:
   ```powershell
   $env:GOOGLE_API_KEY = "your-gemini-api-key"
   ```

2. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

## Running the Agent

Start the server:
```powershell
python -m src.agentbeats.server --host localhost --port 8000
```

The agent card will be available at: `http://localhost:8000/.well-known/agent.json`

## Running the Testing Agent

Run the a2a agent that we are testing:

```powershell
python util/a2a_agent.py
```

## Testing

Run the test script:
```powershell
python test_green_agent.py --target http://localhost:5000 --iterations 1 --mutations 1
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--target` | `http://localhost:5000` | Target agent URL to fuzz |
| `--green-agent` | `http://localhost:8000` | Green agent URL |
| `--iterations` | `1` | Number of fuzzer iterations |
| `--mutations` | `1` | Mutations per iteration |

## Request Format

The agent accepts A2A JSON-RPC requests with this payload:

```json
{
  "target_url": "http://localhost:5000",
  "iterations_limit": 3,
  "mutations_per_iteration": 2,
  "score_threshold": 50.0,
  "initial_prompts": [
    {"prompt": ["your seed prompt"], "score": 0.0}
  ]
}
```

Only `target_url` is required; other fields have defaults.
