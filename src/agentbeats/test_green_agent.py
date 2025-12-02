"""
Test script for the NAAMSE Green Agent.
Run the server first: python -m src.agentbeats.server
"""
import httpx
import uuid
import json


def test_green_agent(
    target_url: str = "http://localhost:5001",
    green_agent_url: str = "http://localhost:8000",
    iterations_limit: int = 3,
    mutations_per_iteration: int = 3,
):
    """Send a test request to the NAAMSE Green Agent."""
    
    # Build the A2A JSON-RPC request
    request = {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "kind": "message",
                "parts": [{
                    "kind": "text",
                    "text": json.dumps({
                        "target_url": target_url,
                        "iterations_limit": iterations_limit,
                        "mutations_per_iteration": mutations_per_iteration,
                    })
                }],
                "messageId": str(uuid.uuid4()),
                "contextId": str(uuid.uuid4()),
            }
        }
    }
    
    print(f"Sending request to {green_agent_url}")
    print(f"Target agent: {target_url}")
    print(f"Config: iterations={iterations_limit}, mutations={mutations_per_iteration}")
    print("-" * 50)
    
    response = httpx.post(
        green_agent_url,
        json=request,
        timeout=300.0  # 5 minute timeout for fuzzing
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    
    return response.json()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the NAAMSE Green Agent")
    parser.add_argument("--target", default="http://localhost:5000", help="Target agent URL to fuzz")
    parser.add_argument("--green-agent", default="http://localhost:8000", help="Green agent URL")
    parser.add_argument("--iterations", type=int, default=1, help="Number of fuzzer iterations")
    parser.add_argument("--mutations", type=int, default=1, help="Mutations per iteration")
    
    args = parser.parse_args()
    
    test_green_agent(
        target_url=args.target,
        green_agent_url=args.green_agent,
        iterations_limit=args.iterations,
        mutations_per_iteration=args.mutations,
    )
