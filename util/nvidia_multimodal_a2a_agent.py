from python_a2a import A2AServer, AgentCard, AgentSkill, Message, TextContent, MessageRole, run_server

import requests
import json
import dotenv
import os
import argparse

dotenv.load_dotenv()

# NVIDIA API configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-4-scout-17b-16e-instruct")


class NvidiaAgent(A2AServer):
    """A simple A2A agent that uses NVIDIA's Llama model for responses."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle_message(self, message):
        if message.content.type == "text":
            print(f"Received message: {message.content.text}")
            message_text = str(bytes(message.content.text, 'utf-8', 'backslashreplace'), 'utf-8')
            
            try:
                invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
                stream = True

                headers = {
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Accept": "text/event-stream" if stream else "application/json"
                }

                payload = {
                    "model": NVIDIA_MODEL,
                    "messages": [{"role":"user","content":message_text}],
                    "max_tokens": 512,
                    "temperature": 1.00,
                    "top_p": 1.00,
                    "frequency_penalty": 0.00,
                    "presence_penalty": 0.00,
                    "stream": stream
                }

                response = requests.post(invoke_url, headers=headers, json=payload)

                content_parts = []

                if stream:
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode("utf-8")
                            if line_str.startswith("data: "):
                                data = line_str[6:]
                                if data == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data)
                                    if "choices" in chunk and chunk["choices"]:
                                        delta = chunk["choices"][0]["delta"]
                                        if "content" in delta:
                                            content_parts.append(delta["content"])
                                            print(delta["content"], end="")
                                except json.JSONDecodeError:
                                    pass
                else:
                    result = response.json()
                    if "choices" in result and result["choices"]:
                        content_parts.append(result["choices"][0]["message"]["content"])
                        print(result["choices"][0]["message"]["content"])

                output_text = "".join(content_parts)
            
            except Exception as e:
                print(f"Error occurred: {e}")
                output_text = "I apologize, but an error occurred while generating a response."
            
            return Message(
                content=TextContent(text=output_text),
                role=MessageRole.AGENT,
                parent_message_id=message.message_id,
                conversation_id=message.conversation_id
            )


# Run the server
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the NVIDIA Llama A2A agent.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind the server")
    parser.add_argument("--port", type=int, default=5001,
                        help="Port to bind the server")
    parser.add_argument("--card-url", type=str,
                        help="URL to advertise in the agent card")
    args = parser.parse_args()

    card_url = args.card_url or f"http://{args.host}:{args.port}"

    skill = AgentSkill(
        id='nvidia_llama_chat',
        name='Chat with NVIDIA Llama',
        description='Responds using NVIDIA Llama model',
        tags=['chat', 'nvidia', 'llama'],
        examples=['Hello', 'What is AI?', 'Explain quantum computing'],
    )
    agent = NvidiaAgent(url=f"http://{args.host}:{args.port}",
                        agent_card=AgentCard(
                            name='NVIDIA Llama Agent',
                            description='An agent powered by NVIDIA Llama',
                            url=f"http://{args.host}:{args.port}",
                            version='1.0.0',
                            default_input_modes=['text'],
                            default_output_modes=['text'],
                            capabilities={
                                "streaming": False,
                                "pushNotifications": False,
                                "stateTransitionHistory": False
                            },
                            skills=[skill],
                        ))
    agent.agent_card.capabilities["streaming"] = False
    run_server(agent, host=args.host, port=args.port)
