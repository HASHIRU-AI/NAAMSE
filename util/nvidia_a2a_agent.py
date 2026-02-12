from python_a2a import A2AServer, AgentCard, AgentSkill, Message, TextContent, MessageRole, run_server
from openai import OpenAI

import dotenv
import os
import argparse

dotenv.load_dotenv()

# NVIDIA API configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-120b")


class NvidiaAgent(A2AServer):
    """A simple A2A agent that uses NVIDIA's GPT model for responses."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )

    def handle_message(self, message):
        if message.content.type == "text":
            print(f"Received message: {message.content.text}")
            message_text = str(bytes(message.content.text, 'utf-8', 'backslashreplace'), 'utf-8')
            
            try:
                completion = self.client.chat.completions.create(
                    model=NVIDIA_MODEL,
                    messages=[{"role":"user","content":message_text}],
                    temperature=1,
                    top_p=1,
                    max_tokens=4096,
                    stream=True
                    )

                reasoning_parts = []
                content_parts = []
                
                for chunk in completion:
                    if not getattr(chunk, "choices", None):
                        continue
                    reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        print(reasoning, end="")
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content_parts.append(chunk.choices[0].delta.content)
                        print(chunk.choices[0].delta.content, end="")
                
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
    parser = argparse.ArgumentParser(description="Run the NVIDIA GPT A2A agent.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind the server")
    parser.add_argument("--port", type=int, default=5001,
                        help="Port to bind the server")
    parser.add_argument("--card-url", type=str,
                        help="URL to advertise in the agent card")
    args = parser.parse_args()

    card_url = args.card_url or f"http://{args.host}:{args.port}"

    skill = AgentSkill(
        id='nvidia_gpt_chat',
        name='Chat with NVIDIA GPT',
        description='Responds using NVIDIA GPT model',
        tags=['chat', 'nvidia', 'gpt'],
        examples=['Hello', 'What is AI?', 'Explain quantum computing'],
    )
    agent = NvidiaAgent(url=f"http://{args.host}:{args.port}",
                        agent_card=AgentCard(
                            name='NVIDIA GPT Agent',
                            description='An agent powered by NVIDIA GPT',
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
