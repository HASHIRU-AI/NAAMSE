from python_a2a import A2AServer, AgentCard, AgentSkill, Message, TextContent, MessageRole, run_server
import ollama

import dotenv
import os
import argparse

dotenv.load_dotenv()

SKIP_LLM = os.getenv("SKIP_LLM", "false").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")


class OllamaAgent(A2AServer):
    """A simple Python A2A agent that uses a local Ollama model to respond to messages."""

    def __init__(self, url, agent_card, model_name=None):
        super().__init__(url=url, agent_card=agent_card)
        self.model_name = model_name or OLLAMA_MODEL

    def handle_message(self, message):
        if message.content.type == "text":
            print(f"Received message: {message.content.text}")
            message_text = str(bytes(message.content.text, 'utf-8', 'backslashreplace'), 'utf-8')
            if SKIP_LLM:
                output_text = "Echo from Ollama agent: " + message_text
            else:
                try:
                    response = ollama.chat(
                        model=self.model_name,
                        messages=[
                            {
                                'role': 'user',
                                'content': message_text,
                            },
                        ]
                    )
                    output_text = response['message']['content']
                except Exception as e:
                    output_text = f"Error calling Ollama model '{self.model_name}': {str(e)}"
                    print(output_text)
            
            return Message(
                content=TextContent(text=f"{output_text}"),
                role=MessageRole.AGENT,
                parent_message_id=message.message_id,
                conversation_id=message.conversation_id
            )


# Run the Python A2A server
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Ollama A2A agent.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind the server")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to bind the server")
    parser.add_argument("--card-url", type=str,
                        help="URL to advertise in the agent card")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL,
                        help="Ollama model name to use (e.g., llama2, mistral, codellama)")
    args = parser.parse_args()

    card_url = args.card_url or f"http://{args.host}:{args.port}"

    skill = AgentSkill(
        id='ollama_chat',
        name='Chat using Ollama',
        description='Responds to messages using a local Ollama model',
        tags=['chat', 'ollama', 'local'],
        examples=['hi', 'hello', 'how are you?'],
    )
    
    agent = OllamaAgent(
        url=f"http://{args.host}:{args.port}",
        model_name=args.model,
        agent_card=AgentCard(
            name='Ollama Chat Agent',
            description=f'A chat agent powered by local Ollama model ({args.model})',
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
        )
    )
    
    agent.agent_card.capabilities["streaming"] = False
    print(f"Starting Ollama agent with model: {args.model}")
    print(f"Server listening on {args.host}:{args.port}")
    run_server(agent, host=args.host, port=args.port)
