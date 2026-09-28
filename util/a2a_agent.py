from python_a2a import A2AServer, AgentCard, AgentSkill, Message, TextContent, MessageRole, run_server
from google import genai
from google.genai.types import GenerateContentConfig

import dotenv
import os
import argparse

dotenv.load_dotenv()

api_key = os.getenv("INVOKE_AGENT_API_KEY") or os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)
SKIP_LLM = os.getenv("SKIP_LLM", "false").lower() == "true"

META_MODEL_API_BASE_URL = "https://api.meta.ai/v1"


def ask_gemini(text: str, model: str) -> str:
    chat = client.chats.create(model=model)
    response = chat.send_message(text, config=GenerateContentConfig(temperature=0.0))
    return response.text


def ask_meta(text: str, model: str) -> str:
    """Ask a Meta Muse Spark model through the OpenAI-compatible Meta Model API."""
    from openai import OpenAI  # lazy: only the meta provider needs the openai package
    meta_client = OpenAI(
        base_url=os.getenv("META_MODEL_API_BASE_URL", META_MODEL_API_BASE_URL),
        api_key=os.getenv("MODEL_API_KEY"),
    )
    response = meta_client.chat.completions.create(
        model=model, temperature=0.0, messages=[{"role": "user", "content": text}])
    return response.choices[0].message.content or ""


PROVIDERS = {"gemini": ask_gemini, "meta": ask_meta}
DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "meta": "muse-spark-1.2"}


class EchoAgent(A2AServer):
    """A simple [Python A2A](python-a2a.html) agent that answers messages with the configured model."""

    provider = "gemini"
    model = DEFAULT_MODELS["gemini"]

    def handle_message(self, message):
        if message.content.type == "text":
            print(f"Received message: {message.content.text}")
            if SKIP_LLM:
                output_text = "No you cannot gaslight me! You said: " + message.content.text
            else:
                output_text = PROVIDERS[self.provider](message.content.text, self.model)
            return Message(
                content=TextContent(text=f"{output_text}"),
                role=MessageRole.AGENT,
                parent_message_id=message.message_id,
                conversation_id=message.conversation_id
            )


# Run the [Python A2A](python-a2a.html) server
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind the server")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to bind the server")
    parser.add_argument("--card-url", type=str,
                        help="URL to advertise in the agent card")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="gemini",
                        help="Model provider backing this agent")
    parser.add_argument("--model", type=str, default=None,
                        help="Model ID (default: gemini-2.5-flash, or muse-spark-1.2 for --provider meta)")
    args = parser.parse_args()
    EchoAgent.provider = args.provider
    EchoAgent.model = args.model or DEFAULT_MODELS[args.provider]
    print(f"Target agent backed by {EchoAgent.provider}:{EchoAgent.model}")

    card_url = args.card_url or f"http://{args.host}:{args.port}"

    skill = AgentSkill(
        id='hello_world',
        name='Returns hello world',
        description='just returns hello world',
        tags=['hello world'],
        examples=['hi', 'hello world'],
    )
    agent = EchoAgent(url=f"http://{args.host}:{args.port}",
                      agent_card=AgentCard(
                          name='Hello World Agent',
                          description='Just a hello world agent',
                          url=f"http://{args.host}:{args.port}",
                          version='1.0.0',
                          default_input_modes=['text'],
                          default_output_modes=['text'],
                          capabilities={
                              "streaming": False,
                              "pushNotifications": False,
                              "stateTransitionHistory": False
                          },
                          # Only the basic skill for the public card
                          skills=[skill],
                      ))
    agent.agent_card.capabilities["streaming"] = False
    run_server(agent, host=args.host, port=args.port)
