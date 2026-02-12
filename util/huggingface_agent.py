from python_a2a import A2AServer, AgentCard, AgentSkill, Message, TextContent, MessageRole, run_server
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import HumanMessage

import dotenv
import os
import argparse
import torch

dotenv.load_dotenv()

SKIP_LLM = os.getenv("SKIP_LLM", "false").lower() == "true"
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "microsoft/Phi-3-mini-128k-instruct")
HF_DEVICE = os.getenv("HF_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
print(f"Configured Hugging Face model: {HF_MODEL_ID}")
print(f"Configured device: {HF_DEVICE}")

class HuggingFaceAgent(A2AServer):
    """A Python A2A agent that uses a local Hugging Face model via LangChain to respond to messages."""

    def __init__(self, url, agent_card, model_id=None, device=None, max_new_tokens=512, temperature=0):
        super().__init__(url=url, agent_card=agent_card)
        self.model_id = model_id or HF_MODEL_ID
        self.device = device or HF_DEVICE
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.llm = None
        
        if not SKIP_LLM:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize the Hugging Face model with LangChain."""
        print(f"Loading model: {self.model_id}")
        print(f"Using device: {self.device}")
        
        # Convert device string to integer expected by HuggingFacePipeline
        # -1 for CPU, 0+ for GPU device index
        device_int = 0 if self.device == "cuda" else -1
        
        try:
            pipeline_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "do_sample": True,
                "top_p": 0.95,
            }
            
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            llm = HuggingFacePipeline.from_model_id(
                model_id=self.model_id,
                task="text-generation",
                device=device_int,
                pipeline_kwargs=pipeline_kwargs,
                model_kwargs=model_kwargs,
            )
            
            # Wrap in ChatHuggingFace for structured output support
            self.llm = ChatHuggingFace(llm=llm)
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def handle_message(self, message):
        if message.content.type == "text":
            print(f"Received message: {message.content.text}")
            
            if SKIP_LLM:
                output_text = "Echo from HuggingFace agent: " + message.content.text
            else:
                try:
                    # Use LangChain's chat interface
                    messages = [HumanMessage(content=message.content.text)]
                    response = self.llm.invoke(messages)
                    output_text = response.content
                    
                except Exception as e:
                    output_text = f"Error generating response with model '{self.model_id}': {str(e)}"
                    print(output_text)
            
            return Message(
                content=TextContent(text=f"{output_text}"),
                role=MessageRole.AGENT,
                parent_message_id=message.message_id,
                conversation_id=message.conversation_id
            )


# Run the Python A2A server
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Hugging Face A2A agent.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind the server")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to bind the server")
    parser.add_argument("--card-url", type=str,
                        help="URL to advertise in the agent card")
    parser.add_argument("--model", type=str, default=HF_MODEL_ID,
                        help="Hugging Face model ID (e.g., microsoft/DialoGPT-medium, meta-llama/Llama-2-7b-chat-hf)")
    parser.add_argument("--device", type=str, default=HF_DEVICE,
                        choices=["cuda", "cpu", "mps"],
                        help="Device to run the model on")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for text generation")
    args = parser.parse_args()

    card_url = args.card_url or f"http://{args.host}:{args.port}"

    skill = AgentSkill(
        id='huggingface_chat',
        name='Chat using Hugging Face',
        description='Responds to messages using a local Hugging Face model via LangChain',
        tags=['chat', 'huggingface', 'local', 'langchain'],
        examples=['hi', 'hello', 'how are you?', 'tell me a story'],
    )
    
    agent = HuggingFaceAgent(
        url=f"http://{args.host}:{args.port}",
        model_id=args.model,
        device=args.device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        agent_card=AgentCard(
            name='Hugging Face Chat Agent',
            description=f'A chat agent powered by local Hugging Face model ({args.model})',
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
    print(f"Starting Hugging Face agent with model: {args.model}")
    print(f"Server listening on {args.host}:{args.port}")
    run_server(agent, host=args.host, port=args.port)
