from python_a2a import A2AServer, AgentCard, AgentSkill, Message, TextContent, MessageRole, run_server
from google import genai
import dotenv
import os

dotenv.load_dotenv()

api_key = os.getenv("INVOKE_AGENT_API_KEY")
client = genai.Client(api_key=api_key)


class EchoAgent(A2AServer):
    """A simple [Python A2A](python-a2a.html) agent that echoes back messages with a prefix."""

    def handle_message(self, message):
        if message.content.type == "text":
            print(f"Received message: {message.content.text}")
            chat = client.chats.create(model="gemini-2.5-flash")
            response = chat.send_message(message.content.text)
            return Message(
                content=TextContent(text=f"{response.text}"),
                role=MessageRole.AGENT,
                parent_message_id=message.message_id,
                conversation_id=message.conversation_id
            )


# Run the [Python A2A](python-a2a.html) server
if __name__ == "__main__":
    skill = AgentSkill(
        id='hello_world',
        name='Returns hello world',
        description='just returns hello world',
        tags=['hello world'],
        examples=['hi', 'hello world'],
    )
    agent = EchoAgent(url="http://localhost:5000",
                      agent_card=AgentCard(
                          name='Hello World Agent',
                          description='Just a hello world agent',
                          url='http://localhost:5000/',
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
    run_server(agent, host="0.0.0.0", port=5000)
