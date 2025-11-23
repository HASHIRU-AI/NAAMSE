from python_a2a import A2AServer, Message, TextContent, MessageRole, run_server

class EchoAgent(A2AServer):
    """A simple [Python A2A](python-a2a.html) agent that echoes back messages with a prefix."""
    
    def handle_message(self, message):
        if message.content.type == "text":
            return Message(
                content=TextContent(text=f"Echo: {message.content.text}"),
                role=MessageRole.AGENT,
                parent_message_id=message.message_id,
                conversation_id=message.conversation_id
            )

# Run the [Python A2A](python-a2a.html) server
if __name__ == "__main__":
    agent = EchoAgent(url="http://localhost:5000")
    run_server(agent, host="0.0.0.0", port=5000)