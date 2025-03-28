from llama_stack_client import Agent, AgentEventLogger
from termcolor import cprint
from dotenv import load_dotenv
import os
from llama_stack_client import LlamaStackClient

load_dotenv()

LLAMA_STACK_SERVER=os.getenv("LLAMA_STACK_SERVER")
LLAMA_STACK_MODEL=os.getenv("LLAMA_STACK_MODEL")

print(LLAMA_STACK_SERVER)
print(LLAMA_STACK_MODEL)

client = LlamaStackClient(
    base_url=LLAMA_STACK_SERVER,
)

agent = Agent(
    client, 
    model=LLAMA_STACK_MODEL,
    instructions="You are a helpful assistant. Use websearch tool to help answer questions.",
    tools=["builtin::websearch"],
)
user_prompts = [
    "Hello",
    "Who won the last Super Bowl?",
]

session_id = agent.create_session("test-session")
for prompt in user_prompts:
    cprint(f"User> {prompt}", "green")
    response = agent.create_turn(
      messages=[
          {
              "role": "user",
              "content": prompt,
          }
      ],
      session_id=session_id,
    )
    for log in AgentEventLogger().log(response):
       log.print()
