import os
from termcolor import cprint

from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger as AgentEventLogger
from llama_stack_client import LlamaStackClient

from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer
from docling.chunking import HybridChunker

from dotenv import load_dotenv

#pip install -qU docling transformers

load_dotenv()

LLAMA_STACK_SERVER=os.getenv("LLAMA_STACK_SERVER")
LLAMA_STACK_MODEL=os.getenv("LLAMA_STACK_MODEL")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256 #https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/discussions/1
DOC_SOURCE = "./docs/allsprings-prospectus-dec-2024.pdf"

print(LLAMA_STACK_SERVER)
print(LLAMA_STACK_MODEL)

client = LlamaStackClient(base_url=os.getenv("LLAMA_STACK_SERVER"))

# Register a vector db
vector_db_id = "my_documents"
response = client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model="all-MiniLM-L6-v2",
    embedding_dimension=384,
    provider_id="faiss",
)

# Read the PDF document using Docling
doc = DocumentConverter().convert(source=DOC_SOURCE).document
# Extract document text chunks
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
chunker = HybridChunker(
    tokenizer=tokenizer,  # instance or model name, defaults to "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer`
    merge_peers=True,  # optional, defaults to True
)

chunk_iter = chunker.chunk(dl_doc=doc)
chunks = list(chunk_iter)
ser_chunks = []
for i, chunk in enumerate(chunks):
    # print(f"=== Chunk #{i} ===")
    # txt_tokens = len(tokenizer.tokenize(chunk.text))
    # print(f"chunk.text ({txt_tokens} tokens):\n{repr(chunk.text)}")

    ser_txt = chunker.serialize(chunk=chunk)
    ser_tokens = len(tokenizer.tokenize(ser_txt))
    # print(f"chunker.serialize(chunk) ({ser_tokens} tokens):\n{repr(ser_txt)}")

    ser_chunks.append({
        "content": ser_txt,
        "mime_type": "text/plain",
        "metadata": {
            "document_id": doc.name,
            "token_count": ser_tokens,
            "source": DOC_SOURCE,
        }
    })

# print(f"chunks vectorized: {len(ser_chunks)}")
# insert the chunks into the vector db
client.vector_io.insert(vector_db_id=vector_db_id, chunks=ser_chunks)

rag_agent = Agent(
    client,
    model=os.getenv("INFERENCE_MODEL"),  
    # Define instructions for the agent ( aka system prompt)
    instructions="You are a helpful assistant specialized in retrieving information from documents containing financial information. Use the knowledge_search tool to help answer questions.",
    enable_session_persistence=False,
    # Define tools available to the agent
    tools=[
        {
            "name": "builtin::rag/knowledge_search",
            "args": {
                "vector_db_ids": [vector_db_id],
            },
        }
    ],
)
session_id = rag_agent.create_session("test-session")

user_prompts = [
    "what are the Example of Expenses after 1 year and after 3 years?",
    "What is the investment Objective for the fund?",
    # "What is the management fees?",
    # "Show me the Example of Expenses as a table?",
    "List the Average Annual Total Returns for the periods ended 12/31/2024 for Class A (before taxes)"
    # "What is the fund's investment strategy?",
    # "What is the fund's risk profile?"
]

# Run the agent loop by calling the `create_turn` method
for prompt in user_prompts:
    cprint(f"User> {prompt}", "green")
    response = rag_agent.create_turn(
        messages=[{"role": "user", "content": prompt}],
        session_id=session_id,
    )
    for log in AgentEventLogger().log(response):
        log.print()

#unregister the vector db
response = client.vector_dbs.unregister(vector_db_id=vector_db_id)