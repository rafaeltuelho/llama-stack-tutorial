import os
import logging
from io import BytesIO

import streamlit as st
from document_processor import process_document
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a helpful assistant that can use tools to answer questions.
"""
LLAMA_STACK_SERVER=os.getenv("LLAMA_STACK_SERVER")
LLAMA_STACK_MODEL=os.getenv("LLAMA_STACK_MODEL")
VISION_MODEL="granite3.2-vision:2b"
VECTOR_DB_ID = "chat_documents"

def register_vector_db(client):
    try:
        response = client.vector_dbs.retrieve(vector_db_id=VECTOR_DB_ID)
        if response:
            print(f"Vector db {VECTOR_DB_ID} already exists.")
            return
    except Exception as e:
        logger.error("Vector db %s does not exist: %s", VECTOR_DB_ID, e)
        response = client.vector_dbs.register(
            vector_db_id=VECTOR_DB_ID,
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
            provider_id="faiss",
        )
        if response:
            print(f"Vector db {VECTOR_DB_ID} registered.")
        else:
            return
def init_session_state():
    # Initialize LlamaStack client and Agent
    client = LlamaStackClient(base_url=LLAMA_STACK_SERVER)

    # Register a vector db
    register_vector_db(client)
    # Register vision model
    client.models.register(
        model_id="granite3.2-vision:2b",
        model_type="llm",
        provider_id="ollama",
        provider_model_id="granite3.2-vision:2b",
        metadata={"description": "granite3.2-vision:2b via ollama"}
    )
    # Register the model and safety shield
    client.models.register(
        model_id="meta-llama/Llama-Guard-3-8B",
        model_type="llm",
        provider_id="ollama",
        provider_model_id="llama-guard3:8b-q4_0",
        metadata={"description": "llama-guard3:8b-q4_0 via ollama"}
    )
    # Register a safety shield
    client.shields.register(shield_id="content_safety", provider_shield_id="Llama-Guard-3-8B")

    agent = Agent(
        client, 
        model=LLAMA_STACK_MODEL,
        instructions=SYSTEM_PROMPT,
        enable_session_persistence=True,
        input_shields=["content_safety"],
        output_shields=["content_safety"],
        # Control the inference loop
        max_infer_iters=5,
        tools=[
            {
                "name": "builtin::rag/knowledge_search",
                "args": { "vector_db_ids": [VECTOR_DB_ID] },
            }
        ])
    agent.create_session("rag-session")
    logger.info("LlamaStack Agent created with session Id: %s", agent.session_id)

    st.session_state.initialized = True
    st.session_state.agent = agent
    st.session_state.client = client

if __name__ == '__main__':

    # Initialize session state if not already done
    if "initialized" not in st.session_state:
        init_session_state()

    agent = st.session_state.agent
    client = st.session_state.client
    agent_session_id = agent.session_id
    logger.info("LlamaStack Agent session Id: %s", agent_session_id)

    st.set_page_config(
        page_title="RAG Chat",
        page_icon=":clipboard:",
        layout="wide",
        initial_sidebar_state="auto",
    )

    st.markdown('# RAG Chat')

    # Chat history management if not initialized
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for new messages
    prompt = st.chat_input(
      "Ask something or upload a PDF file to get started...",
      accept_file=True,
      file_type=["pdf"]
    )

    if prompt and prompt["files"]:
        uploaded_file = prompt["files"][0]
        # st.session_state.files = prompt["files"]
        st.info(f"File uploaded: {uploaded_file.name}", icon="📂")
        # if uploaded_file is not None:
        with st.spinner("Wait until the document is processed...", show_time=True):
            process_document(client, uploaded_file.name, BytesIO(uploaded_file.read()))
            st.success("Document processed!")

    if prompt and prompt.text:
        full_response = ""

        # Add user input to chat history
        st.session_state.messages.append({"role": "user", "content": prompt.text})
        with st.chat_message("user"):
            st.markdown(prompt.text)

        # Get response from LlamaStack API
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            response = agent.create_turn(
                messages=[
                    {
                        "role": "user",
                        "content": prompt.text,
                    }
                ],
                session_id=agent_session_id,
            )
            logger.info("\nAgent response: %s", response)

            for chunk in response:
                # logger.info("response chunk: %s", chunk)
                if chunk.event.payload.event_type == "step_progress":
                    if chunk.event.payload.delta.type == "text":
                        full_response += chunk.event.payload.delta.text
                        message_placeholder.markdown(full_response + "▌")
                
                if chunk.event.payload.event_type == "step_complete":
                    if chunk.event.payload.step_details:
                        step_details = chunk.event.payload.step_details
                        if hasattr(step_details, "violation") and step_details.violation:
                            violation = step_details.violation
                            logger.info("violation: %s", violation)
                            full_response = violation.metadata.get("violation_type", "") + " " + violation.user_message

                message_placeholder.markdown(full_response)
        
            st.session_state.messages.append({"role": "assistant", "content": full_response})

