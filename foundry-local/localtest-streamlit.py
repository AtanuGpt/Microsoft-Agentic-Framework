
import asyncio
import threading
import streamlit as st

from agent_framework.openai import OpenAIChatClient

MODEL = "<<your local model:version"
BASE_URL = "<<foundry local host url>>/v1"


st.set_page_config(
    page_title="Local AI Architect",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Local AI Architect")
st.caption("Powered by Microsoft Foundry Local + Microsoft Agent Framework")


# Create a persistent asyncio event loop
@st.cache_resource
def initialize_runtime():

    loop = asyncio.new_event_loop()

    def start_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(
        target=start_loop,
        daemon=True
    )

    thread.start()

    # Initialize MAF client and agent
    chat_client = OpenAIChatClient(
        base_url=BASE_URL,
        api_key="none",
        model=MODEL
    )

    agent = chat_client.as_agent(
        name="AzureArchitect",
        instructions="You are an experienced Azure Cloud Architect. "
                     "Provide technically detailed and structured responses."
    )

    return loop, agent


loop, agent = initialize_runtime()


# Execute agent asynchronously on the persistent event loop
def run_agent(prompt):

    future = asyncio.run_coroutine_threadsafe(
        agent.run(prompt),
        loop
    )

    return future.result()


# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display conversation history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
if prompt := st.chat_input("Ask your Azure or Gen AI question..."):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare conversation context
    conversation = "\n\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in st.session_state.messages
    )

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                response = run_agent(conversation)
                answer = response.text

            except Exception as e:
                answer = f"Error: {str(e)}"

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# Sidebar
with st.sidebar:

    st.subheader("Model Information")

    st.write("**Model:**", MODEL)
    st.write("**Runtime:** Microsoft Foundry Local")
    st.write("**Framework:** Microsoft Agent Framework")

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
