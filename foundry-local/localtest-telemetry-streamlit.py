import asyncio
import streamlit as st
import pandas as pd
import plotly.express as px

from agent_framework.openai import OpenAIChatClient
from agent_framework.observability import enable_instrumentation

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL = "Your model:version"
BASE_URL = "url:port/v1" #foundry server status will give you the endpoint


# --------------------------------------------------
# OPEN TELEMETRY INITIALIZATION
# --------------------------------------------------

@st.cache_resource
def initialize_telemetry():

    exporter = InMemorySpanExporter()

    provider = TracerProvider(
        resource=Resource.create({
            "service.name": "FoundryLocal-AzureArchitect",
            "deployment.environment.name": "local"
        })
    )

    provider.add_span_processor(
        SimpleSpanProcessor(exporter)
    )

    trace.set_tracer_provider(provider)

    enable_instrumentation(
        enable_sensitive_data=False
    )

    return exporter


exporter = initialize_telemetry()


# --------------------------------------------------
# AGENT EXECUTION
# --------------------------------------------------

async def execute_agent(question):

    chat_client = OpenAIChatClient(
        base_url=BASE_URL,
        api_key="none",
        model=MODEL
    )

    agent = chat_client.as_agent(
        name="AzureArchitect",
        instructions="You are an experienced Azure Cloud Architect."
    )

    response = await agent.run(question)

    return response.text


# --------------------------------------------------
# TELEMETRY EXTRACTION
# --------------------------------------------------

def get_telemetry():

    spans = exporter.get_finished_spans()

    records = []

    for span in spans:

        attrs = dict(span.attributes or {})

        duration = (
            span.end_time - span.start_time
        ) / 1_000_000_000

        context = span.context
        parent = span.parent

        records.append({
            "trace_id": (
                format(context.trace_id, "032x")
                if context else ""
            ),
            "span_id": (
                format(context.span_id, "016x")
                if context else ""
            ),
            "parent_id": (
                format(parent.span_id, "016x")
                if parent else "ROOT"
            ),
            "span_name": span.name,
            "operation": attrs.get(
                "gen_ai.operation.name", "other"
            ),
            "model": attrs.get(
                "gen_ai.response.model",
                attrs.get("gen_ai.request.model", "N/A")
            ),
            "input_tokens": attrs.get(
                "gen_ai.usage.input_tokens", 0
            ),
            "output_tokens": attrs.get(
                "gen_ai.usage.output_tokens", 0
            ),
            "duration": round(duration, 3),
            "status": span.status.status_code.name,
            "start": pd.to_datetime(
                span.start_time, unit="ns", utc=True
            ),
            "end": pd.to_datetime(
                span.end_time, unit="ns", utc=True
            )
        })

    return pd.DataFrame(records)


# --------------------------------------------------
# STREAMLIT DASHBOARD
# --------------------------------------------------

st.set_page_config(
    page_title="Foundry Local Observability",
    page_icon="📊",
    layout="wide"
)

st.title("Foundry Local - AI Observability")

st.caption(
    "Microsoft Agent Framework | OpenTelemetry | Local Inference"
)

st.divider()

# Agent execution section

st.subheader("Agent Execution")

question = st.text_area(
    "Enter your prompt",
    value="Explain Azure Landing Zones and their key design principles."
)

if st.button("Run Agent", type="primary"):

    with st.spinner("Executing Foundry Local agent..."):

        try:
            answer = asyncio.run(execute_agent(question))

            st.session_state["last_answer"] = answer

            st.success("Agent execution completed")

        except Exception as ex:
            st.error(f"Agent execution failed: {ex}")


if "last_answer" in st.session_state:

    with st.expander("View Agent Response", expanded=True):
        st.write(st.session_state["last_answer"])


st.divider()

# Refresh telemetry view

if st.button("Refresh Telemetry"):
    st.rerun()

df = get_telemetry()


if not df.empty:

    # Separate model spans from agent spans
    model_df = df[df["operation"] == "chat"]

    agent_df = df[
        df["operation"] == "invoke_agent"
    ]

    total_input = int(model_df["input_tokens"].sum())
    total_output = int(model_df["output_tokens"].sum())

    avg_latency = (
        model_df["duration"].mean()
        if not model_df.empty else 0
    )

    agent_count = len(agent_df)

    failures = len(
        agent_df[agent_df["status"] == "ERROR"]
    )

    # KPI section

    st.subheader("Performance Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Agent Executions", agent_count)
    c2.metric("Model Latency", f"{avg_latency:.2f}s")
    c3.metric("Input Tokens", total_input)
    c4.metric("Output Tokens", total_output)

    st.metric("Failed Agent Executions", failures)

    st.divider()

    # Token analytics

    st.subheader("Token Consumption")

    token_df = pd.DataFrame({
        "Token Type": ["Input", "Output"],
        "Tokens": [total_input, total_output]
    })

    fig_tokens = px.bar(
        token_df,
        x="Token Type",
        y="Tokens",
        color="Token Type",
        text="Tokens",
        title="Input vs Output Tokens"
    )

    st.plotly_chart(
        fig_tokens,
        use_container_width=True
    )

    # Trace waterfall

    st.subheader("Execution Trace")

    fig_trace = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="span_name",
        color="operation",
        hover_data=[
            "model",
            "duration",
            "input_tokens",
            "output_tokens",
            "status"
        ],
        title="Agent and Model Execution Timeline"
    )

    fig_trace.update_yaxes(autorange="reversed")

    st.plotly_chart(
        fig_trace,
        use_container_width=True
    )

    # Detailed telemetry

    st.subheader("Span Details")

    display_df = df[[
        "trace_id",
        "span_name",
        "operation",
        "model",
        "duration",
        "input_tokens",
        "output_tokens",
        "status"
    ]]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No telemetry available yet. Run your agent to generate traces."
    )

st.caption(
    "Telemetry is retained in memory and will reset when the app process restarts."
)
