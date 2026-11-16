# Run Local LLMs on Windows Using Docker Model Runner

## Overview

Running Large Language Models (LLMs) locally no longer requires a complex inference infrastructure setup.

With **Docker Model Runner**, you can pull, manage, and execute open-source AI models directly on your Windows machine and expose them through an OpenAI-compatible API.

This guide walks through setting up Docker Model Runner, downloading Meta's Llama 3.2 model, running it locally, and integrating it with Python using the OpenAI SDK.

The objective is to establish a completely local LLM inference environment without depending on cloud-hosted AI endpoints.

## Architecture

The following illustrates the high-level architecture:

```text
+--------------------------------------+
|          Python Application          |
|                                      |
|             OpenAI SDK               |
+------------------+-------------------+
                   |
                   | HTTP REST API
                   |
                   v
+--------------------------------------+
|          Docker Model Runner         |
|                                      |
|             llama.cpp                |
|            Inference Engine          |
+------------------+-------------------+
                   |
                   | Model Inference
                   |
                   v
+--------------------------------------+
|             Llama 3.2                |
|                                      |
|          Local Model Weights         |
+--------------------------------------+
```

### Execution Flow

1. Python application sends a chat completion request using the OpenAI SDK.
2. Docker Model Runner receives the HTTP request.
3. The llama.cpp inference engine processes the prompt.
4. Llama 3.2 generates the response.
5. The generated response is returned to the Python application through the API.

## Prerequisites

- Windows 10/11 or a supported Windows Server environment.
- Docker Desktop for Windows.
- Python 3.10 or later.
- Internet connectivity for the initial model download.

### Hardware Configuration

The following hardware configuration was used for this implementation:

| Component | Configuration |
|---|---|
| Operating System | Windows |
| RAM | 16 GB |
| GPU | No dedicated GPU |
| Inference | CPU-based |
| Runtime | Docker Model Runner |
| Model | Meta Llama 3.2 |

> **Note:** Actual inference performance depends on the model variant, quantization, CPU capabilities, context length, and available memory.

---

## Step 1: Install Docker Desktop

Download and install Docker Desktop for Windows.

**Installation Guide:**

https://docs.docker.com/desktop/setup/install/windows-install/

Once installed, launch Docker Desktop and ensure that the Docker Engine is running.

Verify the installation:

```powershell
docker --version
```

## Step 2: Enable Docker Model Runner

Docker Model Runner is integrated with Docker Desktop and allows you to run AI models locally.

Follow these steps:

1. Open Docker Desktop.
2. Navigate to **Settings**.
3. Select **AI**.
4. Enable **Docker Model Runner**.
5. Enable **Host-side TCP support**.
6. Configure the TCP port as `12434`.

Host-side TCP support allows applications running directly on your Windows machine to communicate with the local model inference endpoint.

Verify Model Runner:

```powershell
docker model --help
```

## Step 3: Download the Llama 3.2 Model

Docker Model Runner provides access to AI models through Docker Hub.

Pull Meta's Llama 3.2 model:

```powershell
docker model pull ai/llama3.2
```

This downloads the model artifacts and stores them locally.

Depending on the model variant and quantization, the download size and memory requirements will vary.

## Step 4: Verify the Downloaded Model

List the models available locally:

```powershell
docker model list
```

You should see the downloaded Llama 3.2 model in the output.

## Step 5: Run the Model Locally

Start an interactive chat session:

```powershell
docker model run ai/llama3.2
```

You can now interact with the LLM directly from your terminal.

For example:

```text
What is Generative AI?
```

The model processes the prompt locally and generates a response.

At this point, you have successfully executed an LLM using Docker Model Runner without relying on an external inference API.

## Step 6: Integrate with Python Using OpenAI SDK

One of the interesting capabilities of Docker Model Runner is its OpenAI-compatible API.

This means applications already using the OpenAI Python SDK can communicate with locally hosted models with minimal code changes.

### Install the OpenAI Python SDK

```powershell
pip install openai
```

### Configure the Local Inference Endpoint

Docker Model Runner exposes an OpenAI-compatible API.

**Base URL:**

```text
http://localhost:12434/engines/llama.cpp/v1
```

The `llama.cpp` engine is responsible for executing the model inference workload.

### Python Implementation

Create a Python file named `local_llm.py`:

```python
from openai import OpenAI

# Connect to local Docker Model Runner
client = OpenAI(
    base_url="http://localhost:12434/engines/llama.cpp/v1",
    api_key="not-needed"
)

# Execute a chat completion
response = client.chat.completions.create(
    model="ai/llama3.2",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": "What is agentic AI? Explain in 2 bullet points"
        }
    ],
    temperature=0.3
)

# Print the generated response
print(response.choices[0].message.content)
```

Execute the application:

```powershell
python local_llm.py
```

You should now receive the model-generated response directly in your Python console.

### Understanding the API Integration

The integration follows the standard OpenAI Chat Completions interface.

| Parameter | Description |
|---|---|
| `base_url` | Local Docker Model Runner endpoint |
| `api_key` | Placeholder; local Model Runner does not require an API key |
| `model` | Docker model identifier |
| `messages` | System and user conversation messages |
| `temperature` | Controls response sampling |

The important takeaway is that you can reuse the OpenAI SDK interface while switching the inference backend from a cloud-hosted model to a locally hosted model.

---

## Step 7: Test the API Using PowerShell

You can also invoke the local inference endpoint directly without writing Python code.

This is useful for validating the API connectivity and troubleshooting inference issues.

### Prepare the request

```powershell
$body = @{
    model = "ai/llama3.2"
    messages = @(
        @{
            role = "system"
            content = "You are a helpful AI assistant."
        },
        @{
            role = "user"
            content = "Explain Microsoft Agent Framework."
        }
    )
    temperature = 0.3
    max_tokens = 300
} | ConvertTo-Json -Depth 10
```

### Invoke the API

```powershell
$response = Invoke-RestMethod `
    -Uri "http://127.0.0.1:12434/engines/llama.cpp/v1/chat/completions" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

### Extract the generated response

```powershell
$response.choices[0].message.content
```

You can also inspect the complete response:

```powershell
$response | ConvertTo-Json -Depth 20
```

The response includes generated content, token usage, model metadata, and inference timing information.

---

## Troubleshooting

### 1. Connection Refused

If Python cannot connect to the inference endpoint:

- Verify Docker Desktop is running.
- Verify Docker Model Runner is enabled.
- Ensure host-side TCP support is enabled.
- Confirm that port `12434` is configured.
- Verify that the endpoint is accessible from the host machine.

### 2. Model Not Found

Verify the model is available locally:

```powershell
docker model list
```

If it is missing, pull it again:

```powershell
docker model pull ai/llama3.2
```

### 3. No Response or Slow Inference

Local inference performance depends on available CPU resources, RAM, model size, quantization, and context length.

For CPU-only environments, consider smaller quantized model variants.

### 4. Response Truncated

If the generated response is incomplete, check the configured `max_tokens` value and the response's `finish_reason`.

---

## Conclusion

Docker Model Runner provides a straightforward way to establish a local LLM inference environment without introducing a separate model-serving infrastructure.

The OpenAI-compatible API is particularly useful because existing Python applications can integrate with locally hosted models with minimal changes.

This also creates an interesting foundation for building local AI applications, RAG pipelines, and agentic workflows using frameworks such as **Microsoft Agent Framework (MAF)**.

Instead of sending every inference request to a cloud-hosted LLM, you can now experiment with locally hosted models and build AI applications within your own infrastructure.

## References

- [Docker Desktop Installation](https://docs.docker.com/desktop/setup/install/windows-install/)
- [Docker Model Runner – Getting Started](https://docs.docker.com/ai/model-runner/get-started/)
- [Docker Model Runner – API Reference](https://docs.docker.com/ai/model-runner/api-reference/)
- [Docker Hub – AI Models](https://hub.docker.com/u/ai)
