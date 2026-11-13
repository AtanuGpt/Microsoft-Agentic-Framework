
from openai import OpenAI
import time

client = OpenAI(
    base_url="http://127.0.0.1:56472/v1",
    api_key="not-needed"
)

MODEL = "qwen2.5-0.5b-instruct-generic-cpu:4"

messages = [
    {
        "role": "system",
        "content": "You are an Azure Cloud Architect."
    },
    {
        "role": "user",
        "content": "Explain Azure Landing Zones and their architecture."
    }
]

start_time = time.perf_counter()
first_token_time = None

stream = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.2,
    max_tokens=500,
    stream=True
)

print("\nAssistant:\n")

for chunk in stream:
    if not chunk.choices:
        continue

    content = chunk.choices[0].delta.content

    if content:
        if first_token_time is None:
            first_token_time = time.perf_counter()

        print(content, end="", flush=True)

end_time = time.perf_counter()

print("\n\n--- Performance Metrics ---")

print(f"Time to first token: "
      f"{first_token_time - start_time:.3f} seconds"
      if first_token_time else "No tokens received")

print(f"Total response time: {end_time - start_time:.3f} seconds")
