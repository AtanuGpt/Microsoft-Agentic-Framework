from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:12434/engines/llama.cpp/v1",
    api_key="not-needed"
)

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

print(response.choices[0].message.content)
