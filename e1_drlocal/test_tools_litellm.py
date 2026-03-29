from litellm import completion

tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
}]
models = ["ollama/nemotron-3-nano:4b", "ollama/qwen3.5:0.8b", "ollama/gpt-oss:20b"]

for model in models:
    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": "What's the weather like in Boston today?"}],
            tools=tools
        )
        print(f"{model} SUCCESS")
    except Exception as e:
        print(f"{model} FAILED: {str(e)[:100]}")
