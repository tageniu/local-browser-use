import requests
import json

# Configuration
payload = {
    "model": "gpt-oss:20b",
    "messages": [
        {
            "role": "user",
            "content": "What is 2+2? Think step by step, then give me just the answer."
        }
    ],
    "stream": True
}

# Stream and output tokens
r = requests.post("http://localhost:11434/api/chat", json=payload, stream=True, timeout=120)
for line in r.iter_lines():
    if line:
        data = json.loads(line.decode())
        
        if 'message' in data:
            message = data['message']
            content = message.get('content', '')
            thinking_content = message.get('thinking', '')
            
            # Output thinking tokens
            if thinking_content:
                print(thinking_content, end='', flush=True)
            
            # Output response tokens
            if content:
                print(content, end='', flush=True)
