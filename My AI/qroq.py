import requests
import json
from dotenv import load_dotenv

load_dotenv()  # <-- THIS loads the .env file automatically
print(os.getenv("OPENAI_API_KEY"))


GROQ_API_KEY = GROQ_API_KEY

url = "https://api.groq.com/openai/v1/chat/completions"

payload = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {"role": "user", "content": "hi"}
    ]
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GROQ_API_KEY}"
}

response = requests.post(url, headers=headers, data=json.dumps(payload))

print("Status:", response.status_code)
print("Response:")
print(response.json())
