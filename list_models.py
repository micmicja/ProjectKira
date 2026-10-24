import requests

api_key = "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"
url = "https://api.groq.com/openai/v1/models"
headers = {"Authorization": f"Bearer {api_key}"}

try:
    response = requests.get(url, headers=headers)
    data = response.json()
    
    print("🟢 Active Groq Models mo ngayon:")
    for model in data.get('data', []):
        print(f"- {model['id']}")
except Exception as e:
    print(f"Error: {e}")