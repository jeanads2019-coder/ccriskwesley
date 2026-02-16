import requests
import os

# Try to get Groq API Key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini" # Local
GROQ_MODEL = "llama3-8b-8192" # Cloud fallback

def call_llm(prompt: str) -> str:
    # 1. Try Groq (Cloud Friendly)
    if GROQ_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "model": GROQ_MODEL,
                "temperature": 0.2
            }
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Erro na API Groq: {str(e)}"

    # 2. Try Local Ollama (Desktop Friendly)
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 500,
                "temperature": 0.2
            }
        }
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        return "⚠️ O modelo de IA local (Ollama) não foi detectado. Para usar em nuvem (Vercel/Streamlit Cloud), configure a chave da API GROQ no arquivo .env ou secrets."
    except Exception as e:
        return f"Erro ao chamar LLM: {str(e)}"