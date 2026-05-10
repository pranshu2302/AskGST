from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Get one at aistudio.google.com.")

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
        max_output_tokens=4096,
        thinking_budget=0,
    )

if __name__ == "__main__":
    llm = get_llm()
    response = llm.invoke("What is the capital of France? Answer in one sentence.")
    print("Response:", response.content)
