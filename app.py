import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

app = FastAPI()

# Enable CORS for local HTML files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Custom HR/PIP tools
@tool
def get_pip_guidelines() -> str:
    """Returns official guidelines for Performance Improvement Plans."""
    return (
        "Standard PIP Guidelines:\n"
        "1. Define specific, measurable performance goals.\n"
        "2. Set clear timelines (typically 30, 60, or 90 days).\n"
        "3. Schedule regular check-in meetings (weekly/bi-weekly).\n"
        "4. Provide required resources, training, and support.\n"
        "5. Clearly outline outcomes if expectations are or aren't met."
    )

# 2. Hardcode Groq key to prevent .env loading errors
# REPLACE "gsk_..." BELOW WITH YOUR REAL GROQ API KEY FROM CONSOLE.GROQ.COM
groq_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key)

# 3. Create Agent
tools = [get_pip_guidelines]
agent_executor = create_react_agent(llm, tools)

# 4. FastAPI Route
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response = agent_executor.invoke({"messages": [("user", request.message)]})
    bot_reply = response["messages"][-1].content
    return {"reply": bot_reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)