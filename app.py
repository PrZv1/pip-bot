import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

load_dotenv()

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Internal PIP Tool
@tool
def get_pip_guidelines() -> str:
    """Returns official internal guidelines for Performance Improvement Plans."""
    return (
        "Standard PIP Guidelines:\n"
        "1. Define specific, measurable performance goals.\n"
        "2. Set clear timelines (typically 30, 60, or 90 days).\n"
        "3. Schedule regular check-in meetings (weekly/bi-weekly).\n"
        "4. Provide required resources, training, and support.\n"
        "5. Clearly outline outcomes if expectations are or aren't met."
    )

# 2. Web Search Tool (DuckDuckGo)
search_tool = DuckDuckGoSearchRun()

# Combine both tools
tools = [get_pip_guidelines, search_tool]

# 3. LLM and Agent Setup
# The system prompt instructs the agent when to use web search
system_prompt = (
    "You are an expert HR Performance Improvement Plan (PIP) Assistant specializing in Philippine HR practices and DOLE labor standards. "
    "Use your search tool to reference up-to-date Philippine Labor Code regulations, DOLE guidelines, or specific regional laws when answering."
)

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
agent = create_react_agent(llm, tools, state_modifier=system_prompt)

# 4. Request Data Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []

# 5. Endpoint
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        formatted_messages = []
        for msg in request.history:
            role = "user" if msg.role == "user" else "assistant"
            formatted_messages.append((role, msg.content))
            
        if not formatted_messages or formatted_messages[-1][1] != request.message:
            formatted_messages.append(("user", request.message))

        response = agent.invoke({"messages": formatted_messages})
        bot_reply = response["messages"][-1].content
        return {"reply": bot_reply}

    except Exception as e:
        return {"reply": f"An error occurred: {str(e)}"}
