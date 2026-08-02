import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

app = FastAPI()

# Enable CORS for all frontend origins
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

tools = [get_pip_guidelines]

# 2. Set up LLM and Agent
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
agent = create_react_agent(llm, tools)

# 3. Request Models for Chat History
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []

# 4. Chat Endpoint (PASTE STEP 2 HERE)
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Build prompt using recent history
        formatted_messages = []
        for msg in request.history:
            # Map frontend roles to LangChain roles ('user' / 'assistant')
            role = "user" if msg.role == "user" else "assistant"
            formatted_messages.append((role, msg.content))
            
        # Add latest user message if not already in history
        if not formatted_messages or formatted_messages[-1][1] != request.message:
            formatted_messages.append(("user", request.message))

        # Run agent with history context
        response = agent.invoke({"messages": formatted_messages})
        
        # Get last message response
        bot_reply = response["messages"][-1].content
        return {"reply": bot_reply}

    except Exception as e:
        return {"reply": f"An error occurred: {str(e)}"}
