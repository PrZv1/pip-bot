import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from duckduckgo_search import DDGS
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

# 1. Pydantic Input Schema for Web Search
class SearchInput(BaseModel):
    query: str = Field(description="The web search query string.")

# 2. Custom Web Search Tool
@tool("web_search", args_schema=SearchInput)
def web_search(query: str) -> str:
    """Searches the web for up-to-date Philippine labor laws, DOLE guidelines, or HR regulations."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No relevant search results found."
        
        formatted_results = []
        for r in results:
            formatted_results.append(f"Title: {r['title']}\nSnippet: {r['body']}\n")
        return "\n---\n".join(formatted_results)
    except Exception as e:
        return f"Search error: {str(e)}"

# 3. Internal PIP Tool
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

tools = [get_pip_guidelines, web_search]

# 4. System Prompt & Agent Setup
system_prompt = (
    "You are an expert HR Performance Improvement Plan (PIP) Assistant specializing in Philippine HR practices and DOLE labor standards.\n"
    "When calling tools, provide clean JSON arguments without special text wrappers or custom function syntax.\n"
    "If you ask a search query or answer directly, keep responses accurate, clear, and professional."
)

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
agent = create_react_agent(llm, tools, prompt=system_prompt)

# 5. Request Data Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []

# 6. Endpoint with Fallback Safety
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    formatted_messages = []
    for msg in request.history:
        role = "user" if msg.role == "user" else "assistant"
        formatted_messages.append((role, msg.content))
        
    if not formatted_messages or formatted_messages[-1][1] != request.message:
        formatted_messages.append(("user", request.message))

    try:
        # Primary Attempt: Execute with tools enabled
        response = agent.invoke({"messages": formatted_messages})
        bot_reply = response["messages"][-1].content
        return {"reply": bot_reply}

    except Exception as err:
        # Fallback Attempt: If Groq tool calling fails, answer directly using LLM knowledge base
        try:
            fallback_response = llm.invoke(formatted_messages)
            return {"reply": fallback_response.content}
        except Exception:
            return {"reply": "I encountered an issue processing your request. Please try rephrasing your question."}
