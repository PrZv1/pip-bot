import os
import glob
import io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from duckduckgo_search import DDGS
from langgraph.prebuilt import create_react_agent
import pypdf

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to extract text from a PDF stream
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text if text.strip() else "[PDF contained no extractable text]"
    except Exception as e:
        return f"[Error reading PDF: {str(e)}]"

# 1. KNOWLEDGE BASE LOADER (Reads both PDF and TXT files in /docs)
def load_docs_knowledge_base() -> str:
    docs_text = ""
    docs_dir = "docs"
    if os.path.exists(docs_dir):
        files = glob.glob(os.path.join(docs_dir, "*"))
        for file_path in files:
            filename = os.path.basename(file_path)
            if file_path.endswith(".txt") or file_path.endswith(".md"):
                with open(file_path, "r", encoding="utf-8") as f:
                    docs_text += f"\n--- FILE: {filename} ---\n" + f.read() + "\n"
            elif file_path.endswith(".pdf"):
                with open(file_path, "rb") as f:
                    pdf_bytes = f.read()
                    extracted = extract_text_from_pdf_bytes(pdf_bytes)
                    docs_text += f"\n--- FILE: {filename} ---\n" + extracted + "\n"
    return docs_text if docs_text else "No official template files currently stored in docs/ directory."

# 2. TOOLS SETUP
class SearchInput(BaseModel):
    query: str = Field(description="The search query string.")

@tool("web_search", args_schema=SearchInput)
def web_search(query: str) -> str:
    """Searches the web for up-to-date Philippine labor laws, DOLE guidelines, or HR regulations."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No relevant search results found."
        formatted_results = [f"Title: {r['title']}\nSnippet: {r['body']}\n" for r in results]
        return "\n---\n".join(formatted_results)
    except Exception as e:
        return f"Search error: {str(e)}"

@tool("get_internal_form_templates")
def get_internal_form_templates() -> str:
    """Retrieves official company PIP form templates, PDFs, and internal HR documentation stored in the knowledge base."""
    return load_docs_knowledge_base()

tools = [web_search, get_internal_form_templates]

# 3. SYSTEM PROMPT
system_prompt = (
    "You are an expert HR Performance Improvement Plan (PIP) Assistant specializing in Philippine HR practices and DOLE labor standards.\n\n"
    "Follow this multi-stage PIP agent workflow whenever processing performance data or drafting plans:\n"
    "1. RISK DETECTION: Analyze performance gaps, KPI drops, or attendance deficiencies.\n"
    "2. COACHING RECOMMENDATION: Suggest pre-PIP 1-on-1 coaching actions.\n"
    "3. DOCUMENTATION VALIDATOR: Check against internal templates/PDFs and past coaching notes.\n"
    "4. PIP ELIGIBILITY & DOLE COMPLIANCE: Ensure PIP meets DOLE due process guidelines (twin-notice rule).\n"
    "5. PIP BUILDER: Auto-fill the official PIP form template from the knowledge base with exact employee metrics.\n"
    "6. TIMELINE & SUCCESS COACHING: Provide weekly check-in schedules and guidance.\n\n"
    "When calling tools, output clean arguments. Always structure answers professionally using clean Markdown formatting."
)

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
agent = create_react_agent(llm, tools, prompt=system_prompt)

# 4. CHAT ENDPOINT (Supports TXT & PDF Uploads)
@app.post("/chat")
async def chat_endpoint(
    message: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    try:
        user_content = message
        
        # If user uploaded a file (PDF or text)
        if file:
            file_bytes = await file.read()
            if file.filename.lower().endswith(".pdf"):
                extracted_file_text = extract_text_from_pdf_bytes(file_bytes)
            else:
                extracted_file_text = file_bytes.decode("utf-8", errors="ignore")
                
            user_content += f"\n\n[ATTACHED FILE CONTENT ({file.filename})]:\n{extracted_file_text}"

        messages = [("user", user_content)]

        try:
            response = agent.invoke({"messages": messages})
            bot_reply = response["messages"][-1].content
            return {"reply": bot_reply}
        except Exception:
            fallback = llm.invoke(messages)
            return {"reply": fallback.content}

    except Exception as e:
        return {"reply": f"An error occurred while processing: {str(e)}"}
