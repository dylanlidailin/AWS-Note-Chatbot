import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import uvicorn

# --- LangChain Imports (All necessary) ---
from langchain_aws import ChatBedrock
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from langchain.chains.retrieval_qa import RetrievalQA  # <-- Re-added
from langchain.prompts import PromptTemplate           # <-- Re-added
from langchain.agents import AgentType, initialize_agent # <-- Re-added
from langchain.memory import ConversationBufferMemory    # <-- Re-added
# --- End of LangChain Imports ---

SESSION_STORE = {}
load_dotenv()
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")

search = SerpAPIWrapper(serpapi_api_key=SERP_API_KEY)

search_tool = Tool(
    name="SerpAPI Search",
    func=search.run,
    description="Useful for answering questions about current events or real-time web data"
)

# Load LLM
llm = ChatBedrock(
    model_id="anthropic.claude-3-7-sonnet-20240715-v1:0",
    region_name="us-east-1",
    model_kwargs={"temperature": 0.1}
)

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_file = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_file.read_text())

# === PDF QA Tool ===
def build_chain_from_pdf(path: str) -> RetrievalQA:
    """Build a RetrievalQA chain to be used as a tool."""
    loader = PyPDFLoader(path)
    docs = loader.load_and_split()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    pages = splitter.split_documents(docs)

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name="us-east-1"
    )
    index = FAISS.from_documents(pages, embeddings)
    retriever = index.as_retriever()

    # We use a simple chain_type, as the agent will handle the prompting
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename).suffix or ".pdf"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            path = tmp.name

        # 1. Build the QA chain for the PDF
        qa_chain = build_chain_from_pdf(path)
        if qa_chain is None:
            raise HTTPException(status_code=400, detail="Unable to build QA chain")

        # 2. Create a specific tool for this PDF
        pdf_tool = Tool(
            name="PDF Search",
            func=qa_chain.run, # Use the chain's run method
            description=f"Useful for answering questions about the specific content of the uploaded PDF: {file.filename}"
        )
        
        # 3. Define the list of tools the agent can use
        tools = [search_tool, pdf_tool]
        
        # 4. Create a new memory for this session
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # 5. Initialize the agent
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True, # Set to True to see the agent's thoughts in your terminal
            memory=memory
        )

        # 6. Store the agent, not the retriever
        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = agent

        return {
            "status": "Uploaded",
            "filename": file.filename,
            "session_id": session_id
        }

    except Exception as e:
        print("[PDF Processing Error]", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# === Ask endpoint ===
class Query(BaseModel):
    question: str
    session_id: str

@app.post("/ask")
async def ask(q: Query):
    # 1. Get the agent from the session store
    agent = SESSION_STORE.get(q.session_id)

    if agent is None:
        return {"answer": "Session not found or expired."}

    try:
        # 2. Run the agent with the question
        # The agent will decide to use the 'SerpAPI Search' or 'PDF Search' tool
        response = agent.run(q.question)
        answer = response
        
    except Exception as e:
        answer = f"Error: {str(e)}"

    return {"answer": answer}

@app.get("/healthz")
async def health():
    return {"status": "ok"}

# Main entry point for uvicorn (if you run 'python api.py')
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)