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

# --- LangChain Imports ---
from langchain_classic.agents import AgentExecutor
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
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
def build_chain_from_pdf(path: str):
    """Build a retrieval chain to be used as a tool."""
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

    # Modern chain construction
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Context: {context}"
    )
    prompt = PromptTemplate.from_template(system_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename).suffix or ".pdf"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            path = tmp.name

        retrieval_chain = build_chain_from_pdf(path)
        if retrieval_chain is None:
            raise HTTPException(status_code=400, detail="Unable to build retrieval chain")

        def retrieval_chain_func(query: str) -> str:
            """A wrapper function to invoke the chain with a query."""
            return retrieval_chain.invoke({"input": query})

        pdf_tool = Tool(
            name="PDF Search",
            func=retrieval_chain_func,
            description=f"Useful for answering questions about the specific content of the uploaded PDF: {file.filename}"
        )
        
        tools = [search_tool, pdf_tool]
        
        # Modern agent construction
        prompt = PromptTemplate.from_template(
            """
            You are a helpful assistant. Answer the user's questions to the best of your ability.
            TOOLS:
            ------
            You have access to the following tools:
            {tools}
            To use a tool, please use the following format:
            ```
            Thought: Do I need to use a tool? Yes
            Action: The action to take. Should be one of [{tool_names}]
            Action Input: The input to the action
            Observation: The result of the action
            ```
            When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
            ```
            Thought: Do I need to use a tool? No
            Final Answer: [your response here]
            ```
            Begin!
            Previous conversation history:
            {chat_history}
            New input: {input}
            {agent_scratchpad}
            """
        )

        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = agent_executor

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
    agent_executor = SESSION_STORE.get(q.session_id)

    if agent_executor is None:
        return {"answer": "Session not found or expired."}

    try:
        response = agent_executor.invoke({
            "input": q.question,
            "chat_history": []
        })
        answer = response.get("output", "No answer found.")
        
    except Exception as e:
        answer = f"Error: {str(e)}"

    return {"answer": answer}

@app.get("/healthz")
async def health():
    return {"status": "ok"}

# Main entry point for uvicorn (if you run 'python api.py')
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)