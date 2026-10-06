from dotenv import load_dotenv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

load_dotenv()

required_secret_names = (
    "GROQ_API_KEY",
    "SERPER_API_KEY",
    "HUGGINGFACEHUB_API_TOKEN",
)
secrets = {}
missing_secrets = []
for secret_name in required_secret_names:
    secret_value = os.getenv(secret_name)
    if not secret_value:
        try:
            secret_value = st.secrets.get(secret_name)
        except StreamlitSecretNotFoundError:
            secret_value = None
    if secret_value:
        secrets[secret_name] = secret_value
    else:
        missing_secrets.append(secret_name)

if missing_secrets:
    st.error(
        "Missing required secrets: "
        + ", ".join(missing_secrets)
        + ". Add them in your Streamlit app's Settings > Secrets."
    )
    st.stop()

GROQ_API_KEY = secrets["GROQ_API_KEY"]
SERPER_API_KEY = secrets["SERPER_API_KEY"]
HUGGINGFACEHUB_API_TOKEN = secrets["HUGGINGFACEHUB_API_TOKEN"]

# Load PDF files
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Vector database storage
from langchain_chroma import Chroma
# Create reusable prompts
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent 
from langgraph.checkpoint.memory import InMemorySaver


## data in st sessions 
# Check if document upload status is not stored in session
if "document_uploaded" not in st.session_state:
    # Initialize document upload status as False
    st.session_state.document_uploaded = False

# Check if agent object is not stored in session
if "agent" not in st.session_state:
    # Initialize agent as None (not created yet)
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None 

if "messages" not in st.session_state:
    st.session_state.messages = []


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".py", ".html",
    ".htm", ".xml", ".log", ".sql", ".csv", ".tsv", ".json",
}


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
    )


def load_uploaded_file(uploaded_file, temp_dir):
    file_name = Path(uploaded_file.name).name
    extension = Path(file_name).suffix.lower()
    file_path = Path(temp_dir) / file_name
    file_path.write_bytes(uploaded_file.getvalue())
    metadata = {"source": file_name}

    if extension == ".pdf":
        return PyMuPDFLoader(str(file_path)).load()

    if extension == ".docx":
        from docx import Document as WordDocument

        word_document = WordDocument(str(file_path))
        text = "\n".join(paragraph.text for paragraph in word_document.paragraphs)
        return [Document(page_content=text, metadata=metadata)]

    if extension == ".xlsx":
        from openpyxl import load_workbook

        workbook = load_workbook(file_path, read_only=True, data_only=True)
        sheets = []
        for worksheet in workbook.worksheets:
            rows = [
                "\t".join(str(value or "") for value in row)
                for row in worksheet.iter_rows(values_only=True)
            ]
            sheets.append(f"Sheet: {worksheet.title}\n" + "\n".join(rows))
        return [Document(page_content="\n\n".join(sheets), metadata=metadata)]

    if extension == ".pptx":
        from pptx import Presentation

        presentation = Presentation(str(file_path))
        slides = []
        for slide_number, slide in enumerate(presentation.slides, start=1):
            text = "\n".join(
                shape.text for shape in slide.shapes if hasattr(shape, "text")
            )
            slides.append(f"Slide {slide_number}\n{text}")
        return [Document(page_content="\n\n".join(slides), metadata=metadata)]

    if extension in {".txt", ".md", ".py", ".html", ".htm", ".xml", ".log", ".sql"}:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        return [Document(page_content=text, metadata=metadata)]

    if extension in {".csv", ".tsv", ".json"}:
        if extension == ".json":
            value = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
            text = json.dumps(value, indent=2, ensure_ascii=True)
        else:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        return [Document(page_content=text, metadata=metadata)]

    raise ValueError(f"Unsupported file type: {file_name}")


def process_document(uploaded_files):
    progress = st.progress(0, text="Reading uploaded files...")
    started_at = time.perf_counter()
    with tempfile.TemporaryDirectory() as temp_dir:
        with ThreadPoolExecutor(max_workers=min(8, len(uploaded_files))) as executor:
            file_docs = executor.map(
                lambda uploaded_file: load_uploaded_file(uploaded_file, temp_dir),
                uploaded_files,
            )
        docs = [document for documents in file_docs for document in documents]
    progress.progress(35, text=f"Read {len(docs)} document sections. Splitting text...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    splitted_docs = splitter.split_documents(docs)
    progress.progress(
        50,
        text=f"Created {len(splitted_docs):,} search chunks. Building embeddings...",
    )

    vector_store = InMemoryVectorStore.from_documents(
        documents = splitted_docs,
        embedding = get_embeddings()

    )
    progress.progress(90, text="Preparing the question-answering assistant...")

    from langchain_groq import ChatGroq
    llm = ChatGroq(model="openai/gpt-oss-20b")


    @tool
    def retriever_tool(query: str):

        """
        This tool helps retrieve relevant data from the PDF document.
        The PDF contains medical record details.
        """

        # Print message to confirm tool is called with query
        print("Query", query)

        # Search top 4 similar chunks from vector database
        docs = vector_store.similarity_search(query = query, k = 4)
        context = ""

        for doc in docs:

            # Append chunk text to context with spacing
            context += doc.page_content + "\n\n"

        print("Context: " , context)
        return context


    # Create a system prompt to guide LLM behavior
    System_Prompt = """
    You are a helpful assistant that answers questions using retrieved context.
    ALWAYS use the `retriever_tool` tool for questions requiring external knowledge.
    """

    memory = InMemorySaver()

    # Create agent using LLM, tools, and system prompt
    agent = create_agent(
        model = llm,
        tools = [retriever_tool],
        system_prompt = System_Prompt, 
        checkpointer = memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True
    progress.progress(100, text=f"Ready in {time.perf_counter() - started_at:.1f} seconds.")

### upload ui 
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(
        label="Select files",
        type=None,
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, XLSX, PPTX, TXT, Markdown, CSV, JSON, HTML, XML, SQL, and Python files.",
    )
    if uploaded:
        with st.spinner("Processing......"):
            supported_files = []
            unsupported_files = []
            for file in uploaded:
                if Path(file.name).suffix.lower() in SUPPORTED_EXTENSIONS:
                    supported_files.append(file)
                else:
                    unsupported_files.append(file.name)

            if unsupported_files:
                st.warning("Skipped unsupported files: " + ", ".join(unsupported_files))
            if supported_files:
                process_document(supported_files)
                st.rerun()
            else:
                st.error("No supported files were uploaded.")


### chat_ui
if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask anything related to uploaded documents....")
    if query:
        st.session_state.messages.append({"role":"user", "content":query})

        st.chat_message("user").markdown(query)

        response = st.session_state.agent.invoke(
            {"messages":[{"role":"user", "content":query}]},
            {"configurable":{"thread_id":1}}
        )

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai", "content":answer})

