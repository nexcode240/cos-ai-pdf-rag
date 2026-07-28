"""LangGraph RAG flow: analyze → retrieve → grade → answer."""
from typing import Literal, TypedDict

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from ..config import settings

INSUFFICIENT_INFORMATION = (
    "Insufficient information to answer the question based on the available documents."
)


class RAGState(TypedDict):
    """Shared state for the RAG graph."""

    question: str
    model: str
    pdfs: list[dict[str, str]]
    search_queries: list[str]
    documents: list[Document]
    documents_relevant: bool
    answer: str
    sources: list[dict[str, str | int | None]]
    reasoning_steps: list[str]


def _append_step(state: RAGState, message: str) -> list[str]:
    """Append a reasoning step and return the updated list."""
    steps = list(state.get("reasoning_steps") or [])
    steps.append(message)
    return steps


def analyze_question(state: RAGState) -> RAGState:
    """Analyze the user question and produce retrieval-oriented queries."""
    llm = ChatOllama(model=state["model"])
    prompt = ChatPromptTemplate.from_template(
        """Analyze the user question for document retrieval.
Rewrite it into 2 focused search queries that capture the key entities,
intent, and likely document wording. Return only the queries, one per line.

Question: {question}
"""
    )
    raw = (prompt | llm | StrOutputParser()).invoke({"question": state["question"]})
    queries = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
    if not queries:
        queries = [state["question"]]

    return {
        **state,
        "search_queries": queries[:3],
        "reasoning_steps": _append_step(
            state,
            f"Analyzed question into search queries: {', '.join(queries[:3])}",
        ),
    }


def retrieve_documents(state: RAGState) -> RAGState:
    """Retrieve candidate chunks from each selected PDF collection."""
    pdfs = state.get("pdfs") or []
    if not pdfs:
        return {
            **state,
            "documents": [],
            "reasoning_steps": _append_step(state, "No PDFs found to query."),
        }

    llm = ChatOllama(model=state["model"])
    embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 2
different versions of the given user question to retrieve relevant documents from
a vector database. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    all_docs: list[Document] = []
    steps = list(state.get("reasoning_steps") or [])
    steps.append(
        f"Retrieving documents from {len(pdfs)} PDF(s): "
        f"{', '.join(pdf['name'] for pdf in pdfs)}"
    )

    search_question = (
        state["search_queries"][0]
        if state.get("search_queries")
        else state["question"]
    )

    for pdf in pdfs:
        vector_db = Chroma(
            persist_directory=settings.VECTOR_DB_DIR,
            embedding_function=embeddings,
            collection_name=pdf["collection_name"],
        )
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(search_kwargs={"k": 3}),
            llm,
            prompt=query_prompt,
        )
        try:
            docs = retriever.invoke(search_question)
            for doc in docs:
                doc.metadata.setdefault("pdf_name", pdf["name"])
                doc.metadata.setdefault("pdf_id", pdf["pdf_id"])
            all_docs.extend(docs)
            steps.append(f"Found {len(docs)} chunks in {pdf['name']}")
        except (RuntimeError, ValueError, OSError) as exc:
            steps.append(f"Error retrieving from {pdf['name']}: {exc}")

    steps.append(f"Total chunks retrieved: {len(all_docs)}")
    return {**state, "documents": all_docs[:10], "reasoning_steps": steps}


def grade_documents(state: RAGState) -> RAGState:
    """Decide whether retrieved documents are relevant to the question."""
    documents = state.get("documents") or []
    if not documents:
        return {
            **state,
            "documents_relevant": False,
            "reasoning_steps": _append_step(
                state, "No documents retrieved — marking as not relevant."
            ),
        }

    llm = ChatOllama(model=state["model"])
    context = "\n---\n".join(
        f"[Source: {doc.metadata.get('pdf_name', 'Unknown')}]\n{doc.page_content}"
        for doc in documents
    )
    prompt = ChatPromptTemplate.from_template(
        """You are a grader assessing whether retrieved document context is relevant
to the user question. Reply with exactly "yes" or "no".

Question: {question}

Context:
{context}

Is the context relevant enough to answer the question?"""
    )
    raw = (prompt | llm | StrOutputParser()).invoke(
        {"question": state["question"], "context": context}
    )
    relevant = raw.strip().lower().startswith("yes")
    return {
        **state,
        "documents_relevant": relevant,
        "reasoning_steps": _append_step(
            state,
            "Documents graded as relevant."
            if relevant
            else "Documents graded as not relevant.",
        ),
    }


def generate_answer(state: RAGState) -> RAGState:
    """Generate an answer from relevant documents."""
    documents = state.get("documents") or []
    context = "\n---\n".join(
        f"[Source: {doc.metadata.get('pdf_name', 'Unknown')}]\n{doc.page_content}"
        for doc in documents
    )
    llm = ChatOllama(model=state["model"])
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based ONLY on the following context from PDF documents.
Cite the source document name for each piece of information.

Context:
{context}

Question: {question}

Answer:"""
    )
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": state["question"]}
    )
    sources: list[dict[str, str | int | None]] = [
        {
            "pdf_name": str(doc.metadata.get("pdf_name") or "Unknown"),
            "pdf_id": str(doc.metadata.get("pdf_id") or ""),
            "chunk_index": int(doc.metadata.get("chunk_index") or 0),
        }
        for doc in documents
    ]
    updated: RAGState = {
        **state,
        "answer": answer,
        "sources": sources,
        "reasoning_steps": _append_step(
            state, "Generated answer with source citations."
        ),
    }
    return updated


def insufficient_information(state: RAGState) -> RAGState:
    """Return a fixed response when documents are not relevant."""
    return {
        **state,
        "answer": INSUFFICIENT_INFORMATION,
        "sources": [],
        "reasoning_steps": _append_step(
            state, "Returning insufficient information response."
        ),
    }


def route_after_grade(state: RAGState) -> Literal["generate_answer", "insufficient_information"]:
    """Route based on document relevance."""
    if state.get("documents_relevant"):
        return "generate_answer"
    return "insufficient_information"


def build_rag_graph():
    """Compile the LangGraph RAG workflow."""
    graph = StateGraph(RAGState)
    graph.add_node("analyze_question", analyze_question)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("insufficient_information", insufficient_information)

    graph.add_edge(START, "analyze_question")
    graph.add_edge("analyze_question", "retrieve_documents")
    graph.add_edge("retrieve_documents", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {
            "generate_answer": "generate_answer",
            "insufficient_information": "insufficient_information",
        },
    )
    graph.add_edge("generate_answer", END)
    graph.add_edge("insufficient_information", END)
    return graph.compile(name="pdf_rag_graph")


rag_graph = build_rag_graph()

# LangGraph / LangSmith Studio entrypoint (`langgraph.json` → pdf_rag).
graph = rag_graph
