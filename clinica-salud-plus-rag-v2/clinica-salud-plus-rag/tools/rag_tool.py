"""
tools/rag_tool.py
Herramienta RAG de consulta semántica sobre la base de conocimiento de Clínica Salud Plus.
Recupera fragmentos relevantes desde ChromaDB usando embeddings de OpenAI.
"""
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool


# ── Configuración de embeddings ─────────────────────────────────────────────
def _get_embeddings() -> OpenAIEmbeddings:
    """Inicializa el modelo de embeddings con GitHub Models / OpenAI."""
    token    = os.environ.get("GITHUB_TOKEN") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")
    return OpenAIEmbeddings(
        api_key=token,
        base_url=base_url,
        model="text-embedding-3-small",
    )


# ── Construcción del vectorstore (singleton en módulo) ───────────────────────
_vectorstore: Chroma | None = None


def _build_vectorstore(data_path: str = "data/info_clinica.txt") -> Chroma:
    """
    Carga el documento de la clínica, lo divide en chunks y construye
    el vectorstore ChromaDB. Se crea una sola vez por sesión.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    # 1. Cargar documento
    loader    = TextLoader(data_path, encoding="utf-8")
    documentos = loader.load()

    # 2. Dividir en chunks (IL1.3: texto con solapamiento para mantener contexto)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documentos)

    # 3. Crear vectorstore en memoria
    embeddings   = _get_embeddings()
    _vectorstore = Chroma.from_documents(chunks, embeddings)
    return _vectorstore


def get_retriever(top_k: int = 3):
    """Retorna el retriever configurado con top_k resultados."""
    vs = _build_vectorstore()
    return vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": 0.3},
    )


# ── Herramienta LangChain ────────────────────────────────────────────────────
@tool
def buscar_informacion_clinica(consulta: str) -> str:
    """
    Busca información relevante sobre la Clínica Salud Plus.
    Úsala para responder preguntas sobre horarios, médicos, especialidades,
    preparación de exámenes, convenios, pagos y preguntas frecuentes.
    Devuelve los fragmentos más relevantes encontrados en la base de conocimiento.
    """
    retriever = get_retriever(top_k=3)
    docs = retriever.invoke(consulta)

    if not docs:
        return (
            "No se encontró información específica sobre esa consulta en la base "
            "de conocimiento de la clínica. Considera solicitar más detalles al paciente "
            "o derivar a la recepción llamando al 600 123 4567."
        )

    fragmentos = []
    for i, doc in enumerate(docs, 1):
        fragmentos.append(f"[Fragmento {i}]\n{doc.page_content.strip()}")

    return "\n\n".join(fragmentos)
