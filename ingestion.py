import os
import re
import concurrent.futures
from typing import List, Tuple
import PyPDF2
import chromadb
from chromadb.utils import embedding_functions
from llm import call_llm


_CHUNK_SUMMARY_PROMPT = """Summarize this screenplay scene in 2-3 sentences. Cover: what happens, who is involved, and the emotional tone. Be concise.

Scene Title: {title}
Scene Text:
{text}

Reply with only the summary."""


def _summarize_chunk(title: str, text: str) -> str:
    try:
        return call_llm(
            _CHUNK_SUMMARY_PROMPT.format(title=title, text=text[:2000]),
            use_smart_model=False
        )
    except Exception:
        return text[:200]


def extract_text_from_pdf(pdf_file) -> str:
    reader = PyPDF2.PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text.strip()


def detect_scenes(text: str) -> List[Tuple[str, str]]:
    scene_pattern = re.compile(
        r'(?:^|\n)((?:INT\.|EXT\.|INT/EXT\.|SCENE\s+\d+|ACT\s+[IVX\d]+)[^\n]*)',
        re.IGNORECASE | re.MULTILINE
    )
    matches = list(scene_pattern.finditer(text))

    if len(matches) < 2:
        return _chunk_by_size(text, chunk_size=3600)

    scenes = []
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        scene_text = text[start:end].strip()
        if scene_text:
            scenes.append((title, scene_text))

    return scenes


def _chunk_by_size(text: str, chunk_size: int = 3600) -> List[Tuple[str, str]]:
    chunks = []
    words = text.split()
    current_chunk = []
    current_size = 0

    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        if current_size >= chunk_size:
            chunk_text = " ".join(current_chunk)
            chunks.append((f"Section {len(chunks) + 1}", chunk_text))
            current_chunk = []
            current_size = 0

    if current_chunk:
        chunks.append((f"Section {len(chunks) + 1}", " ".join(current_chunk)))

    return chunks


def store_chunks_in_chromadb(
    scenes: List[Tuple[str, str]],
    collection_name: str = "script_chunks"
) -> chromadb.Collection:
    """
    For each scene chunk:
      - stores full text as the document (for retrieval)
      - generates a vector embedding via text-embedding-3-small
      - generates a short LLM summary stored in metadata

    This lets pipeline nodes choose between full text (detailed analysis)
    and summary (broad context gathering) depending on the task.
    """
    client = chromadb.EphemeralClient()

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    collection = client.create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Generate summaries in parallel (one gpt-4o-mini call per chunk)
    def summarize(args):
        idx, title, text = args
        return idx, _summarize_chunk(title, text)

    summaries = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for idx, summary in executor.map(summarize, [(i, t, tx) for i, (t, tx) in enumerate(scenes)]):
            summaries[idx] = summary

    ids, documents, metadatas = [], [], []
    for i, (title, text) in enumerate(scenes):
        ids.append(f"scene_{i}")
        documents.append(text[:2000])       # full text for retrieval
        metadatas.append({
            "scene_index": i,
            "scene_title": title,
            "char_count": len(text),
            "summary": summaries.get(i, "")  # LLM summary for fast context
        })

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


def get_all_summaries(collection: chromadb.Collection) -> List[dict]:
    """Return all chunks ordered by scene index with their summaries."""
    result = collection.get(include=["metadatas", "documents"])
    chunks = []
    for i, meta in enumerate(result["metadatas"]):
        chunks.append({
            "scene_index": meta["scene_index"],
            "scene_title": meta["scene_title"],
            "summary": meta.get("summary", ""),
            "text": result["documents"][i],
        })
    return sorted(chunks, key=lambda x: x["scene_index"])


def query_relevant_chunks(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 5
) -> List[dict]:
    """Semantic search — returns chunks sorted by narrative order."""
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count())
    )
    chunks = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        chunks.append({
            "scene_title": meta["scene_title"],
            "scene_index": meta["scene_index"],
            "summary": meta.get("summary", ""),
            "text": results["documents"][0][i],
            "relevance_score": 1 - results["distances"][0][i],
        })
    return sorted(chunks, key=lambda x: x["scene_index"])
