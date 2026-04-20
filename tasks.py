import json
import logging
import shutil
from celery_app import celery_app
from ingestion import detect_scenes, store_chunks_in_chromadb
from pipeline import run_analysis
from config import CHROMA_BASE_DIR

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="tasks.analyse_script",
    max_retries=1,               # retry once if the whole task fails
    default_retry_delay=10,
)
def analyse_script(self, title: str, raw_text: str):
    """
    Celery task that runs the full analysis pipeline.

    Accepts raw text (not the PDF file — PDF is read in app.py before submitting).
    Returns the full results dict serialisable as JSON.

    Flow:
        1. Detect scenes from raw text
        2. Index into ChromaDB (with per-job isolated path)
        3. Run LangGraph pipeline
        4. Clean up ChromaDB directory
        5. Return results as JSON-safe dict
    """
    job_id = self.request.id
    chroma_path = f"{CHROMA_BASE_DIR}/{job_id}"
    logger.info(f"[{job_id}] Starting analysis for '{title}'")

    try:
        # ── Ingestion ─────────────────────────────────────────────────────────
        self.update_state(state="PROGRESS", meta={"step": "Detecting scenes..."})
        scenes_raw = detect_scenes(raw_text)
        scenes = [
            {"scene_index": i, "scene_title": t, "text": tx}
            for i, (t, tx) in enumerate(scenes_raw)
        ]
        logger.info(f"[{job_id}] Detected {len(scenes)} scenes")

        # ── ChromaDB Indexing ─────────────────────────────────────────────────
        self.update_state(state="PROGRESS", meta={"step": f"Indexing {len(scenes)} scenes into ChromaDB..."})
        collection = store_chunks_in_chromadb(
            scenes_raw,
            collection_name=f"job_{job_id[:8]}",
            persist_path=chroma_path
        )
        logger.info(f"[{job_id}] ChromaDB indexed at {chroma_path}")

        # ── Pipeline ──────────────────────────────────────────────────────────
        self.update_state(state="PROGRESS", meta={"step": "Running analysis pipeline..."})
        results = run_analysis(title, raw_text, scenes, collection)
        logger.info(f"[{job_id}] Pipeline complete")

        # Strip non-serialisable objects (ChromaDB collection) before returning
        safe_results = {
            k: v for k, v in results.items()
            if k not in ("collection", "full_text", "scenes")
        }

        return {"status": "complete", "results": safe_results}

    except Exception as exc:
        logger.error(f"[{job_id}] Task failed: {exc}", exc_info=True)
        raise self.retry(exc=exc)

    finally:
        # Always clean up the ChromaDB directory for this job
        try:
            shutil.rmtree(chroma_path, ignore_errors=True)
            logger.info(f"[{job_id}] Cleaned up ChromaDB at {chroma_path}")
        except Exception:
            pass
