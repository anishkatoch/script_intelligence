"""
LangGraph-based analysis pipeline for script intelligence.

Each node retrieves what it needs from ChromaDB:
  - summary_node         → all chunk summaries (broad overview)
  - emotional_arc_node   → full text per scene (detailed per-scene analysis)
  - engagement_node      → semantically relevant chunks per factor
  - suggestions_node     → semantically relevant chunks per critic lens
  - cliffhanger_node     → high-tension chunks from end of script
"""

import json
import concurrent.futures
from typing import TypedDict, List, Optional, Annotated, Any
import operator

import chromadb
from langgraph.graph import StateGraph, END

from schemas import (
    StorySummary, EmotionalArc, SceneEmotion,
    EngagementScore, EngagementFactor,
    ImprovementSuggestions, Suggestion,
    CliffhangerMoment, FullAnalysis
)
from llm import call_llm_json
from ingestion import get_all_summaries, query_relevant_chunks
from prompts import (
    SUMMARY_PROMPT, SCENE_EMOTION_PROMPT, EMOTIONAL_ARC_SYNTHESIS_PROMPT,
    HOOK_AGENT_PROMPT, CONFLICT_AGENT_PROMPT, TENSION_AGENT_PROMPT,
    CLIFFHANGER_AGENT_PROMPT, ENGAGEMENT_SYNTHESIS_PROMPT,
    DIALOGUE_CRITIC_PROMPT, PACING_CRITIC_PROMPT,
    CONFLICT_CRITIC_PROMPT, EMOTIONAL_CRITIC_PROMPT,
    SUGGESTIONS_SYNTHESIS_PROMPT, CLIFFHANGER_DETECTION_PROMPT
)


# ─── State ───────────────────────────────────────────────────────────────────

class ScriptState(TypedDict):
    title: str
    full_text: str
    scenes: List[dict]
    collection: Any                                      # chromadb.Collection

    scene_emotions: Annotated[List[dict], operator.add]

    summary: Optional[dict]
    emotional_arc: Optional[dict]
    engagement_score: Optional[dict]
    suggestions: Optional[dict]
    cliffhanger: Optional[dict]

    current_step: str
    errors: Annotated[List[str], operator.add]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _format_chunks(chunks: List[dict], use_summary: bool = False) -> str:
    """Format retrieved chunks into a single context string for the LLM."""
    parts = []
    for c in chunks:
        content = c["summary"] if use_summary else c["text"]
        parts.append(f"[{c['scene_title']}]\n{content}")
    return "\n\n---\n\n".join(parts)


# ─── Node: Summary ───────────────────────────────────────────────────────────

def summary_node(state: ScriptState) -> dict:
    """
    Builds the overall story summary from all chunk summaries.
    Using summaries (not raw text) keeps the prompt compact while
    covering the full script — no truncation needed.
    """
    try:
        all_chunks = get_all_summaries(state["collection"])
        context = _format_chunks(all_chunks, use_summary=True)
        prompt = SUMMARY_PROMPT.format(script_text=context)
        result = call_llm_json(prompt, use_smart_model=True)
        return {"summary": result, "current_step": "summary_complete"}
    except Exception as e:
        return {"errors": [f"Summary node error: {str(e)}"], "current_step": "summary_failed"}


# ─── Node: Emotional Arc ─────────────────────────────────────────────────────

def emotional_arc_node(state: ScriptState) -> dict:
    """
    Per-scene emotion analysis using chunk summaries from ChromaDB.
    Summaries are ~10x cheaper than raw text and sufficient for emotion detection.
    Processes scenes in order so each has context of the previous emotion.
    """
    all_chunks = get_all_summaries(state["collection"])  # already sorted by scene_index
    scene_emotions = []
    prev_emotion = "none (this is the first scene)"

    for chunk in all_chunks:
        try:
            prompt = SCENE_EMOTION_PROMPT.format(
                scene_index=chunk["scene_index"],
                scene_title=chunk["scene_title"],
                scene_text=chunk["summary"],   # summary instead of raw text
                prev_emotion=prev_emotion
            )
            result = call_llm_json(prompt, use_smart_model=False)
            scene_emotions.append(result)
            prev_emotion = f"{result.get('dominant_emotion', 'unknown')} (intensity: {result.get('intensity', 0.5)})"
        except Exception:
            scene_emotions.append({
                "scene_index": chunk["scene_index"],
                "scene_title": chunk["scene_title"],
                "dominant_emotion": "unknown",
                "intensity": 0.5,
                "emotional_shift": None
            })

    try:
        synthesis_prompt = EMOTIONAL_ARC_SYNTHESIS_PROMPT.format(
            scene_emotions_data=json.dumps(scene_emotions, indent=2)
        )
        arc = call_llm_json(synthesis_prompt, use_smart_model=True)
        arc["scene_emotions"] = scene_emotions
        return {
            "scene_emotions": scene_emotions,
            "emotional_arc": arc,
            "current_step": "emotional_arc_complete"
        }
    except Exception as e:
        return {
            "scene_emotions": scene_emotions,
            "errors": [f"Arc synthesis error: {str(e)}"],
            "current_step": "emotional_arc_failed"
        }


# ─── Node: Engagement Score ───────────────────────────────────────────────────

def engagement_node(state: ScriptState) -> dict:
    """
    Four specialized agents each query ChromaDB for the chunks most
    relevant to their evaluation lens, then a synthesis agent scores.
    """
    collection = state["collection"]
    scenes = state["scenes"]

    # Semantic search for each agent's specific need
    hook_chunks     = query_relevant_chunks(collection, "opening hook grabbing attention inciting incident introduction setup", n_results=4)
    conflict_chunks = query_relevant_chunks(collection, "conflict struggle character motivation obstacle", n_results=4)
    tension_chunks  = query_relevant_chunks(collection, "tension suspense danger stakes rising action", n_results=4)
    ending_chunks   = query_relevant_chunks(collection, "ending resolution climax final conclusion payoff", n_results=4)

    # Always include the last 2 scenes for the cliffhanger agent regardless of semantic score
    all_scenes_ordered = sorted(scenes, key=lambda s: s["scene_index"])
    final_scenes = all_scenes_ordered[-2:] if len(all_scenes_ordered) >= 2 else all_scenes_ordered
    seen_ending = {c["scene_index"] for c in ending_chunks}
    for s in final_scenes:
        if s["scene_index"] not in seen_ending:
            ending_chunks.append({"scene_title": s["scene_title"], "scene_index": s["scene_index"], "text": s["text"][:800], "summary": ""})

    hook_text    = _format_chunks(sorted(hook_chunks,    key=lambda x: x["scene_index"]))
    conflict_text = _format_chunks(sorted(conflict_chunks, key=lambda x: x["scene_index"]))
    tension_text  = _format_chunks(sorted(tension_chunks,  key=lambda x: x["scene_index"]))
    ending_text   = _format_chunks(sorted(ending_chunks,   key=lambda x: x["scene_index"]))

    prompts = {
        "hook": HOOK_AGENT_PROMPT.format(opening_text=hook_text),
        "conflict": CONFLICT_AGENT_PROMPT.format(script_text=conflict_text),
        "tension": TENSION_AGENT_PROMPT.format(script_text=tension_text),
        "cliffhanger_factor": CLIFFHANGER_AGENT_PROMPT.format(ending_text=ending_text),
    }

    factors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_key = {
            executor.submit(call_llm_json, prompt, False): key
            for key, prompt in prompts.items()
        }
        for future in concurrent.futures.as_completed(future_to_key):
            try:
                factors.append(future.result())
            except Exception:
                factors.append({
                    "factor_name": future_to_key[future],
                    "score": 5.0,
                    "reasoning": "Could not evaluate",
                    "evidence": "N/A"
                })

    try:
        synthesis = call_llm_json(
            ENGAGEMENT_SYNTHESIS_PROMPT.format(factors_data=json.dumps(factors, indent=2)),
            use_smart_model=True
        )
        synthesis["factors"] = factors
        return {"engagement_score": synthesis, "current_step": "engagement_complete"}
    except Exception as e:
        return {
            "errors": [f"Engagement synthesis error: {str(e)}"],
            "current_step": "engagement_failed"
        }


# ─── Node: Improvement Suggestions ───────────────────────────────────────────

def suggestions_node(state: ScriptState) -> dict:
    """
    Four critics each get the chunks most relevant to their lens,
    then a synthesis agent deduplicates and prioritizes.
    """
    collection = state["collection"]

    dialogue_chunks  = query_relevant_chunks(collection, "dialogue conversation character speech", n_results=4)
    pacing_chunks    = query_relevant_chunks(collection, "pacing rhythm scene transitions momentum", n_results=4)
    conflict_chunks  = query_relevant_chunks(collection, "conflict stakes obstacles character goals", n_results=4)
    emotional_chunks = query_relevant_chunks(collection, "emotion feeling character arc transformation", n_results=4)

    critic_prompts = [
        DIALOGUE_CRITIC_PROMPT.format(script_text=_format_chunks(dialogue_chunks)),
        PACING_CRITIC_PROMPT.format(script_text=_format_chunks(pacing_chunks)),
        CONFLICT_CRITIC_PROMPT.format(script_text=_format_chunks(conflict_chunks)),
        EMOTIONAL_CRITIC_PROMPT.format(script_text=_format_chunks(emotional_chunks)),
    ]

    all_suggestions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(call_llm_json, p, False) for p in critic_prompts]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if isinstance(result, list):
                    all_suggestions.extend(result)
            except Exception:
                pass

    try:
        final = call_llm_json(
            SUGGESTIONS_SYNTHESIS_PROMPT.format(all_suggestions=json.dumps(all_suggestions, indent=2)),
            use_smart_model=True
        )
        return {"suggestions": final, "current_step": "suggestions_complete"}
    except Exception as e:
        return {
            "suggestions": {"suggestions": all_suggestions[:6], "most_critical_fix": "See suggestions above"},
            "errors": [f"Suggestions synthesis error: {str(e)}"],
            "current_step": "suggestions_complete"
        }


# ─── Node: Cliffhanger Detection ─────────────────────────────────────────────

def cliffhanger_node(state: ScriptState) -> dict:
    """
    Retrieves the most tension-heavy and suspenseful chunks from ChromaDB,
    weighted toward the end of the script, then identifies the cliffhanger.
    """
    collection = state["collection"]
    tension_chunks = query_relevant_chunks(
        collection, "cliffhanger suspense revelation twist danger unresolved", n_results=5
    )
    # Bias toward the end: also grab the last 2 scenes directly
    all_chunks = get_all_summaries(collection)
    end_chunks = all_chunks[-2:] if len(all_chunks) >= 2 else all_chunks

    seen = {c["scene_index"] for c in tension_chunks}
    for c in end_chunks:
        if c["scene_index"] not in seen:
            tension_chunks.append(c)

    context = _format_chunks(sorted(tension_chunks, key=lambda x: x["scene_index"]))

    try:
        result = call_llm_json(
            CLIFFHANGER_DETECTION_PROMPT.format(script_text=context),
            use_smart_model=True
        )
        return {"cliffhanger": result, "current_step": "cliffhanger_complete"}
    except Exception as e:
        return {
            "errors": [f"Cliffhanger node error: {str(e)}"],
            "current_step": "cliffhanger_failed"
        }


# ─── Graph ────────────────────────────────────────────────────────────────────

def build_analysis_graph() -> StateGraph:
    graph = StateGraph(ScriptState)
    graph.add_node("run_summary", summary_node)
    graph.add_node("run_emotional_arc", emotional_arc_node)
    graph.add_node("run_engagement", engagement_node)
    graph.add_node("run_suggestions", suggestions_node)
    graph.add_node("run_cliffhanger", cliffhanger_node)

    graph.set_entry_point("run_summary")
    graph.add_edge("run_summary", "run_emotional_arc")
    graph.add_edge("run_emotional_arc", "run_engagement")
    graph.add_edge("run_engagement", "run_suggestions")
    graph.add_edge("run_suggestions", "run_cliffhanger")
    graph.add_edge("run_cliffhanger", END)

    return graph.compile()


def run_analysis(
    title: str,
    full_text: str,
    scenes: List[dict],
    collection: chromadb.Collection
) -> ScriptState:
    graph = build_analysis_graph()
    initial_state: ScriptState = {
        "title": title,
        "full_text": full_text,
        "scenes": scenes,
        "collection": collection,
        "scene_emotions": [],
        "summary": None,
        "emotional_arc": None,
        "engagement_score": None,
        "suggestions": None,
        "cliffhanger": None,
        "current_step": "starting",
        "errors": []
    }
    return graph.invoke(initial_state)
