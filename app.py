import streamlit as st
import os
import re
import json
from dotenv import load_dotenv

load_dotenv()

from ingestion import extract_text_from_pdf, detect_scenes, store_chunks_in_chromadb
from pipeline import run_analysis

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Bullet — Script Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Mono:wght@300;400&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #ffffff;
    --surface: #f5f5f5;
    --surface2: #ebebeb;
    --border: #d0d0d0;
    --accent: #c9a800;
    --accent2: #e05a00;
    --text: #1a1a1a;
    --muted: #666666;
    --danger: #d32f2f;
    --success: #2e7d32;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

.stApp { background: var(--bg); }

/* Labels */
label, .stTextInput label, .stFileUploader label {
    color: var(--text) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    color: var(--text) !important;
}

/* General text */
p, span, div {
    color: var(--text);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1200px; }

/* Header */
.bullet-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 0.25rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}

.bullet-logo {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -1px;
    line-height: 1;
}

.bullet-tagline {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 3px;
    text-transform: uppercase;
}

/* Upload zone */
.upload-zone {
    border: 1px dashed var(--border);
    border-radius: 4px;
    padding: 3rem 2rem;
    text-align: center;
    background: var(--surface);
    transition: border-color 0.2s;
}

/* Analysis cards */
.analysis-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.75rem;
}

.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.5rem;
}

/* Score display */
.score-display {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin: 1rem 0;
}

.score-number {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}

.score-denom {
    font-family: 'DM Mono', monospace;
    font-size: 1rem;
    color: var(--muted);
}

/* Factor bars */
.factor-row {
    margin-bottom: 0.75rem;
}

.factor-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.25rem;
    display: flex;
    justify-content: space-between;
}

.factor-bar-bg {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}

.factor-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 2px;
    transition: width 0.8s ease;
}

/* Emotion arc */
.emotion-chip {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 2px;
}

/* Suggestion pills */
.suggestion-item {
    border-left: 2px solid var(--accent);
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
    background: var(--surface2);
}

.suggestion-category {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 0.25rem;
}

.priority-high { border-left-color: var(--danger); }
.priority-medium { border-left-color: var(--accent); }
.priority-low { border-left-color: var(--muted); }

/* Cliffhanger block */
.cliffhanger-quote {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 1.2rem;
    color: var(--accent);
    border-left: 3px solid var(--accent);
    padding: 0.5rem 1rem;
    margin: 1rem 0;
}

/* Progress */
.step-indicator {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 1px;
}

.step-active {
    color: var(--accent);
}

/* Genre tag */
.genre-tag {
    display: inline-block;
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* stProgress override */
.stProgress > div > div > div > div {
    background: var(--accent) !important;
}

/* Override Streamlit button */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
    border-radius: 2px !important;
    font-weight: 500 !important;
}

.stButton > button:hover {
    background: #f5d060 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 4px !important;
}

/* Text input */
.stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 2px !important;
}

</style>
""", unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="bullet-header">
    <div class="bullet-logo">bullet.</div>
    <div class="bullet-tagline">Script Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def get_emotion_color(emotion: str) -> str:
    colors = {
        "grief": "#6b7db3", "sadness": "#6b7db3", "sorrow": "#6b7db3",
        "tension": "#e85d4a", "fear": "#e85d4a", "anxiety": "#e85d4a",
        "anger": "#ff4444", "rage": "#ff4444",
        "hope": "#44cc88", "joy": "#44cc88", "relief": "#44cc88",
        "love": "#e879a0", "romance": "#e879a0",
        "mystery": "#9b7fd4", "suspense": "#9b7fd4",
        "shock": "#ff9933", "surprise": "#ff9933",
        "melancholy": "#7b9eb8",
    }
    for key, color in colors.items():
        if key in emotion.lower():
            return color
    return "#555555"


def render_score_bar(label: str, score: float, evidence: str = ""):
    pct = (score / 10) * 100
    color = "#44cc88" if score >= 7 else "#e8c547" if score >= 5 else "#ff4444"
    st.markdown(f"""
    <div class="factor-row">
        <div class="factor-label">
            <span>{label.replace('_', ' ')}</span>
            <span style="color: {color}">{score:.1f}</span>
        </div>
        <div class="factor-bar-bg">
            <div class="factor-bar-fill" style="width:{pct}%; background: {color}"></div>
        </div>
        {f'<div style="font-size:0.72rem; color: #555555; margin-top: 0.25rem; font-style: italic">{evidence}</div>' if evidence else ''}
    </div>
    """, unsafe_allow_html=True)


# ─── Main UI ──────────────────────────────────────────────────────────────────

col_upload, col_info = st.columns([3, 2])

with col_upload:
    st.markdown('<div class="card-label">Upload Script</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your PDF script here",
        type=["pdf"],
        label_visibility="collapsed"
    )
    script_title = st.text_input("Script Title", placeholder="e.g. The Last Message", label_visibility="visible")

with col_info:
    st.markdown("""
    <div class="analysis-card" style="height: 100%">
        <div class="card-label">What this analyses</div>
        <div style="font-size: 0.85rem; color: #555555; line-height: 1.8">
            ◦ Narrative summary & genre detection<br>
            ◦ Scene-by-scene emotional arc<br>
            ◦ Engagement scoring across 4 dimensions<br>
            ◦ Specialist critique from 4 story lenses<br>
            ◦ Cliffhanger & tension detection
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

if uploaded_file and script_title:
    if st.button("▶  Analyse Script"):

        # ── Ingestion ──
        with st.spinner("Extracting & indexing script..."):
            raw_text = extract_text_from_pdf(uploaded_file)
            scenes_raw = detect_scenes(raw_text)
            scenes = [
                {"scene_index": i, "scene_title": title, "text": text}
                for i, (title, text) in enumerate(scenes_raw)
            ]
            safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', script_title[:20]).strip('_') or "script"
            collection = store_chunks_in_chromadb(scenes_raw, collection_name=f"script_{safe_title}")

        st.markdown(f"""
        <div style="font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #555555; margin-bottom: 1.5rem">
            ✓ Detected {len(scenes)} scenes · {len(raw_text):,} characters · Indexed in ChromaDB
        </div>
        """, unsafe_allow_html=True)

        # ── Analysis Steps Progress ──
        steps = ["Summary", "Emotional Arc", "Engagement", "Suggestions", "Cliffhanger"]
        progress_bar = st.progress(0)
        step_text = st.empty()

        def update_progress(step_num, step_name):
            progress_bar.progress((step_num) / len(steps))
            step_text.markdown(f'<div class="step-indicator">Analysing → <span class="step-active">{step_name}</span></div>', unsafe_allow_html=True)

        update_progress(0, "Summary")

        # ── Run Pipeline ──
        results = run_analysis(script_title, raw_text, scenes, collection)

        progress_bar.progress(1.0)
        step_text.markdown('<div class="step-indicator" style="color: #44cc88">✓ Analysis complete</div>', unsafe_allow_html=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════
        # RESULTS RENDERING
        # ══════════════════════════════════════════════════════════════════

        # ── Summary ──────────────────────────────────────────────────────
        if results.get("summary"):
            s = results["summary"]
            st.markdown(f"""
            <div class="analysis-card">
                <div class="card-label">Story Summary</div>
                <div class="card-title">{script_title}</div>
                <span class="genre-tag">{s.get('genre', 'Drama')}</span>
                <p style="margin-top: 1rem; font-size: 0.95rem; line-height: 1.7; color: #333333">{s.get('summary', '')}</p>
                <div style="margin-top: 0.75rem">
                    <span style="font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #555555; text-transform: uppercase; letter-spacing: 2px">Central Conflict</span>
                    <p style="font-size: 0.85rem; color: #333333; margin-top: 0.25rem">{s.get('central_conflict', '')}</p>
                </div>
                <div style="margin-top: 0.5rem">
                    <span style="font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #555555; text-transform: uppercase; letter-spacing: 2px">Characters</span>
                    <p style="font-size: 0.85rem; color: #333333; margin-top: 0.25rem">{', '.join(s.get('main_characters', []))}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Emotional Arc ─────────────────────────────────────────────────
        if results.get("emotional_arc"):
            arc = results["emotional_arc"]
            scene_emotions = arc.get("scene_emotions", [])

            st.markdown(f"""
            <div class="analysis-card">
                <div class="card-label">Emotional Arc</div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem">
                    <div style="font-family: 'Playfair Display', serif; font-size: 1.1rem; font-style: italic; color: #f0c040">{arc.get('arc_pattern', '')}</div>
                    <div style="font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #a8a8c0; text-transform: uppercase; letter-spacing: 2px">Peak: {arc.get('emotional_peak_scene', '')}</div>
                </div>
                <p style="font-size: 0.85rem; color: #a8a8c0; line-height: 1.6; margin-bottom: 0.5rem">{arc.get('arc_description', '')}</p>
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(len(scene_emotions) if scene_emotions else 1)
            for i, se in enumerate(scene_emotions):
                color = get_emotion_color(se.get("dominant_emotion", ""))
                intensity = se.get("intensity", 0.5)
                opacity = round(0.4 + (intensity * 0.6), 2)
                with cols[i]:
                    st.markdown(f"""
                    <div style="text-align:center; padding: 4px;">
                        <div style="background: {color}33; color: {color}; border: 1px solid {color}88; border-radius: 3px; padding: 4px 6px; font-size: 0.7rem; font-family: monospace; opacity: {opacity}; margin-bottom: 4px;">
                            {se.get('dominant_emotion', '')[:10]}
                        </div>
                        <div style="font-size: 0.6rem; color: #a8a8c0; font-family: monospace; word-break: break-word;">
                            {se.get('scene_title', '')[:14]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Engagement Score ──────────────────────────────────────────────
        if results.get("engagement_score"):
            eng = results["engagement_score"]
            score = eng.get("overall_score", 0)
            confidence = eng.get("confidence", 0.8)
            factors = eng.get("factors", [])

            score_color = "#44cc88" if score >= 7 else "#e8c547" if score >= 5 else "#ff4444"

            col_score, col_factors = st.columns([1, 2])

            with col_score:
                st.markdown(f"""
                <div class="analysis-card" style="height: 100%">
                    <div class="card-label">Engagement Score</div>
                    <div class="score-display">
                        <div class="score-number" style="color: {score_color}">{score:.1f}</div>
                        <div class="score-denom">/ 10</div>
                    </div>
                    <div style="font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #555555; margin-bottom: 0.5rem">
                        Confidence: {confidence*100:.0f}%
                    </div>
                    <p style="font-size: 0.8rem; color: #333333; font-style: italic; line-height: 1.5">{eng.get('verdict', '')}</p>
                </div>
                """, unsafe_allow_html=True)

            with col_factors:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-label">Factor Breakdown</div>', unsafe_allow_html=True)
                for factor in factors:
                    render_score_bar(
                        factor.get("factor_name", ""),
                        factor.get("score", 0),
                        factor.get("evidence", "")
                    )
                st.markdown('</div>', unsafe_allow_html=True)

        # ── Improvement Suggestions ───────────────────────────────────────
        if results.get("suggestions"):
            sug_data = results["suggestions"]
            suggestions = sug_data.get("suggestions", [])
            critical_fix = sug_data.get("most_critical_fix", "")

            st.markdown('<div class="analysis-card"><div class="card-label">Improvement Suggestions</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background: #fff3e0; border: 1px solid #ffb74d; border-radius: 2px; padding: 0.75rem 1rem; margin-bottom: 1rem">
                <span style="font-family: 'DM Mono', monospace; font-size: 0.65rem; letter-spacing: 2px; text-transform: uppercase; color: #e05a00">Most Critical</span>
                <p style="font-size: 0.88rem; color: #e8c547; margin-top: 0.25rem; margin-bottom: 0">{critical_fix}</p>
            </div>
            """, unsafe_allow_html=True)
            for sug in suggestions:
                priority = sug.get("priority", "medium")
                example_html = ""
                if sug.get("example"):
                    example_html = f'<div style="font-family: DM Mono, monospace; font-size: 0.75rem; color: #7a5c00; margin-top: 0.5rem; padding: 0.4rem; background: #fffde7; border-radius: 2px; border: 1px solid #ffe082">→ {sug["example"]}</div>'
                st.markdown(f"""
                <div class="suggestion-item priority-{priority}">
                    <div class="suggestion-category">{sug.get('category', '').replace('_', ' ')}</div>
                    <div style="font-size: 0.88rem; color: #1a1a1a; line-height: 1.5">{sug.get('suggestion', '')}</div>
                    <div style="font-size: 0.78rem; color: #555555; margin-top: 0.25rem">{sug.get('reasoning', '')}</div>
                    {example_html}
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Cliffhanger ───────────────────────────────────────────────────
        if results.get("cliffhanger"):
            ch = results["cliffhanger"]
            eff_score = ch.get("effectiveness_score", 0)

            st.markdown(f"""
            <div class="analysis-card" style="border-color: #3a2a00">
                <div class="card-label" style="color: #ff6b35">Cliffhanger Detection</div>
                <div class="cliffhanger-quote">"{ch.get('moment', '')}"</div>
                <div style="display: flex; gap: 1rem; margin-bottom: 0.75rem; flex-wrap: wrap">
                    <div>
                        <div style="font-family: 'DM Mono', monospace; font-size: 0.6rem; color: #555555; text-transform: uppercase; letter-spacing: 2px">Tension Type</div>
                        <div style="font-size: 0.85rem; color: #ff6b35">{ch.get('tension_type', '')}</div>
                    </div>
                    <div>
                        <div style="font-family: 'DM Mono', monospace; font-size: 0.6rem; color: #555555; text-transform: uppercase; letter-spacing: 2px">Effectiveness</div>
                        <div style="font-size: 0.85rem; color: #e8c547">{eff_score:.1f} / 10</div>
                    </div>
                </div>
                <div style="font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #555555; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.25rem">Mechanism</div>
                <p style="font-size: 0.85rem; color: #333333; line-height: 1.6">{ch.get('mechanism', '')}</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Errors (if any) ───────────────────────────────────────────────
        if results.get("errors"):
            with st.expander("⚠ Pipeline warnings"):
                for err in results["errors"]:
                    st.text(err)

        # ── Raw JSON Export ───────────────────────────────────────────────
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        export_data = {
            "title": script_title,
            "summary": results.get("summary"),
            "emotional_arc": results.get("emotional_arc"),
            "engagement_score": results.get("engagement_score"),
            "suggestions": results.get("suggestions"),
            "cliffhanger": results.get("cliffhanger"),
        }
        st.download_button(
            label="↓  Export JSON Analysis",
            data=json.dumps(export_data, indent=2),
            file_name=f"{script_title.replace(' ', '_')}_analysis.json",
            mime="application/json"
        )

elif not uploaded_file:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; color: #444">
        <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; color: #333333; margin-bottom: 0.5rem">Upload a PDF script to begin</div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase">Supports standard screenplay format</div>
    </div>
    """, unsafe_allow_html=True)
