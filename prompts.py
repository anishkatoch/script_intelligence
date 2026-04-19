SUMMARY_PROMPT = """You are a professional script analyst for a short-form content platform.

Analyze the following script and return a JSON object with this exact structure:
{{
  "summary": "4-5 line narrative summary covering setup, conflict, and stakes",
  "genre": "detected genre",
  "central_conflict": "the core conflict driving the story",
  "main_characters": ["character1", "character2"]
}}

Return ONLY the JSON. No preamble, no markdown backticks.

SCRIPT:
{script_text}
"""


SCENE_EMOTION_PROMPT = """You are an expert in narrative emotional analysis.

Analyze the emotional content of this single scene and return a JSON object:
{{
  "scene_index": {scene_index},
  "scene_title": "{scene_title}",
  "dominant_emotion": "single dominant emotion (e.g. grief, tension, hope, fear, love, anger)",
  "intensity": 0.0 to 1.0 (float),
  "emotional_shift": "how emotion shifted compared to previous scene context, or null if first scene"
}}

Previous scene emotion context: {prev_emotion}

Return ONLY the JSON. No preamble, no markdown backticks.

SCENE:
{scene_text}
"""


EMOTIONAL_ARC_SYNTHESIS_PROMPT = """You are a narrative structure expert.

Given the following per-scene emotional analysis data, synthesize the overall emotional arc.

Return a JSON object:
{{
  "overall_dominant_emotion": "the most dominant emotion across the full script",
  "arc_pattern": "e.g. Rising tension, Emotional rollercoaster, Tragic descent, Redemptive climb",
  "arc_description": "2-3 lines describing how emotion evolves across the story",
  "emotional_peak_scene": "title of the scene where emotion peaks most powerfully"
}}

Return ONLY the JSON. No preamble, no markdown backticks.

SCENE EMOTIONS DATA:
{scene_emotions_data}
"""


# Engagement sub-agent prompts - each evaluates one specific factor independently

HOOK_AGENT_PROMPT = """You are a specialist in evaluating opening hooks for short-form scripted content.

Evaluate ONLY the opening hook strength of this script. Consider:
- Does the opening immediately create intrigue?
- Is there a compelling question raised in the first scene?
- Would an audience keep watching after the first 30 seconds?

Return a JSON object:
{{
  "factor_name": "opening_hook",
  "score": 0.0 to 10.0,
  "reasoning": "your reasoning",
  "evidence": "specific line or moment from the opening that most influenced your score"
}}

Return ONLY the JSON. No preamble, no markdown backticks.

SCRIPT OPENING (first 2 scenes):
{opening_text}
"""


CONFLICT_AGENT_PROMPT = """You are a specialist in evaluating character conflict and dramatic stakes.

Evaluate ONLY the character conflict of this script. Consider:
- Are the opposing forces clearly defined?
- Are the stakes high enough to generate investment?
- Is the conflict specific and personal rather than generic?

Return a JSON object:
{{
  "factor_name": "character_conflict",
  "score": 0.0 to 10.0,
  "reasoning": "your reasoning",
  "evidence": "specific line or moment that best represents the conflict"
}}

Return ONLY the JSON. No preamble, no markdown backticks.

SCRIPT:
{script_text}
"""


TENSION_AGENT_PROMPT = """You are a specialist in evaluating narrative tension and pacing.

Evaluate ONLY the tension build of this script. Consider:
- Does tension escalate progressively?
- Are there effective moments of release and re-escalation?
- Is the pacing controlled or does it feel rushed/slow?

Return a JSON object:
{{
  "factor_name": "tension_build",
  "score": 0.0 to 10.0,
  "reasoning": "your reasoning",
  "evidence": "specific moment where tension is handled most effectively or ineffectively"
}}

Return ONLY the JSON. No preamble, no markdown backticks.

SCRIPT:
{script_text}
"""


CLIFFHANGER_AGENT_PROMPT = """You are a specialist in evaluating cliffhangers and narrative hooks.

Evaluate ONLY the cliffhanger/hook quality of this script. Consider:
- Does the script end or pause on an unresolved tension?
- Is there a revelation or question that compels continued viewing?
- Is the cliffhanger earned by the preceding narrative?

Return a JSON object:
{{
  "factor_name": "cliffhanger_presence",
  "score": 0.0 to 10.0,
  "reasoning": "your reasoning",
  "evidence": "the specific moment or line that functions as the cliffhanger"
}}

Return ONLY the JSON. No preamble, no markdown backticks.

SCRIPT ENDING (last 2 scenes):
{ending_text}
"""


ENGAGEMENT_SYNTHESIS_PROMPT = """You are a senior content strategist synthesizing multiple engagement evaluations.

Given these independent factor scores, produce a final engagement assessment.

Return a JSON object:
{{
  "overall_score": weighted average score 0.0-10.0,
  "confidence": 0.0 to 1.0 (how confident you are given the evidence),
  "factors": [the exact factor objects provided below],
  "verdict": "one compelling sentence summarizing the engagement potential"
}}

Return ONLY the JSON. No preamble, no markdown backticks.

FACTOR SCORES:
{factors_data}
"""


# Suggestion critic prompts - each evaluates one storytelling dimension

DIALOGUE_CRITIC_PROMPT = """You are a dialogue specialist for short-form scripted content.

Analyze ONLY the dialogue quality of this script and provide 2 specific, actionable improvement suggestions focused on dialogue.

Each suggestion must reference a specific moment in the script.

Return a JSON array of suggestion objects:
[
  {{
    "category": "dialogue",
    "priority": "high/medium/low",
    "suggestion": "specific actionable suggestion",
    "reasoning": "why this would improve the script",
    "example": "optional rewrite example of a specific line"
  }}
]

Return ONLY the JSON array. No preamble, no markdown backticks.

SCRIPT:
{script_text}
"""


PACING_CRITIC_PROMPT = """You are a pacing and structure specialist for short-form scripted content.

Analyze ONLY the pacing of this script and provide 2 specific, actionable improvement suggestions focused on pacing and scene structure.

Return a JSON array of suggestion objects:
[
  {{
    "category": "pacing",
    "priority": "high/medium/low",
    "suggestion": "specific actionable suggestion",
    "reasoning": "why this would improve the script",
    "example": null
  }}
]

Return ONLY the JSON array. No preamble, no markdown backticks.

SCRIPT:
{script_text}
"""


CONFLICT_CRITIC_PROMPT = """You are a conflict and dramatic stakes specialist.

Analyze ONLY the conflict structure of this script and provide 2 specific improvement suggestions.

Return a JSON array of suggestion objects:
[
  {{
    "category": "conflict",
    "priority": "high/medium/low",
    "suggestion": "specific actionable suggestion",
    "reasoning": "why this would improve the script",
    "example": null
  }}
]

Return ONLY the JSON array. No preamble, no markdown backticks.

SCRIPT:
{script_text}
"""


EMOTIONAL_CRITIC_PROMPT = """You are an emotional impact specialist for storytelling.

Analyze ONLY the emotional impact of this script and provide 2 specific improvement suggestions focused on emotional resonance and payoff.

Return a JSON array of suggestion objects:
[
  {{
    "category": "emotional_impact",
    "priority": "high/medium/low",
    "suggestion": "specific actionable suggestion",
    "reasoning": "why this would improve the script",
    "example": null
  }}
]

Return ONLY the JSON array. No preamble, no markdown backticks.

SCRIPT:
{script_text}
"""


SUGGESTIONS_SYNTHESIS_PROMPT = """You are a senior script editor synthesizing critique from multiple specialists.

Given suggestions from four specialist critics, rank and deduplicate them into a final prioritized list. Remove any redundant suggestions, keeping the most specific and actionable version.

Return a JSON object:
{{
  "suggestions": [ordered list of suggestion objects, highest priority first, max 6],
  "most_critical_fix": "the single most important improvement in one sentence"
}}

Return ONLY the JSON. No preamble, no markdown backticks.

ALL SUGGESTIONS:
{all_suggestions}
"""


CLIFFHANGER_DETECTION_PROMPT = """You are an expert in narrative suspense and cliffhanger mechanics.

Identify the single most suspenseful or cliffhanger moment in this script and explain exactly why it works as a narrative device.

Return a JSON object:
{{
  "moment": "the exact line or moment",
  "scene_context": "brief context of where this occurs in the story",
  "mechanism": "the specific narrative mechanism that makes this work (e.g. information asymmetry, unresolved emotional question, physical threat, revelation)",
  "tension_type": "Unresolved question / Emotional revelation / Physical threat / Betrayal / Time pressure",
  "effectiveness_score": 0.0 to 10.0
}}

Return ONLY the JSON. No preamble, no markdown backticks.

SCRIPT:
{script_text}
"""
