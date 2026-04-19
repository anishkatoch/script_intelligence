from pydantic import BaseModel, Field
from typing import List, Optional


class StorySummary(BaseModel):
    summary: str = Field(description="3-4 line narrative summary of the script")
    genre: str = Field(description="Detected genre of the script")
    central_conflict: str = Field(description="The core conflict driving the story")
    main_characters: List[str] = Field(description="List of main characters identified")


class SceneEmotion(BaseModel):
    scene_index: int
    scene_title: str
    dominant_emotion: str
    intensity: float = Field(ge=0.0, le=1.0, description="Emotion intensity 0-1")
    emotional_shift: Optional[str] = Field(description="How emotion shifted from previous scene")


class EmotionalArc(BaseModel):
    scene_emotions: List[SceneEmotion]
    overall_dominant_emotion: str
    arc_pattern: str = Field(description="e.g. Rising tension, Emotional rollercoaster, Steady buildup")
    arc_description: str = Field(description="2-3 line description of how emotion evolves")
    emotional_peak_scene: str = Field(description="Scene where emotion peaks most strongly")


class EngagementFactor(BaseModel):
    factor_name: str
    score: float = Field(ge=0.0, le=10.0)
    reasoning: str
    evidence: str = Field(description="Specific line or moment from the script that drove this score")


class EngagementScore(BaseModel):
    overall_score: float = Field(ge=0.0, le=10.0)
    confidence: float = Field(ge=0.0, le=1.0)
    factors: List[EngagementFactor]
    verdict: str = Field(description="One line overall verdict on engagement potential")


class Suggestion(BaseModel):
    category: str = Field(description="dialogue / pacing / conflict / emotional_impact")
    priority: str = Field(description="high / medium / low")
    suggestion: str
    reasoning: str
    example: Optional[str] = Field(description="Optional rewrite example")


class ImprovementSuggestions(BaseModel):
    suggestions: List[Suggestion]
    most_critical_fix: str = Field(description="The single most important improvement")


class CliffhangerMoment(BaseModel):
    moment: str = Field(description="The exact line or moment")
    scene_context: str
    mechanism: str = Field(description="Why this works as a cliffhanger - the narrative mechanism")
    tension_type: str = Field(description="e.g. Unresolved question, Emotional revelation, Threat")
    effectiveness_score: float = Field(ge=0.0, le=10.0)


class FullAnalysis(BaseModel):
    title: str
    summary: StorySummary
    emotional_arc: EmotionalArc
    engagement_score: EngagementScore
    suggestions: ImprovementSuggestions
    cliffhanger: Optional[CliffhangerMoment]
