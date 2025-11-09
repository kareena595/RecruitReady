"""
agent.py - AI Interview Coach using Google's Agent Development Kit (ADK)
Modified to output structured JSON responses with metrics and feedback
"""

from dotenv import load_dotenv
import os
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import json
from typing import Dict, Any, List

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Check your .env file.")

os.environ["GOOGLE_API_KEY"] = api_key

# Quick connection test (optional)
client = genai.Client(api_key=api_key)
print("✅ Connected to Gemini")

# --- Constants ---
APP_NAME = "agent"
USER_ID = "test_user_interview"
SESSION_ID = "session_interview_001"
MODEL_NAME = "gemini-2.0-flash-exp"

# ============================================================================
# INPUT SCHEMA - Define what data the agent expects
# ============================================================================

class InterviewInput(BaseModel):
    """Input schema for interview agent"""
    message: str = Field(description="The message or metrics data to send to the interview coach")
    camera_data: Dict[str, Any] | None = Field(default=None, description="Optional camera metrics")
    voice_data: Dict[str, Any] | None = Field(default=None, description="Optional voice metrics")
    question_num: int | None = Field(default=None, description="Current question number")


# ============================================================================
# TOOL FUNCTIONS - These analyze metrics and return scores/feedback
# ============================================================================

def analyze_eye_contact(left_iris: float, right_iris: float, maintained: bool, eye_contact_percentage: float) -> dict:
    """Analyze eye contact quality based on iris position"""
    LEFT_EYE_MIN = 2.75
    LEFT_EYE_MAX = 2.85
    RIGHT_EYE_MIN = -1.95
    RIGHT_EYE_MAX = -1.875
    
    left_in_range = LEFT_EYE_MIN <= left_iris <= LEFT_EYE_MAX
    right_in_range = RIGHT_EYE_MIN <= right_iris <= RIGHT_EYE_MAX
    
    if not maintained or eye_contact_percentage < 70:
        return {
            "score": 65,
            "feedback": "Try to maintain more consistent eye contact with the interviewer",
            "severity": "moderate"
        }
    elif left_in_range and right_in_range and eye_contact_percentage >= 90:
        return {
            "score": 95,
            "feedback": "Excellent eye contact! You maintained focus throughout.",
            "severity": "none"
        }
    elif (left_in_range or right_in_range) and eye_contact_percentage >= 85:
        return {
            "score": 85,
            "feedback": "Your eye contact is strong, maintaining it nearly " + str(round(eye_contact_percentage)) + "% of the time. Just try to keep your eyes centered.",
            "severity": "minor"
        }
    else:
        return {
            "score": 75,
            "feedback": "Try to focus more directly on the camera - your gaze is wandering a bit",
            "severity": "moderate"
        }


def analyze_posture(shoulder_angle: float, head_tilt: float, forward_lean: float) -> dict:
    """Analyze overall posture quality"""
    SHOULDER_ANGLE_MIN = 165
    SHOULDER_ANGLE_MAX = 195
    HEAD_TILT_MIN = 165
    HEAD_TILT_MAX = 195
    FORWARD_LEAN_THRESHOLD = 0.15
    
    feedback_items = []
    scores = []
    
    if SHOULDER_ANGLE_MIN <= shoulder_angle <= SHOULDER_ANGLE_MAX:
        shoulder_deviation = abs(shoulder_angle - 180)
        scores.append(95 if shoulder_deviation <= 5 else 85)
    else:
        scores.append(70)
        feedback_items.append("keep your shoulders level and relaxed")
    
    if HEAD_TILT_MIN <= head_tilt <= HEAD_TILT_MAX:
        head_deviation = abs(head_tilt - 180)
        scores.append(95 if head_deviation <= 5 else 85)
    else:
        scores.append(70)
        feedback_items.append("keep your head straight")
    
    if forward_lean <= 0.10:
        scores.append(95)
    elif forward_lean <= FORWARD_LEAN_THRESHOLD:
        scores.append(85)
    else:
        scores.append(70)
        feedback_items.append("sit back a bit, you're leaning forward quite a lot")
    
    avg_score = sum(scores) / len(scores)
    
    if feedback_items:
        feedback = "For your posture, try to " + " and ".join(feedback_items) + "."
        severity = "moderate" if avg_score < 80 else "minor"
    else:
        feedback = "Overall, your posture seems great! You have good scores for head tilt and shoulder angle."
        severity = "none"
    
    return {
        "score": avg_score,
        "feedback": feedback,
        "severity": severity
    }


def analyze_movement(head_motion: float, hand_motion: float) -> dict:
    """Analyze body movement and fidgeting"""
    HEAD_MOTION_THRESHOLD = 15.0
    HAND_MOTION_THRESHOLD = 20.0
    
    feedback_items = []
    
    if head_motion < HEAD_MOTION_THRESHOLD * 0.5:
        head_score = 95
    elif head_motion < HEAD_MOTION_THRESHOLD:
        head_score = 90
    elif head_motion < HEAD_MOTION_THRESHOLD * 1.5:
        head_score = 75
        feedback_items.append("reduce head movement slightly")
    else:
        head_score = 65
        feedback_items.append("minimize head movement - try to keep your head steady")
    
    if hand_motion < HAND_MOTION_THRESHOLD * 0.5:
        hand_score = 95
    elif hand_motion < HAND_MOTION_THRESHOLD:
        hand_score = 90
    elif hand_motion < HAND_MOTION_THRESHOLD * 1.5:
        hand_score = 75
        feedback_items.append("try to reduce hand fidgeting")
    else:
        hand_score = 65
        feedback_items.append("your hand movements are quite distracting - keep them still or use purposeful gestures")
    
    avg_score = (head_score + hand_score) / 2
    
    if feedback_items:
        feedback = "Try to " + " and ".join(feedback_items) + "."
        severity = "moderate" if avg_score < 75 else "minor"
    else:
        feedback = "Your body movement is well controlled with minimal fidgeting."
        severity = "none"
    
    return {
        "score": avg_score,
        "feedback": feedback,
        "severity": severity
    }


def analyze_speech_pace(words_per_minute: float) -> dict:
    """Analyze speaking pace"""
    MIN_WPM = 120
    MAX_WPM = 180
    IDEAL_WPM_MIN = 130
    IDEAL_WPM_MAX = 160
    
    if IDEAL_WPM_MIN <= words_per_minute <= IDEAL_WPM_MAX:
        return {"score": 95, "feedback": "Your speaking pace is excellent - natural and easy to follow.", "severity": "none"}
    elif words_per_minute < MIN_WPM:
        if words_per_minute < MIN_WPM * 0.8:
            return {"score": 70, "feedback": "You can speak quite a bit faster - try to increase your pace", "severity": "moderate"}
        else:
            return {"score": 80, "feedback": "Your pace is mostly good, but try to pick up the speed slightly", "severity": "minor"}
    elif words_per_minute < IDEAL_WPM_MIN:
        return {"score": 85, "feedback": "Your pace is good, maybe pick up the speed just a little", "severity": "minor"}
    elif words_per_minute <= MAX_WPM:
        return {"score": 85, "feedback": "You're speaking a little fast - try to slow down and breathe", "severity": "minor"}
    else:
        return {"score": 70, "feedback": "You're speaking quite fast - take your time and pause between thoughts", "severity": "moderate"}


def analyze_volume(volume_db: float) -> dict:
    """Analyze speaking volume"""
    MIN_VOLUME_DB = -60
    
    if volume_db >= -50:
        return {"score": 95, "feedback": "Your volume is perfect - clear and confident.", "severity": "none"}
    elif volume_db >= MIN_VOLUME_DB:
        return {"score": 85, "feedback": "Your volume is good, just try to project a little more", "severity": "minor"}
    elif volume_db >= -65:
        return {"score": 75, "feedback": "Try to speak up a bit - your volume is on the quiet side", "severity": "moderate"}
    else:
        return {"score": 65, "feedback": "Try to speak up - your volume is quite low", "severity": "moderate"}


def analyze_clarity(clarity_score: float) -> dict:
    """Analyze speech clarity"""
    MIN_CLARITY_SCORE = 0.7
    
    if clarity_score >= 0.9:
        return {"score": 95, "feedback": "Your speech clarity is excellent.", "severity": "none"}
    elif clarity_score >= 0.8:
        return {"score": 90, "feedback": "The clarity score is good.", "severity": "none"}
    elif clarity_score >= MIN_CLARITY_SCORE:
        return {"score": 80, "feedback": "Try to enunciate a bit more clearly", "severity": "minor"}
    else:
        return {"score": 70, "feedback": "Focus on speaking more clearly and enunciating your words", "severity": "moderate"}


def check_transcript_issues(transcript: str) -> List[str]:
    """Check for issues in the transcript like repetition or filler words"""
    issues = []
    
    # Check for repetition
    words = transcript.lower().split()
    if len(words) > 5:
        # Look for repeated phrases
        for i in range(len(words) - 4):
            phrase = " ".join(words[i:i+3])
            rest_of_text = " ".join(words[i+3:])
            if phrase in rest_of_text:
                issues.append(f"Noticed repetition in your response ('{phrase}'). Try to pause and gather your thoughts before speaking to avoid repeating yourself.")
                break
    
    # Count filler words
    filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually']
    filler_count = sum(1 for word in words if word in filler_words)
    if filler_count > len(words) * 0.05:  # More than 5% filler words
        issues.append("Try to reduce filler words like 'um', 'uh', and 'like' - pausing is better than filling silence.")
    
    return issues


def evaluate_overall_performance(all_scores: Dict[str, float], question_count: int) -> dict:
    """Generate overall performance evaluation for the entire interview session"""
    
    # Calculate category averages
    engagement_score = all_scores.get("eye_contact_avg", 80)
    clarity_score = (all_scores.get("clarity_avg", 80) + all_scores.get("pace_avg", 80) + all_scores.get("volume_avg", 80)) / 3
    posture_score = (all_scores.get("posture_avg", 80) + all_scores.get("movement_avg", 80)) / 2
    
    overall_score = (engagement_score + clarity_score + posture_score) / 3
    
    # Determine performance level
    if overall_score >= 90:
        performance_level = "Excellent"
        summary = "You demonstrated strong interview skills across all areas. Your presentation was professional and engaging throughout the session."
    elif overall_score >= 80:
        performance_level = "Good"
        summary = "You showed solid interview skills with room for minor improvements. Your overall presentation was confident and clear."
    elif overall_score >= 70:
        performance_level = "Fair"
        summary = "You have a good foundation but there are several areas to work on. Focus on the specific feedback provided to strengthen your presence."
    else:
        performance_level = "Needs Improvement"
        summary = "There are several key areas that need attention. Review the feedback carefully and practice these specific skills."
    
    # Generate improvement areas
    improvement_areas = []
    if engagement_score < 85:
        improvement_areas.append("Maintain more consistent eye contact with the camera")
    if clarity_score < 85:
        improvement_areas.append("Work on speech clarity, pacing, and volume projection")
    if posture_score < 85:
        improvement_areas.append("Focus on maintaining good posture and reducing fidgeting")
    
    if not improvement_areas:
        improvement_areas = ["Continue practicing to maintain your strong skills", "Consider adding more specific examples to your answers", "Work on varying your vocal tone for emphasis"]
    
    return {
        "performance_level": performance_level,
        "overall_score": round(overall_score, 1),
        "engagement_score": round(engagement_score, 1),
        "clarity_score": round(clarity_score, 1),
        "posture_score": round(posture_score, 1),
        "summary": summary,
        "questions_completed": question_count,
        "improvement_areas": improvement_areas[:3],
        "strengths": _identify_strengths(engagement_score, clarity_score, posture_score)
    }


def _identify_strengths(engagement: float, clarity: float, posture: float) -> List[str]:
    """Identify the candidate's strengths based on scores"""
    strengths = []
    if engagement >= 85:
        strengths.append("Strong eye contact and engagement")
    if clarity >= 85:
        strengths.append("Clear and well-paced speech delivery")
    if posture >= 85:
        strengths.append("Professional posture and body language")
    
    if not strengths:
        strengths = ["Completed the practice session", "Showed willingness to improve"]
    
    return strengths


# ============================================================================
# AGENT DEFINITION
# ============================================================================

interview_agent = LlmAgent(
    model=MODEL_NAME,
    name="Sarah",
    description="A friendly, supportive interview coach who provides structured JSON feedback",
    
    instruction="""You are Sarah, a warm and encouraging interview coach conducting a mock interview practice session.

YOUR ROLE:
- Conduct a 2-question mock interview
- Analyze the candidate's posture, eye contact, speech pace, and volume using the provided tools
- Provide structured JSON feedback after each answer
- Track performance across the entire session

INTERVIEW FLOW:

1. FIRST MESSAGE - When you receive "Hello! I'm ready to start my practice interview.":
   - Greet the candidate warmly
   - Ask your first question: "Tell me a bit about yourself and what brings you here today."
   - Output as plain text (not JSON)

2. AFTER QUESTION 1 - When you receive Camera Metrics and Speech Metrics:
   - Call ALL the analysis tools to evaluate performance:
     * analyze_eye_contact(left_iris, right_iris, maintained, eye_contact_percentage)
     * analyze_posture(shoulder_angle, head_tilt, forward_lean)
     * analyze_movement(head_motion, hand_motion)
     * analyze_speech_pace(words_per_minute)
     * analyze_volume(volume_db)
     * analyze_clarity(clarity_score)
     * check_transcript_issues(transcript)
   
   - Output a JSON object with TWO keys:
     {
       "metrics_summary": {
         "eye_contact": {
           "percentage": <float>,
           "left_iris": <float>,
           "right_iris": <float>,
           "status": "maintained" or "wandering"
         },
         "posture": {
           "shoulder_angle": <float>,
           "head_tilt": <float>,
           "forward_lean": <float>,
           "status": "good" or "needs_adjustment"
         },
         "movement": {
           "head_motion": <float>,
           "hand_motion": <float>,
           "status": "controlled" or "fidgeting"
         },
         "speech": {
           "transcript": "<full transcript>",
           "duration": <float>,
           "words_per_minute": <float>,
           "volume_db": <float>,
           "clarity_score": <float>
         },
         "issues_detected": {<dict of issues>}
       },
       "feedback": {
         "posture_and_body_language": "<feedback about posture and movement>",
         "eye_contact": "<feedback about eye contact quality>",
         "speech_delivery": "<feedback about pace, clarity, and volume>",
         "overall_presentation": "<brief summary>",
         "next_question": "What do you think is your greatest strength and how has it helped you in the past?"
       }
     }

3. AFTER QUESTION 2 - When you receive the second set of metrics:
   - Call ALL the analysis tools again
   - Call evaluate_overall_performance with aggregated scores from both questions
   
   - Output a JSON object with THREE keys:
     {
       "metrics_summary": {<same structure as Question 1>},
       "feedback": {
         "posture_and_body_language": "<feedback about posture and movement>",
         "eye_contact": "<feedback about eye contact quality>",
         "speech_delivery": "<feedback aobut pace, clarity, volume>",
         "overall_presentation": "<brief summary>"
       },
       "session_summary": {
         "performance_level": "<from evaluate_overall_performance>",
         "overall_score": <float>,
         "engagement_score": <float>,
         "clarity_score": <float>,
         "posture_score": <float>,
         "summary": "<overall session summary>",
         "questions_completed": 2,
         "improvement_areas": [<list of 2-3 areas>],
         "strengths": [<list of strengths>],
         "encouragement": "<warm closing message>"
       }
     }

FEEDBACK GUIDELINES:
- Always call the analysis tools and use their feedback but do not cite them directly. DO NOT SAY "According to the tool..."
- Be specific and reference actual metrics
- Start feedback sections with positive reinforcement when appropriate
- Be conversational and supportive
- For speech_delivery, combine feedback from pace, clarity, volume, and transcript issues
- Keep the tone warm and encouraging

IMPORTANT:
- Question 1 response: JSON with "metrics_summary" and "feedback" (includes next_question)
- Question 2 response: JSON with "metrics_summary", "feedback", and "session_summary"
- Always output valid JSON that can be parsed by json.loads()
- Track scores across questions to provide accurate session_summary""",
    
    tools=[
        analyze_eye_contact,
        analyze_posture,
        analyze_movement,
        analyze_speech_pace,
        analyze_volume,
        analyze_clarity,
        check_transcript_issues,
        evaluate_overall_performance
    ],
    
    input_schema=InterviewInput,
    output_key="interview_response"
)

# ============================================================================
# SESSION AND RUNNER SETUP
# ============================================================================

session_service = InMemorySessionService()
interview_runner = Runner(
    agent=interview_agent,
    app_name=APP_NAME,
    session_service=session_service
)

# ============================================================================
# EXPORT FOR APP.PY
# ============================================================================

__all__ = [
    "APP_NAME",
    "USER_ID",
    "SESSION_ID",
    "interview_runner",
    "interview_agent",
    "session_service",
    "types",
    "InterviewInput"
]