"""
app.py - Main application integrating Gemini Live with camera and voice monitoring
Continuously streams camera metrics and voice data until speech detection ends
"""

import asyncio
import json
from collections import deque
from typing import Optional
from dotenv import load_dotenv
import os

# Import camera and voice streaming functions
from camera import stream_camera_metrics
from voice import stream_voice_with_text_vad

# Import Gemini Live SDK
from google import genai

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

MODEL_NAME = "gemini-2.0-flash-exp"
QUESTION_COUNT = 2

# Agent system instruction
AGENT_INSTRUCTION = """You are Sarah, a warm and encouraging interview coach conducting a mock interview practice session.

YOUR ROLE:
- Conduct a 2-question mock interview
- Analyze the candidate's posture, eye contact, speech pace, and volume in REAL-TIME
- Provide constructive feedback after each answer
- Maintain a friendly, casual tone throughout
- End with an overall performance summary

INTERVIEW FLOW:
1. Greet the candidate warmly and ask your first question: "Tell me a bit about yourself and what brings you here today."

2. While the candidate is speaking, you will receive CONTINUOUS updates with:
   - Camera metrics (posture, eye contact, movement)
   - Voice metrics (transcription, pace, volume, clarity)

3. After the candidate finishes speaking, you should:
   a) Analyze their overall performance across all metrics
   b) Give specific, actionable feedback in a friendly way
   c) If this is Question 1, ask your second question: "What do you think is your greatest strength and how has it helped you in the past?"
   d) If this is Question 2, provide a final summary with overall scores

FEEDBACK STYLE:
- Start with positive reinforcement (e.g., "Great job!" or "Nice work!")
- Be specific: mention exact behaviors you observed
- Keep feedback conversational and supportive
- Limit feedback to 2-3 key points per response
- Use phrases like "try to", "consider", "it might help to" rather than commands

METRICS INTERPRETATION:
Camera Metrics:
- Eye Contact: iris positions should be within specified ranges
- Posture: shoulder angle and head tilt should be near 180°
- Forward Lean: should be below 0.15
- Movement: head motion <15, hand motion <20

Voice Metrics:
- WPM: ideal 130-160, acceptable 120-180
- Volume: above -60 dB
- Clarity: above 0.70

FINAL SUMMARY (after Question 2):
- Congratulate them on completing the practice session
- Provide overall scores for: Engagement, Clarity, and Posture
- Give 2-3 concrete next steps for improvement
- End with encouragement

IMPORTANT:
- Respond naturally based on the metrics you observe
- Keep your tone warm, never harsh or discouraging
- Remember this is practice - the goal is to help them improve"""


# ============================================================================
# METRICS AGGREGATION
# ============================================================================

class MetricsAggregator:
    """Aggregates camera and voice metrics during a response"""
    
    def __init__(self, max_history: int = 100):
        self.camera_history = deque(maxlen=max_history)
        self.voice_text = ""
        self.voice_metrics = None
        
    def add_camera_metrics(self, metrics: dict):
        """Add camera metrics to history"""
        self.camera_history.append(metrics)
    
    def set_voice_metrics(self, metrics: dict):
        """Set final voice metrics when speech completes"""
        self.voice_metrics = metrics
        self.voice_text = metrics.get('text', '')
    
    def get_average_camera_metrics(self) -> dict:
        """Calculate average camera metrics"""
        if not self.camera_history:
            return {}
        
        # Average numeric metrics
        avg_metrics = {
            'shoulder_angle': 0,
            'head_tilt': 0,
            'forward_lean': 0,
            'head_motion_score': 0,
            'hand_motion_score': 0,
            'eye_contact_maintained_pct': 0,
            'left_iris_relative': 0,
            'right_iris_relative': 0
        }
        
        eye_contact_count = 0
        valid_iris_count = 0
        
        for metrics in self.camera_history:
            avg_metrics['shoulder_angle'] += metrics.get('shoulder_angle', 180)
            avg_metrics['head_tilt'] += metrics.get('head_tilt', 180)
            avg_metrics['forward_lean'] += metrics.get('forward_lean', 0)
            avg_metrics['head_motion_score'] += metrics.get('head_motion_score', 0)
            avg_metrics['hand_motion_score'] += metrics.get('hand_motion_score', 0)
            
            if metrics.get('eye_contact_maintained'):
                eye_contact_count += 1
            
            left_iris = metrics.get('left_iris_relative')
            right_iris = metrics.get('right_iris_relative')
            if left_iris is not None and right_iris is not None:
                avg_metrics['left_iris_relative'] += left_iris
                avg_metrics['right_iris_relative'] += right_iris
                valid_iris_count += 1
        
        count = len(self.camera_history)
        for key in ['shoulder_angle', 'head_tilt', 'forward_lean', 
                    'head_motion_score', 'hand_motion_score']:
            avg_metrics[key] /= count
        
        avg_metrics['eye_contact_maintained_pct'] = (eye_contact_count / count) * 100
        
        if valid_iris_count > 0:
            avg_metrics['left_iris_relative'] /= valid_iris_count
            avg_metrics['right_iris_relative'] /= valid_iris_count
        
        return avg_metrics
    
    def get_summary_message(self, question_num: int) -> str:
        """Generate summary message for the agent"""
        camera_avg = self.get_average_camera_metrics()
        
        message = f"Question {question_num} Response Complete:\n\n"
        
        # Voice metrics
        if self.voice_metrics:
            message += "VOICE ANALYSIS:\n"
            message += f"- Transcript: \"{self.voice_text}\"\n"
            message += f"- Words Per Minute: {self.voice_metrics.get('words_per_minute', 0):.0f} (ideal: 130-160)\n"
            message += f"- Volume: {self.voice_metrics.get('volume_db', -50):.1f} dB (min: -60 dB)\n"
            message += f"- Clarity Score: {self.voice_metrics.get('clarity_score', 0.8):.2f} (min: 0.70)\n"
            message += f"- Speech Duration: {self.voice_metrics.get('speech_duration', 0):.1f}s\n\n"
        
        # Camera metrics
        if camera_avg:
            message += "CAMERA ANALYSIS (Averaged):\n"
            message += f"- Eye Contact Maintained: {camera_avg['eye_contact_maintained_pct']:.0f}% of time\n"
            message += f"- Left Iris Position: {camera_avg['left_iris_relative']:.3f} (ideal: 2.75-2.85)\n"
            message += f"- Right Iris Position: {camera_avg['right_iris_relative']:.3f} (ideal: -1.95 to -1.875)\n"
            message += f"- Shoulder Angle: {camera_avg['shoulder_angle']:.1f}° (ideal: ~180°)\n"
            message += f"- Head Tilt: {camera_avg['head_tilt']:.1f}° (ideal: ~180°)\n"
            message += f"- Forward Lean: {camera_avg['forward_lean']:.2f} (max: 0.15)\n"
            message += f"- Head Motion: {camera_avg['head_motion_score']:.1f} px/frame (max: 15.0)\n"
            message += f"- Hand Motion: {camera_avg['hand_motion_score']:.1f} px/frame (max: 20.0)\n\n"
        
        message += "Please provide feedback on this response."
        
        return message
    
    def reset(self):
        """Reset for next question"""
        self.camera_history.clear()
        self.voice_text = ""
        self.voice_metrics = None


# ============================================================================
# GEMINI LIVE SESSION
# ============================================================================

class InterviewSession:
    """Manages Gemini Live session for the interview"""
    
    def __init__(self):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.session = None
        self.current_question = 0
        self.aggregator = MetricsAggregator()
        
    async def start_session(self):
        """Initialize Gemini Live session"""
        print("\n🎬 Starting Gemini Live Session...")
        
        config = {
            "generation_config": {
                "response_modalities": ["TEXT"]
            }
        }
        
        self.session = self.client.aio.live.connect(
            model=MODEL_NAME,
            config=config
        )
        
        # Send system instruction
        await self.session.send(AGENT_INSTRUCTION, end_of_turn=True)
        
        print("✅ Session started!\n")
    
    async def send_greeting(self):
        """Start the interview"""
        print("👤 User: Hello! I'm ready to start my practice interview.\n")
        await self.session.send("Hello! I'm ready to start my practice interview.", end_of_turn=True)
        
        # Get agent's response
        async for response in self.session.receive():
            if response.text:
                print(f"🤖 Sarah: {response.text}\n")
                break
        
        self.current_question = 1
    
    async def monitor_and_respond(self):
        """Monitor camera and voice until speech ends, then send to agent"""
        
        # Start camera stream in background
        camera_task = asyncio.create_task(self._monitor_camera())
        
        # Start voice stream (blocking until speech ends)
        print(f"\n📹 Monitoring camera and voice for Question {self.current_question}...")
        print("🎤 Speak now... (will detect when you stop)\n")
        
        voice_complete = False
        
        try:
            for voice_data in stream_voice_with_text_vad(
                no_speech_duration=2.0,
                transcribe_interval=0.5
            ):
                if voice_data["type"] == "speech_started":
                    print("🔴 Speech detected, recording...")
                
                elif voice_data["type"] == "status":
                    # Show real-time transcription
                    text = voice_data.get("text", "")
                    if text:
                        print(f"\r💬 {text[:70]}...", end="", flush=True)
                
                elif voice_data["type"] == "speech_complete":
                    print("\n✅ Speech complete!\n")
                    self.aggregator.set_voice_metrics(voice_data["metrics"])
                    voice_complete = True
                    break
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
        
        finally:
            # Stop camera monitoring
            camera_task.cancel()
            try:
                await camera_task
            except asyncio.CancelledError:
                pass
        
        if voice_complete:
            # Send aggregated metrics to agent
            await self._send_response_to_agent()
    
    async def _monitor_camera(self):
        """Background task to continuously monitor camera"""
        try:
            for camera_metrics in stream_camera_metrics(show_video=True):
                self.aggregator.add_camera_metrics(camera_metrics)
                await asyncio.sleep(0.033)  # ~30 FPS
        except asyncio.CancelledError:
            pass
    
    async def _send_response_to_agent(self):
        """Send aggregated metrics to agent and get feedback"""
        summary = self.aggregator.get_summary_message(self.current_question)
        
        print(f"\n{'='*70}")
        print("📤 Sending to Agent:")
        print(f"{'='*70}")
        print(summary[:300] + "..." if len(summary) > 300 else summary)
        print(f"{'='*70}\n")
        
        # Send to Gemini Live
        await self.session.send(summary, end_of_turn=True)
        
        # Receive agent's response
        print("🤖 Sarah: ", end="", flush=True)
        full_response = ""
        
        async for response in self.session.receive():
            if response.text:
                print(response.text, end="", flush=True)
                full_response += response.text
            
            if response.server_content and response.server_content.turn_complete:
                break
        
        print("\n")
        
        # Move to next question or end
        self.current_question += 1
        self.aggregator.reset()
        
        return full_response
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

async def main():
    """Main application flow"""
    print("\n" + "="*70)
    print("🎓 INTERVIEW COACH - LIVE SESSION")
    print("="*70)
    print("\nThis session will:")
    print("1. Ask you 2 interview questions")
    print("2. Monitor your camera (posture, eye contact, movement)")
    print("3. Monitor your voice (pace, volume, clarity)")
    print("4. Provide real-time feedback after each answer")
    print("\nPress Ctrl+C to stop at any time.")
    print("="*70 + "\n")
    
    session = InterviewSession()
    
    try:
        # Start Gemini Live session
        await session.start_session()
        
        # Send greeting and get first question
        await session.send_greeting()
        
        # Answer questions
        while session.current_question <= QUESTION_COUNT:
            await session.monitor_and_respond()
            
            if session.current_question <= QUESTION_COUNT:
                input("\n⏸️  Press Enter when ready for next question...")
        
        print("\n" + "="*70)
        print("🎉 Interview Practice Complete!")
        print("="*70)
        print("\nGreat job! Review Sarah's feedback to improve your interview skills.")
        print("="*70 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Session interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await session.close()
        print("👋 Session closed. Goodbye!\n")


if __name__ == "__main__":
    asyncio.run(main())