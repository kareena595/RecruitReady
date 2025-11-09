"""
FastAPI + Socket.IO backend for real-time video/audio processing
Modified to use split camera and speech feedback agents
"""

import asyncio
import base64
import cv2
import numpy as np
import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from collections import deque
from typing import Optional, Dict, Any
import json
import time
import threading
import queue

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera import PostureDetector
from voice import stream_voice_with_text_vad

# Add interview_agent folder to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'interview_agent'))
from agent import (
    APP_NAME, USER_ID, SESSION_ID,
    camera_runner, speech_runner, coordinator_runner, summary_runner,
    session_service, types
)

# Initialize Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=False,
    engineio_logger=False
)

# Initialize FastAPI
app = FastAPI(title="Mock Interview API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wrap FastAPI app with Socket.IO
socket_app = socketio.ASGIApp(sio, app, socketio_path='/socket.io')

# ============================================================================
# GLOBAL STATE MANAGEMENT
# ============================================================================

class InterviewSession:
    """Manages the state of an interview session"""
    
    def __init__(self):
        self.camera_active = False
        self.interview_active = False
        self.recording_active = False
        
        # Camera state
        self.video_capture = None
        self.posture_detector = None
        self.current_frame = None
        self.camera_thread = None
        self.camera_stop_event = threading.Event()
        
        # Metrics storage
        self.camera_metrics_buffer = []
        self.voice_queue = queue.Queue()
        self.voice_stop_event = threading.Event()
        self.manual_stop_event = threading.Event()
        self.voice_thread = None
        self.current_transcript = ""
        self.speech_metrics = None
        
        # Interview state
        self.current_question = 0
        self.total_questions = 2
        self.session_initialized = False
        
        # Score tracking for final summary
        self.all_scores = {
            "eye_contact_scores": [],
            "posture_scores": [],
            "movement_scores": [],
            "pace_scores": [],
            "volume_scores": [],
            "clarity_scores": []
        }
        
        # Client tracking
        self.client_sid = None
    
    def reset_recording_state(self):
        """Reset state for next recording"""
        self.recording_active = False
        self.camera_metrics_buffer = []
        self.current_transcript = ""
        self.speech_metrics = None
        self.voice_stop_event.clear()
        self.manual_stop_event.clear()
    
    def cleanup(self):
        """Full cleanup of session"""
        self.camera_stop_event.set()
        self.voice_stop_event.set()
        
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        
        if self.voice_thread and self.voice_thread.is_alive():
            self.voice_thread.join(timeout=2.0)
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        if self.posture_detector:
            self.posture_detector.release()
            self.posture_detector = None
        
        self.camera_active = False
        self.interview_active = False
        self.recording_active = False
        self.current_question = 0
        self.camera_stop_event.clear()
        
        # Reset score tracking
        self.all_scores = {
            "eye_contact_scores": [],
            "posture_scores": [],
            "movement_scores": [],
            "pace_scores": [],
            "volume_scores": [],
            "clarity_scores": []
        }

# Global session instance
session = InterviewSession()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def aggregate_camera_metrics(all_metrics):
    """Calculate average camera metrics from collected data"""
    if not all_metrics:
        return {}
    
    total_frames = len(all_metrics)
    
    avg_metrics = {
        "shoulder_angle": sum(d.get('shoulder_angle', 180) for d in all_metrics) / total_frames,
        "head_tilt": sum(d.get('head_tilt', 180) for d in all_metrics) / total_frames,
        "forward_lean": sum(d.get('forward_lean', 0) for d in all_metrics) / total_frames,
        "head_motion": sum(d.get('head_motion_score', 0) for d in all_metrics) / total_frames,
        "hand_motion": sum(d.get('hand_motion_score', 0) for d in all_metrics) / total_frames,
    }
    
    # Calculate eye contact percentage
    eye_contact_frames = sum(1 for d in all_metrics if d.get('eye_contact_maintained', True))
    avg_metrics['eye_contact_percentage'] = (eye_contact_frames / total_frames) * 100
    
    # Get most recent iris positions
    last_frame = all_metrics[-1]
    avg_metrics['left_iris_relative'] = last_frame.get('left_iris_relative')
    avg_metrics['right_iris_relative'] = last_frame.get('right_iris_relative')
    avg_metrics['eye_contact_maintained'] = last_frame.get('eye_contact_maintained', True)
    
    avg_metrics['total_frames'] = total_frames
    
    return avg_metrics


def voice_collector_thread(voice_queue, stop_event, manual_stop_event, sid):
    """Thread function to collect voice metrics"""
    speech_metrics = None
    accumulated_text = ""
    speech_detected = False
    
    try:
        print(f"[VOICE] Voice collector started for {sid}...")
        
        for data in stream_voice_with_text_vad(silence_duration=3.0, transcribe_interval=5.0):
            
            if data["type"] == "speech_started":
                print(f"[VOICE] Speech detected!")
                speech_detected = True
                asyncio.run(sio.emit('speech_status', {
                    'status': 'speaking',
                    'message': 'Speech detected'
                }, to=sid))
            
            elif data["type"] == "status":
                current_text = data.get("text", "")
                accumulated_text = current_text
                word_count = data.get("word_count", 0)
                
                print(f"[VOICE] Transcript update: {len(current_text)} chars, {word_count} words")
                
                asyncio.run(sio.emit('transcript_update', {
                    'transcript': current_text,
                    'word_count': word_count
                }, to=sid))
            
            elif data["type"] == "speech_complete":
                speech_metrics = data["metrics"]
                print(f"[VOICE] Speech complete! Text: {len(speech_metrics.get('text', ''))} chars")
                
                asyncio.run(sio.emit('speech_complete', {
                    'status': 'complete',
                    'transcript': speech_metrics.get('text', ''),
                    'metrics': speech_metrics
                }, to=sid))
                
                stop_event.set()
                break
            
            if manual_stop_event.is_set() and speech_detected:
                print(f"[VOICE] Manual stop detected - finalizing...")
                if accumulated_text and len(accumulated_text.strip()) > 0:
                    word_count = len(accumulated_text.split())
                    speech_metrics = {
                        'text': accumulated_text.strip(),
                        'speech_duration': (word_count / 140) * 60 if word_count > 0 else 0.0,
                        'words_per_minute': 140,
                        'volume_db': -55.0,
                        'clarity_score': 0.75,
                        'timestamp': time.time()
                    }
                    print(f"[VOICE] Captured {word_count} words from partial transcript")
                stop_event.set()
                break
    
    except Exception as e:
        print(f"[VOICE ERROR] {e}")
        import traceback
        traceback.print_exc()
        stop_event.set()
    finally:
        voice_queue.put(speech_metrics)
        print(f"[VOICE] Voice collector stopped")


def camera_collector_thread(sid):
    """Thread function to collect camera metrics"""
    print(f"[CAMERA] Camera collector started for {sid}")
    
    while not session.camera_stop_event.is_set():
        try:
            if not session.video_capture or not session.video_capture.isOpened():
                time.sleep(0.1)
                continue
            
            success, frame = session.video_capture.read()
            if not success:
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            annotated_frame, metrics = session.posture_detector.process_frame(frame)
            session.current_frame = annotated_frame
            
            if session.recording_active and metrics:
                session.camera_metrics_buffer.append(metrics.to_dict())
            
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            metrics_data = metrics.to_dict() if metrics else {}
            
            asyncio.run(sio.emit('video_frame', {
                'frame': f'data:image/jpeg;base64,{frame_base64}',
                'metrics': metrics_data,
                'recording': session.recording_active
            }, to=sid))
            
            time.sleep(0.033)
            
        except Exception as e:
            print(f"[CAMERA ERROR] {e}")
            time.sleep(0.1)
    
    print(f"[CAMERA] Camera collector stopped")


async def get_camera_feedback_async(camera_metrics, question_num):
    """Get camera/posture feedback from agent"""
    
    camera_text = f"""Camera Metrics for Question {question_num} (Averaged over {camera_metrics.get('total_frames', 0)} frames):
- Eye Contact: {camera_metrics.get('eye_contact_percentage', 0):.1f}% maintained
- Left Iris Position: {camera_metrics.get('left_iris_relative', 0):.3f}
- Right Iris Position: {camera_metrics.get('right_iris_relative', 0):.3f}
- Shoulder Angle: {camera_metrics.get('shoulder_angle', 180):.1f}°
- Head Tilt: {camera_metrics.get('head_tilt', 180):.1f}°
- Forward Lean: {camera_metrics.get('forward_lean', 0.0):.3f}
- Head Motion: {camera_metrics.get('head_motion', 0):.1f} px/frame
- Hand Motion: {camera_metrics.get('hand_motion', 0):.1f} px/frame

Analyze these metrics and provide concise feedback in JSON format."""
    
    print(f"[CAMERA_AGENT] Getting camera feedback for Q{question_num}...")
    
    user_content = types.Content(
        role='user',
        parts=[types.Part(text=camera_text)]
    )
    
    response_text = ""
    async for event in camera_runner.run_async(
        user_id=USER_ID,
        session_id=f"{SESSION_ID}_camera",
        new_message=user_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text
    
    print(f"[CAMERA_AGENT] Response: {len(response_text)} chars")
    return response_text


async def get_speech_feedback_async(speech_metrics, question_num):
    """Get speech feedback from agent"""
    
    if speech_metrics:
        speech_text = f"""Speech Metrics for Question {question_num}:
- Transcript: "{speech_metrics.get('text', 'N/A')}"
- Duration: {speech_metrics.get('speech_duration', 0):.1f}s
- Words Per Minute: {speech_metrics.get('words_per_minute', 0):.0f}
- Volume: {speech_metrics.get('volume_db', 0):.1f} dB
- Clarity Score: {speech_metrics.get('clarity_score', 0):.2f}

Analyze these metrics and provide concise feedback in JSON format."""
    else:
        speech_text = "No speech detected for this answer. Provide brief feedback about the lack of response."
    
    print(f"[SPEECH_AGENT] Getting speech feedback for Q{question_num}...")
    
    user_content = types.Content(
        role='user',
        parts=[types.Part(text=speech_text)]
    )
    
    response_text = ""
    async for event in speech_runner.run_async(
        user_id=USER_ID,
        session_id=f"{SESSION_ID}_speech",
        new_message=user_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text
    
    print(f"[SPEECH_AGENT] Response: {len(response_text)} chars")
    return response_text


async def get_next_question_async(question_num):
    """Get next question from coordinator agent"""
    
    if question_num == 1:
        message = "Please provide Question 2 now."
    else:
        message = "Interview complete."
    
    user_content = types.Content(
        role='user',
        parts=[types.Part(text=message)]
    )
    
    response_text = ""
    async for event in coordinator_runner.run_async(
        user_id=USER_ID,
        session_id=f"{SESSION_ID}_coordinator",
        new_message=user_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text
    
    return response_text


async def get_overall_summary_async():
    """Get overall interview summary"""
    
    # Calculate averages from collected scores
    avg_scores = {
        "eye_contact_avg": sum(session.all_scores["eye_contact_scores"]) / len(session.all_scores["eye_contact_scores"]) if session.all_scores["eye_contact_scores"] else 80,
        "posture_avg": sum(session.all_scores["posture_scores"]) / len(session.all_scores["posture_scores"]) if session.all_scores["posture_scores"] else 80,
        "movement_avg": sum(session.all_scores["movement_scores"]) / len(session.all_scores["movement_scores"]) if session.all_scores["movement_scores"] else 80,
        "pace_avg": sum(session.all_scores["pace_scores"]) / len(session.all_scores["pace_scores"]) if session.all_scores["pace_scores"] else 80,
        "volume_avg": sum(session.all_scores["volume_scores"]) / len(session.all_scores["volume_scores"]) if session.all_scores["volume_scores"] else 80,
        "clarity_avg": sum(session.all_scores["clarity_scores"]) / len(session.all_scores["clarity_scores"]) if session.all_scores["clarity_scores"] else 80,
    }
    
    summary_text = f"""Generate overall interview summary based on these scores:
- Eye Contact Average: {avg_scores['eye_contact_avg']:.1f}
- Posture Average: {avg_scores['posture_avg']:.1f}
- Movement Average: {avg_scores['movement_avg']:.1f}
- Pace Average: {avg_scores['pace_avg']:.1f}
- Volume Average: {avg_scores['volume_avg']:.1f}
- Clarity Average: {avg_scores['clarity_avg']:.1f}

Questions completed: 2"""
    
    print(f"[SUMMARY_AGENT] Getting overall summary...")
    
    user_content = types.Content(
        role='user',
        parts=[types.Part(text=summary_text)]
    )
    
    response_text = ""
    async for event in summary_runner.run_async(
        user_id=USER_ID,
        session_id=f"{SESSION_ID}_summary",
        new_message=user_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text
    
    print(f"[SUMMARY_AGENT] Response: {len(response_text)} chars")
    return response_text


def parse_json_response(response_text):
    """Parse JSON from agent response, handling markdown wrapping"""
    try:
        cleaned = response_text.strip()
        
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        
        cleaned = cleaned.strip()
        return json.loads(cleaned)
    
    except json.JSONDecodeError as e:
        print(f"[JSON ERROR] {e}")
        print(f"[JSON ERROR] Trying to extract JSON from response...")
        
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        return None


# ============================================================================
# SOCKET.IO EVENT HANDLERS
# ============================================================================

@sio.event
async def connect(sid, environ):
    """Called when a client connects"""
    print(f"✓ Client connected: {sid}")
    session.client_sid = sid
    await sio.emit('connection_status', {
        'status': 'connected',
        'message': 'Connected to backend'
    }, to=sid)


@sio.event
async def disconnect(sid):
    """Called when a client disconnects"""
    print(f"✗ Client disconnected: {sid}")
    session.cleanup()


@sio.event
async def start_camera(sid, data):
    """Start camera preview"""
    print(f"[CAMERA] Client {sid} requested camera start")
    
    if session.camera_active:
        await sio.emit('error', {'message': 'Camera already active'}, to=sid)
        return
    
    try:
        if session.posture_detector is None:
            session.posture_detector = PostureDetector()
        
        session.video_capture = cv2.VideoCapture(0)
        if not session.video_capture.isOpened():
            raise Exception("Failed to open camera")
        
        session.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        session.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        success, test_frame = session.video_capture.read()
        if not success:
            raise Exception("Camera opened but failed to read frame")
        
        session.camera_active = True
        session.camera_stop_event.clear()
        
        session.camera_thread = threading.Thread(
            target=camera_collector_thread,
            args=(sid,),
            daemon=True
        )
        session.camera_thread.start()
        
        await sio.emit('camera_status', {
            'status': 'started',
            'message': 'Camera preview started'
        }, to=sid)
        
        print(f"[CAMERA] Camera started successfully")
        
    except Exception as e:
        print(f"[ERROR] Failed to start camera: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit('error', {'message': f'Failed to start camera: {str(e)}'}, to=sid)


@sio.event
async def stop_camera(sid, data):
    """Stop camera preview"""
    print(f"[CAMERA] Client {sid} requested camera stop")
    session.cleanup()
    await sio.emit('camera_status', {'status': 'stopped'}, to=sid)


@sio.event
async def start_interview(sid, data):
    """Initialize the interview and send first question"""
    print(f"[INTERVIEW] Client {sid} requested interview start")
    
    if not session.camera_active:
        await sio.emit('error', {'message': 'Please start camera first'}, to=sid)
        return
    
    try:
        if not session.session_initialized:
            try:
                await session_service.create_session(
                    app_name=f"{APP_NAME}_camera",
                    user_id=USER_ID,
                    session_id=f"{SESSION_ID}_camera"
                )
                await session_service.create_session(
                    app_name=f"{APP_NAME}_speech",
                    user_id=USER_ID,
                    session_id=f"{SESSION_ID}_speech"
                )
                await session_service.create_session(
                    app_name=f"{APP_NAME}_coordinator",
                    user_id=USER_ID,
                    session_id=f"{SESSION_ID}_coordinator"
                )
                await session_service.create_session(
                    app_name=f"{APP_NAME}_summary",
                    user_id=USER_ID,
                    session_id=f"{SESSION_ID}_summary"
                )
                session.session_initialized = True
                print(f"[AGENT] Sessions created")
            except Exception as e:
                print(f"[AGENT] Session note: {e}")
        
        session.interview_active = True
        session.current_question = 1
        
        # Get greeting and first question from coordinator
        print(f"[COORDINATOR] Getting initial greeting...")
        user_content = types.Content(
            role='user',
            parts=[types.Part(text="Hello! I'm ready to start my practice interview.")]
        )
        
        greeting = ""
        async for event in coordinator_runner.run_async(
            user_id=USER_ID,
            session_id=f"{SESSION_ID}_coordinator",
            new_message=user_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                greeting = event.content.parts[0].text
        
        if not greeting:
            greeting = "Hello! Welcome to your mock interview. Let's begin. Tell me a bit about yourself and what brings you here today."
        
        await sio.emit('interview_started', {
            'status': 'started',
            'question': greeting,
            'question_number': 1,
            'total_questions': session.total_questions
        }, to=sid)
        
        print(f"[INTERVIEW] Started - Question 1 sent")
        
    except Exception as e:
        print(f"[ERROR] Failed to start interview: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit('error', {'message': f'Failed to start interview: {str(e)}'}, to=sid)


@sio.event
async def start_recording(sid, data):
    """Start recording answer"""
    print(f"[RECORDING] Client {sid} requested recording start")
    
    if not session.interview_active:
        await sio.emit('error', {'message': 'Please start interview first'}, to=sid)
        return
    
    if session.recording_active:
        await sio.emit('error', {'message': 'Already recording'}, to=sid)
        return
    
    try:
        session.recording_active = True
        session.camera_metrics_buffer = []
        session.current_transcript = ""
        session.speech_metrics = None
        session.voice_stop_event.clear()
        session.manual_stop_event.clear()
        
        while not session.voice_queue.empty():
            try:
                session.voice_queue.get_nowait()
            except queue.Empty:
                break
        
        session.voice_thread = threading.Thread(
            target=voice_collector_thread,
            args=(session.voice_queue, session.voice_stop_event, session.manual_stop_event, sid),
            daemon=True
        )
        session.voice_thread.start()
        
        await sio.emit('recording_status', {
            'status': 'started',
            'message': 'Recording your answer...'
        }, to=sid)
        
        print(f"[RECORDING] Started for question {session.current_question}")
        
        asyncio.create_task(monitor_voice_completion(sid))
        
    except Exception as e:
        print(f"[ERROR] Failed to start recording: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit('error', {'message': f'Failed to start recording: {str(e)}'}, to=sid)


@sio.event
async def stop_recording(sid, data):
    """Stop recording manually"""
    print(f"[RECORDING] Client {sid} requested manual stop")
    
    if not session.recording_active:
        await sio.emit('error', {'message': 'Not currently recording'}, to=sid)
        return
    
    session.manual_stop_event.set()
    
    await sio.emit('recording_status', {
        'status': 'stopped',
        'message': 'Processing your answer...'
    }, to=sid)


async def monitor_voice_completion(sid):
    """Monitor voice thread completion and process answer"""
    print(f"[MONITOR] Monitoring voice completion...")
    
    loop = asyncio.get_event_loop()
    
    def wait_for_voice():
        session.voice_thread.join(timeout=120.0)
    
    await loop.run_in_executor(None, wait_for_voice)
    
    if not session.recording_active:
        print(f"[MONITOR] Recording already stopped")
        return
    
    print(f"[MONITOR] Voice thread completed, processing answer...")
    
    speech_metrics = None
    try:
        speech_metrics = session.voice_queue.get_nowait()
    except queue.Empty:
        print(f"[MONITOR] No speech data in queue")
    
    session.speech_metrics = speech_metrics
    session.recording_active = False
    
    await process_answer(sid)


async def process_answer(sid):
    """Process collected metrics and get AI feedback - SPLIT into camera and speech"""
    print(f"[PROCESSING] Processing answer for question {session.current_question}")
    
    try:
        # Aggregate camera metrics
        camera_summary = aggregate_camera_metrics(session.camera_metrics_buffer)
        
        print(f"[PROCESSING] Camera metrics: {len(session.camera_metrics_buffer)} frames")
        print(f"[PROCESSING] Speech metrics: {session.speech_metrics is not None}")
        
        # Get BOTH feedbacks in parallel
        camera_response, speech_response = await asyncio.gather(
            get_camera_feedback_async(camera_summary, session.current_question),
            get_speech_feedback_async(session.speech_metrics, session.current_question)
        )
        
        # Parse responses
        camera_feedback = parse_json_response(camera_response)
        speech_feedback = parse_json_response(speech_response)
        
        print(f"[PROCESSING] Camera feedback parsed: {camera_feedback is not None}")
        print(f"[PROCESSING] Speech feedback parsed: {speech_feedback is not None}")
        
        if camera_feedback:
            print(f"[PROCESSING] Camera feedback keys: {list(camera_feedback.keys())}")
        if speech_feedback:
            print(f"[PROCESSING] Speech feedback keys: {list(speech_feedback.keys())}")
        
        # Prepare response data
        response_data = {
            'question_number': session.current_question,
            'camera_feedback': camera_feedback or {"error": "Could not parse camera feedback"},
            'speech_feedback': speech_feedback or {"error": "Could not parse speech feedback"},
            'metrics_summary': {
                'camera': camera_summary,
                'speech': session.speech_metrics
            }
        }
        
        # Check if interview is complete
        if session.current_question >= session.total_questions:
            # Get overall summary
            print(f"[PROCESSING] Getting overall summary...")
            summary_response = await get_overall_summary_async()
            summary_data = parse_json_response(summary_response)
            
            response_data['session_summary'] = summary_data or {"error": "Could not parse summary"}
            response_data['interview_complete'] = True
            
            session.interview_active = False
            print(f"[INTERVIEW] Complete - Session summary sent")
        else:
            # Get next question
            next_q = await get_next_question_async(session.current_question)
            response_data['next_question'] = next_q
            response_data['interview_complete'] = False
            
            session.current_question += 1
            print(f"[INTERVIEW] Moving to question {session.current_question}")
        
        # Send combined feedback to frontend
        await sio.emit('interview_feedback', response_data, to=sid)
        
        # Reset for next recording
        session.reset_recording_state()
        
    except Exception as e:
        print(f"[ERROR] Failed to process answer: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit('error', {'message': f'Failed to process answer: {str(e)}'}, to=sid)


# ============================================================================
# HTTP ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "service": "Mock Interview API",
        "camera_active": session.camera_active,
        "interview_active": session.interview_active,
        "recording_active": session.recording_active
    }


@app.get("/session_status")
async def get_session_status():
    """Get current session status"""
    return {
        "camera_active": session.camera_active,
        "interview_active": session.interview_active,
        "recording_active": session.recording_active,
        "current_question": session.current_question,
        "total_questions": session.total_questions
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    session.cleanup()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("="*70)
    print("Starting Mock Interview API")
    print("="*70)
    print("Socket.IO: http://localhost:8000/socket.io")
    print("Health:    http://localhost:8000/")
    print("="*70)
    
    uvicorn.run(
        "api:socket_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )