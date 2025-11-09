"""
tempApp.py - Camera data collection with agent feedback
Direct extension of dummy.py - collects data, then sends to agent
"""

import asyncio
import json
from camera import stream_camera_metrics
from interview_agent.agent import (
    APP_NAME,
    USER_ID,
    SESSION_ID,
    interview_runner,
    interview_agent,
    session_service,
    types
)


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
    
    # Count issues
    all_issues = []
    for d in all_metrics:
        all_issues.extend(d.get('issues', []))
    
    issue_counts = {}
    for issue in all_issues:
        issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    avg_metrics['issue_summary'] = issue_counts
    avg_metrics['total_frames'] = total_frames
    
    return avg_metrics


def format_camera_metrics(camera_data):
    """Format camera metrics into readable string for agent"""
    return f"""Camera Metrics (Averaged over {camera_data.get('total_frames', 0)} frames):
- Eye Contact: {camera_data.get('eye_contact_percentage', 0):.1f}% maintained
- Left Iris Position: {camera_data.get('left_iris_relative', 0):.3f} (valid: 2.75-2.85)
- Right Iris Position: {camera_data.get('right_iris_relative', 0):.3f} (valid: -1.95 to -1.875)
- Shoulder Angle: {camera_data.get('shoulder_angle', 180):.1f}° (acceptable: 165-195°)
- Head Tilt: {camera_data.get('head_tilt', 180):.1f}° (acceptable: 165-195°)
- Forward Lean: {camera_data.get('forward_lean', 0.0):.3f} (threshold: 0.15)
- Head Motion: {camera_data.get('head_motion', 0):.1f} px/frame (threshold: 15.0)
- Hand Motion: {camera_data.get('hand_motion', 0):.1f} px/frame (threshold: 20.0)
- Issues Detected: {camera_data.get('issue_summary', {})}"""


async def send_to_agent(camera_metrics):
    """Send metrics to agent and get feedback"""
    # Format message for agent
    message = f"""Interview Response Analysis (Camera Data Only):

{format_camera_metrics(camera_metrics)}

Please analyze these camera metrics and provide feedback on posture, eye contact, and body language."""
    
    print(f"\n{'='*70}")
    print(">>> Sending to Agent...")
    print(f"{'='*70}")
    
    # Create user content
    user_content = types.Content(
        role='user',
        parts=[types.Part(text=message)]
    )
    
    # Get agent response
    final_response = "No response received."
    async for event in interview_runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
    
    print(f"\n<<< Agent Feedback:")
    print(final_response)
    print(f"{'='*70}\n")


async def main():
    """Main function - extends dummy.py with agent functionality"""
    
    # Setup agent session
    print("\n" + "="*70)
    print("🎯 CAMERA DATA COLLECTION + AGENT FEEDBACK")
    print("="*70)
    
    try:
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
        print(f"✅ Session created: {SESSION_ID}\n")
    except Exception as e:
        print(f"Session note: {e}\n")
    
    # Start interview
    print("🚀 Starting interview...")
    user_content = types.Content(
        role='user',
        parts=[types.Part(text="Hello! I'm ready to start my practice interview.")]
    )
    async for event in interview_runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print(f"\nAgent: {event.content.parts[0].text}\n")
    
    # EXACT SAME CODE AS DUMMY.PY - Collect camera data
    all_metrics = []  # List to store all frames' metrics
    
    print("="*70)
    print("Starting camera metrics collection...")
    print("Press 'q' in the video window to stop and send data to agent.")
    print("="*70 + "\n")
    
    try:
        # Stream metrics from camera.py
        for metrics in stream_camera_metrics(show_video=True):
            all_metrics.append(metrics)  # Accumulate each frame's metrics
    except KeyboardInterrupt:
        print("Interrupted by user.")
    
    # Data collection finished - now process and send to agent
    print("\n" + "="*70)
    print(f"✅ Collection complete! Total frames: {len(all_metrics)}")
    print("="*70)
    
    if all_metrics:
        # Show sample data
        print("\nSample data (first 3 frames):")
        print(json.dumps(all_metrics[:3], indent=2))
        
        # Aggregate metrics
        print("\nAggregating metrics...")
        camera_metrics = aggregate_camera_metrics(all_metrics)
        
        print("\n" + "="*70)
        print("📊 AGGREGATED METRICS")
        print("="*70)
        print(format_camera_metrics(camera_metrics))
        print("="*70)
        
        # Send to agent
        await send_to_agent(camera_metrics)
    else:
        print("\n⚠️  No data collected!")
    
    print("\n" + "="*70)
    print("✅ COMPLETE")
    print("="*70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()