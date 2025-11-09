# app.py
import asyncio
import time
from test_agent import root_agent  # Import your agent definition
from google.adk.sessions import LiveSession

# Example: function to simulate streaming camera metrics
async def stream_camera_metrics():
    for i in range(10):
        metrics = {
            "shoulder_angle": 180 - i,
            "head_tilt": 180 + i,
            "forward_lean": 0.1 * i,
            "timestamp": time.time()
        }
        yield metrics
        await asyncio.sleep(0.5)

async def main():
    # Start a live session with the agent
    async with LiveSession(agent=root_agent) as session:
        print("Live session started. Streaming metrics...")

        # Simulate sending metrics to the agent
        async for metrics in stream_camera_metrics():
            await session.send_observation(metrics)  # Send live metrics
            # Optional: receive agent feedback
            response = await session.receive_response()
            if response:
                print("Agent feedback:", response)

        print("Live session finished.")

if __name__ == "__main__":
    asyncio.run(main())
