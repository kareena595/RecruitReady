# API Integration Guide

This guide explains how to use the FastAPI + Socket.IO backend with your Next.js frontend.

## Architecture

```
┌─────────────────┐         Socket.IO          ┌──────────────────┐
│                 │  ←─────────────────────→   │                  │
│  Next.js        │  WebSocket Connection      │  FastAPI         │
│  Frontend       │                            │  Backend         │
│  (localhost:3000)│                           │  (localhost:8000)│
│                 │                            │                  │
└─────────────────┘                            └──────────────────┘
                                                        │
                                                        ↓
                                                ┌──────────────────┐
                                                │  Camera/Voice    │
                                                │  Processing      │
                                                └──────────────────┘
```

## Getting Started

### 1. Start the Backend

```bash
# Navigate to project root
cd "/Users/jacob/Documents/Hackathon/AI ATL '25/video_meeting_analyzer"

# Activate virtual environment
source .venv/bin/activate

# Run the API server
cd frontend
python api.py
```

The backend will start on `http://localhost:8000`

Available endpoints:
- Socket.IO: `http://localhost:8000/socket.io`
- HTTP Video Feed: `http://localhost:8000/video_feed`
- Health Check: `http://localhost:8000/`
- Metrics API: `http://localhost:8000/metrics`

### 2. Start the Frontend

```bash
# In a new terminal
cd "/Users/jacob/Documents/Hackathon/AI ATL '25/video_meeting_analyzer/frontend"

# Run Next.js dev server
npm run dev
```

The frontend will start on `http://localhost:3000`

## Socket.IO Events

### Client → Server (Emit)

| Event | Data | Description |
|-------|------|-------------|
| `start_camera` | `{}` | Start camera and begin streaming |
| `stop_camera` | `{}` | Stop camera streaming |
| `process_audio` | `{audio: ArrayBuffer}` | Send audio data for processing |
| `user_message` | `{message: string, timestamp: number}` | Send user input to AI agent |

### Server → Client (Listen)

| Event | Data | Description |
|-------|------|-------------|
| `connection_status` | `{status: string, message: string}` | Connection established |
| `camera_status` | `{status: 'started'\|'stopped', message: string}` | Camera state changed |
| `video_frame` | `{frame: string, metrics: Metrics}` | Processed video frame + metrics |
| `ai_response` | `{message: string, timestamp: number}` | AI agent response |
| `error` | `{message: string}` | Error occurred |

## Metrics Interface

```typescript
interface Metrics {
  shoulder_angle: number;          // Degrees
  head_tilt: number;              // Degrees
  forward_lean: number;           // Ratio
  head_motion_score: number;      // Pixels per frame
  hand_motion_score: number;      // Pixels per frame
  eye_contact_maintained: boolean;
  eye_contact_duration: number;   // Seconds
  is_slouching: boolean;
  is_tilted: boolean;
  is_leaning: boolean;
  is_head_moving: boolean;
  is_hand_fidgeting: boolean;
  issues: string[];               // Array of detected issues
  timestamp: number;              // Unix timestamp
}
```

## Using the Example Component

I've created an example component at `app/app/page_socketio_example.tsx` that shows:

1. **Connection Management**
   - Connects automatically when page loads
   - Shows connection status
   - Handles reconnection

2. **Camera Control**
   - Start/Stop camera buttons
   - Receives processed video frames
   - Displays real-time metrics

3. **Bidirectional Communication**
   - Send messages to backend
   - Receive AI responses

### To use it:

**Option 1: Replace existing page**
```bash
cp app/app/page_socketio_example.tsx app/app/page.tsx
```

**Option 2: Use as reference**
Copy the Socket.IO logic into your existing `app/app/page.tsx`

## Example Usage

### Basic Connection

```typescript
import { io } from "socket.io-client";

// Connect to backend
const socket = io("http://localhost:8000");

// Listen for connection
socket.on("connect", () => {
  console.log("Connected!");
});

// Start camera
socket.emit("start_camera", {});

// Receive video frames
socket.on("video_frame", (data) => {
  const { frame, metrics } = data;
  // frame is base64 encoded image: data:image/jpeg;base64,...
  // Display in <img src={frame} />
});
```

### Send Audio Data

```typescript
// Capture audio from microphone
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const mediaRecorder = new MediaRecorder(stream);

mediaRecorder.ondataavailable = (event) => {
  const audioBlob = event.data;

  // Convert to ArrayBuffer
  audioBlob.arrayBuffer().then((buffer) => {
    socket.emit("process_audio", { audio: buffer });
  });
};

mediaRecorder.start();
```

### Communicate with AI Agent

```typescript
// Send message to AI
socket.emit("user_message", {
  message: "Tell me about my posture",
  timestamp: Date.now()
});

// Receive AI response
socket.on("ai_response", (data) => {
  console.log("AI says:", data.message);
});
```

## HTTP Endpoints (Alternative)

If you prefer HTTP over Socket.IO:

### Video Feed (MJPEG Stream)
```html
<img src="http://localhost:8000/video_feed" alt="Live Video" />
```

### Get Current Metrics
```typescript
const response = await fetch("http://localhost:8000/metrics");
const data = await response.json();
console.log(data.metrics);
```

## Troubleshooting

### Connection Issues

1. **Check backend is running**
   ```bash
   curl http://localhost:8000/
   ```

2. **Check CORS settings**
   The backend allows `localhost:3000` and `localhost:3001` by default.
   To add more origins, edit `api.py`:
   ```python
   allow_origins=["http://localhost:3000", "http://your-port"]
   ```

3. **Socket.IO not connecting**
   - Open browser console
   - Look for Socket.IO connection errors
   - Try switching transports: `transports: ["polling"]` or `["websocket"]`

### Camera Not Starting

1. **Check camera permissions** - Browser needs camera access
2. **Check if camera is in use** - Close other apps using camera
3. **Check backend logs** - Look for error messages

### No Video Frames

1. **Verify camera started** - Check `camera_status` event
2. **Check network tab** - Look for WebSocket messages
3. **Inspect console** - Look for JavaScript errors

## Next Steps

1. **Integrate voice processing**
   - Import `voice.py` in `api.py`
   - Add audio processing in `process_audio` event handler

2. **Connect AI agent**
   - Import agent from `interview_agent/agent.py`
   - Process user messages with agent
   - Stream agent responses back

3. **Add video upload**
   - Allow frontend to send video frames to backend
   - Process uploaded video instead of live camera

4. **Persist session data**
   - Store metrics in database
   - Track user progress over time

## File Structure

```
frontend/
├── api.py                          # FastAPI + Socket.IO backend
├── app/
│   ├── app/
│   │   ├── page.tsx               # Current app page
│   │   └── page_socketio_example.tsx  # Example Socket.IO integration
│   └── components/
│       └── Node.tsx               # Metric display component
└── package.json                   # Now includes socket.io-client
```

## Dependencies Installed

### Backend (Python)
- `fastapi` - Web framework
- `python-socketio` - Socket.IO server
- `uvicorn[standard]` - ASGI server
- `aiofiles` - Async file operations

### Frontend (npm)
- `socket.io-client` - Socket.IO client library

All dependencies are already installed! Just start the servers and you're ready to go.
