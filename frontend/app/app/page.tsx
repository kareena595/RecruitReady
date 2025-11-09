//page.tsx

"use client";

import { useEffect, useState, useRef } from "react";
import { io, Socket } from "socket.io-client";
import Node from "../components/Node";

interface VideoFrame {
  frame: string;
  metrics: any;
  recording: boolean;
}

interface InterviewFeedback {
  question_number: number;
  camera_feedback?: {
    eye_contact_feedback?: string;
    posture_feedback?: string;
    movement_feedback?: string;
    overall_camera_summary?: string;
    error?: string;
  };
  speech_feedback?: {
    pace_feedback?: string;
    clarity_and_volume_feedback?: string;
    content_feedback?: string;
    overall_speech_summary?: string;
    error?: string;
  };
  next_question?: string;
  metrics_summary?: any;
  session_summary?: {
    performance_level: string;
    overall_score: number;
    engagement_score: number;
    clarity_score: number;
    posture_score: number;
    summary: string;
    questions_completed: number;
    improvement_areas: string[];
    strengths: string[];
    encouragement?: string;
  };
  interview_complete: boolean;
}

export default function HomePage() {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [connected, setConnected] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [interviewActive, setInterviewActive] = useState(false);
  const [recording, setRecording] = useState(false);
  const [interviewComplete, setInterviewComplete] = useState(false);
  
  const [videoFrame, setVideoFrame] = useState<string | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState<string>("");
  const [questionNumber, setQuestionNumber] = useState<number>(0);
  const [totalQuestions, setTotalQuestions] = useState<number>(2);
  const [transcript, setTranscript] = useState<string>("");
  const [wordCount, setWordCount] = useState<number>(0);
  
  const [postureFeedback, setPostureFeedback] = useState<string>("");
  const [speechFeedback, setSpeechFeedback] = useState<string>("");
  const [overallFeedback, setOverallFeedback] = useState<string>("");
  const [lastTranscript, setLastTranscript] = useState<string>("");

  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    // Initialize Socket.IO connection
    const newSocket = io("http://localhost:8000", {
      transports: ["websocket", "polling"],
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    socketRef.current = newSocket;
    setSocket(newSocket);

    // Connection handlers
    newSocket.on("connect", () => {
      console.log("✓ Connected to backend!");
      setConnected(true);
    });

    newSocket.on("disconnect", () => {
      console.log("✗ Disconnected from backend");
      setConnected(false);
      setCameraActive(false);
      setInterviewActive(false);
      setRecording(false);
    });

    newSocket.on("connection_status", (data) => {
      console.log("Connection status:", data);
    });

    // Camera handlers
    newSocket.on("camera_status", (data) => {
      console.log("Camera status:", data);
      if (data.status === "started") {
        setCameraActive(true);
      } else if (data.status === "stopped") {
        setCameraActive(false);
      }
    });

    newSocket.on("video_frame", (data: VideoFrame) => {
      setVideoFrame(data.frame);
      setRecording(data.recording || false);
    });

    // Interview handlers
    newSocket.on("interview_started", (data) => {
      console.log("Interview started:", data);
      setInterviewActive(true);
      setCurrentQuestion(data.question);
      setQuestionNumber(data.question_number);
      setTotalQuestions(data.total_questions);
      setInterviewComplete(false);
      setPostureFeedback("");
      setSpeechFeedback("");
      setOverallFeedback("");
    });

    // Recording handlers
    newSocket.on("recording_status", (data) => {
      console.log("Recording status:", data);
      if (data.status === "started") {
        setRecording(true);
        setTranscript("");
        setWordCount(0);
      } else if (data.status === "stopped") {
        setRecording(false);
      }
    });

    newSocket.on("transcript_update", (data) => {
      setTranscript(data.transcript);
      setWordCount(data.word_count || 0);
    });

    newSocket.on("speech_status", (data) => {
      console.log("Speech status:", data);
    });

    newSocket.on("speech_complete", (data) => {
      console.log("Speech complete:", data);
      setTranscript(data.transcript);
      setLastTranscript(data.transcript);
    });

    // Feedback handlers
    newSocket.on("interview_feedback", (data: InterviewFeedback) => {
      console.log("Interview feedback received:", data);
      console.log("Camera feedback:", data.camera_feedback);
      console.log("Speech feedback:", data.speech_feedback);
      
      // Format and display camera/posture feedback
      if (data.camera_feedback) {
        const cameraText = formatCameraFeedback(data.camera_feedback);
        setPostureFeedback(cameraText);
        console.log("Camera feedback set:", cameraText);
      }
      
      // Format and display speech feedback
      if (data.speech_feedback) {
        const speechText = formatSpeechFeedback(data.speech_feedback);
        setSpeechFeedback(speechText);
        console.log("Speech feedback set:", speechText);
      }
      
      if (data.interview_complete && data.session_summary) {
        // Interview is complete - show overall feedback
        const overallText = formatOverallFeedback(data.session_summary);
        setOverallFeedback(overallText);
        setInterviewComplete(true);
        setInterviewActive(false);
      } else if (data.next_question) {
        // Move to next question
        setCurrentQuestion(data.next_question);
        setQuestionNumber(prev => prev + 1);
        setTranscript("");
        setLastTranscript("");
        
        // Clear feedback for next question
        setPostureFeedback("");
        setSpeechFeedback("");
      }
    });

    // Error handler
    newSocket.on("error", (data) => {
      console.error("Error from backend:", data);
      alert(`Error: ${data.message}`);
    });

    // Cleanup
    return () => {
      newSocket.close();
    };
  }, []);

  const formatCameraFeedback = (feedback: any): string => {
    if (!feedback) {
      return "No posture feedback available";
    }
    
    if (feedback.error) {
      return `Error: ${feedback.error}`;
    }
    
    const parts = [];
    
    if (feedback.eye_contact_feedback) {
      parts.push(`👁️ Eye Contact:\n${feedback.eye_contact_feedback}`);
    }
    
    if (feedback.posture_feedback) {
      parts.push(`🧍 Posture:\n${feedback.posture_feedback}`);
    }
    
    if (feedback.movement_feedback) {
      parts.push(`🤚 Movement:\n${feedback.movement_feedback}`);
    }
    
    if (feedback.overall_camera_summary) {
      parts.push(`📸 Overall:\n${feedback.overall_camera_summary}`);
    }
    
    if (parts.length === 0) {
      return "Processing posture feedback...";
    }
    
    return parts.join("\n\n");
  };

  const formatSpeechFeedback = (feedback: any): string => {
    if (!feedback) {
      return "No speech feedback available";
    }
    
    if (feedback.error) {
      return `Error: ${feedback.error}`;
    }
    
    const parts = [];
    
    if (feedback.pace_feedback) {
      parts.push(`⏱️ Pace:\n${feedback.pace_feedback}`);
    }
    
    if (feedback.clarity_and_volume_feedback) {
      parts.push(`🔊 Clarity & Volume:\n${feedback.clarity_and_volume_feedback}`);
    }
    
    if (feedback.content_feedback) {
      parts.push(`💬 Content:\n${feedback.content_feedback}`);
    }
    
    if (feedback.overall_speech_summary) {
      parts.push(`🗣️ Overall:\n${feedback.overall_speech_summary}`);
    }
    
    if (parts.length === 0) {
      return "Processing speech feedback...";
    }
    
    return parts.join("\n\n");
  };

  const formatOverallFeedback = (summary: any): string => {
    if (!summary) return "No overall feedback available";
    
    if (summary.error) {
      return `Error: ${summary.error}`;
    }
    
    let text = `🎯 Performance Level: ${summary.performance_level}\n`;
    text += `📊 Overall Score: ${summary.overall_score}/100\n\n`;
    text += `${summary.summary}\n\n`;
    
    if (summary.strengths && summary.strengths.length > 0) {
      text += `💪 Strengths:\n`;
      summary.strengths.forEach((s: string) => {
        text += `  • ${s}\n`;
      });
    }
    
    if (summary.improvement_areas && summary.improvement_areas.length > 0) {
      text += `\n📈 Areas for Improvement:\n`;
      summary.improvement_areas.forEach((a: string) => {
        text += `  • ${a}\n`;
      });
    }
    
    if (summary.encouragement) {
      text += `\n💙 ${summary.encouragement}`;
    }
    
    return text;
  };

  const startCamera = () => {
    if (socket && connected) {
      console.log("Starting camera...");
      socket.emit("start_camera", {});
    }
  };

  const stopCamera = () => {
    if (socket && connected) {
      console.log("Stopping camera...");
      socket.emit("stop_camera", {});
      setInterviewActive(false);
      setRecording(false);
      setInterviewComplete(false);
      setPostureFeedback("");
      setSpeechFeedback("");
      setOverallFeedback("");
      setCurrentQuestion("");
    }
  };

  const startInterview = () => {
    if (socket && connected && cameraActive) {
      console.log("Starting interview...");
      socket.emit("start_interview", {});
      setPostureFeedback("");
      setSpeechFeedback("");
      setOverallFeedback("");
      setQuestionNumber(0);
      setInterviewComplete(false);
    }
  };

  const startRecording = () => {
    if (socket && connected && interviewActive && !recording) {
      console.log("Starting recording...");
      socket.emit("start_recording", {});
    }
  };

  const stopRecording = () => {
    if (socket && connected && recording) {
      console.log("Stopping recording...");
      socket.emit("stop_recording", {});
    }
  };

  return (
    <div className="flex flex-col w-[90vw] h-[90vh] bg-[#1e1e1e] rounded-[10px] shadow-[0_0_px_rgba(0,0,0,0.2)] p-5 bg-gradient-to-br from-gray-900 via-gray-800 to-slate-900">
      {/* Connection Status Bar */}
      <div className="mb-4 p-3 rounded-lg bg-gray-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`} />
          <span className="text-sm text-gray-300">
            {connected ? "Connected to Backend" : "Disconnected"}
          </span>
          {interviewActive && !interviewComplete && (
            <span className="text-sm text-blue-400 ml-4">
              Question {questionNumber} of {totalQuestions}
            </span>
          )}
          {recording && (
            <span className="text-sm text-red-400 ml-4 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
              Recording...
            </span>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={startCamera}
            disabled={!connected || cameraActive}
            className="px-4 py-2 text-sm bg-emerald-500 hover:bg-emerald-600 disabled:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition"
          >
            {cameraActive ? "Camera Active" : "Start Camera"}
          </button>
          <button
            onClick={stopCamera}
            disabled={!connected || !cameraActive}
            className="px-4 py-2 text-sm bg-red-500 hover:bg-red-600 disabled:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition"
          >
            Stop Camera
          </button>
          <button
            onClick={startInterview}
            disabled={!connected || !cameraActive || interviewActive}
            className="px-4 py-2 text-sm bg-blue-500 hover:bg-blue-600 disabled:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition"
          >
            {interviewActive ? "Interview Active" : "Start Interview"}
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex flex-row flex-1 gap-5">
        {/* Video Panel */}
        <div className="flex-[3] flex flex-col justify-center items-center relative">
          <h1 className="font-light m-0 mb-5 tracking-[2px]">
            Live Video Stream
          </h1>

          {videoFrame ? (
            <div className="relative max-h-[70%] w-full">
              <img
                src={videoFrame}
                alt="Live Stream"
                className="w-full rounded-lg object-cover transition-transform duration-300"
              />
              {recording && (
                <div className="absolute top-4 right-4 bg-red-500 text-white px-3 py-1 rounded-full text-sm flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-white animate-pulse"></span>
                  REC
                </div>
              )}
            </div>
          ) : (
            <div className="max-h-[70%] w-full rounded-lg bg-gray-800 flex items-center justify-center text-gray-500">
              {cameraActive ? "Waiting for video..." : "Camera not active"}
            </div>
          )}

          {/* Current Question Display */}
          {currentQuestion && !interviewComplete && (
            <div className="mt-6 w-full p-4 bg-blue-900/30 rounded-lg border border-blue-500/30">
              <h3 className="text-lg font-semibold text-blue-300 mb-2">Current Question:</h3>
              <p className="text-white">{currentQuestion}</p>
            </div>
          )}

          {/* Recording Controls */}
          {interviewActive && !interviewComplete && (
            <div className="mt-4 flex gap-3">
              <button
                onClick={startRecording}
                disabled={recording}
                className="px-6 py-3 text-lg bg-red-500 hover:bg-red-600 disabled:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition font-semibold"
              >
                {recording ? "Recording..." : "🎤 Record Answer"}
              </button>
              <button
                onClick={stopRecording}
                disabled={!recording}
                className="px-6 py-3 text-lg bg-gray-600 hover:bg-gray-700 disabled:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition"
              >
                ⏹️ Stop
              </button>
            </div>
          )}

          {/* Live Transcript */}
          {transcript && recording && (
            <div className="mt-4 w-full p-4 bg-gray-800 rounded-lg border border-gray-600">
              <h3 className="text-sm font-semibold text-gray-400 mb-2">
                Your Answer ({wordCount} words):
              </h3>
              <p className="text-white text-sm">{transcript}</p>
            </div>
          )}
        </div>

        {/* Feedback Panels */}
        <div className="flex-1 flex justify-center items-center border-l border-[#444] pl-5">
          <div className="w-full h-full flex flex-col gap-3 overflow-y-auto pr-1.5 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-[#1e1e1e] [&::-webkit-scrollbar-track]:rounded [&::-webkit-scrollbar-thumb]:bg-[#555] [&::-webkit-scrollbar-thumb]:rounded [&::-webkit-scrollbar-thumb:hover]:bg-[#666]">
            
            {/* Posture Feedback Panel */}
            <Node
              title="Posture Feedback"
              content={
                postureFeedback
                  ? postureFeedback
                  : interviewActive && !interviewComplete
                  ? "Answer the question to receive feedback on your posture, eye contact, and body language.\n\n(Feedback will appear here after you finish recording)"
                  : interviewComplete
                  ? "See Overall Feedback panel below for complete session analysis."
                  : "Posture feedback will appear here after each answer."
              }
            />

            {/* Speech Feedback Panel */}
            <Node
              title="Speech Feedback"
              content={
                speechFeedback
                  ? speechFeedback
                  : interviewActive && !interviewComplete
                  ? "Answer the question to receive feedback on your speaking pace, clarity, volume, and content.\n\n(Feedback will appear here after you finish recording)"
                  : interviewComplete
                  ? "See Overall Feedback panel below for complete session analysis."
                  : "Speech feedback will appear here after each answer."
              }
            />

            {/* Overall Feedback Panel */}
            <Node
              title="Overall Feedback"
              content={
                overallFeedback
                  ? overallFeedback
                  : interviewComplete
                  ? "Interview complete! Overall feedback displayed above."
                  : "Complete the interview to receive your overall performance summary."
              }
            />
          </div>
        </div>
      </div>
    </div>
  );
}