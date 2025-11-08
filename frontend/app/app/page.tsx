"use client";

import { useEffect, useState } from "react";
import Node from "../components/Node";

interface LiveData {
  message: string;
}

export default function HomePage() {
  const [data, setData] = useState<LiveData | null>(null);
  const videoUrl = "http://localhost:5000/video_feed"; // Change to your Flask URL

  useEffect(() => {
    const eventSource = new EventSource("http://localhost:5000/live-data");

    eventSource.onmessage = (event) => {
      setData(JSON.parse(event.data));
    };

    return () => eventSource.close();
  }, []);

  const zoomIn = () => {
    document.getElementById("video-stream")!.style.transform = "scale(1.2)";
  };

  const zoomOut = () => {
    document.getElementById("video-stream")!.style.transform = "scale(1)";
  };

  return (
    <div className="flex flex-row w-[90vw] h-[90vh] bg-[#1e1e1e] rounded-[10px] shadow-[0_0_25px_rgba(0,0,0,0.8)] p-5">
      <div className="flex-[3] flex flex-col justify-center items-center relative pr-5">
        <h1 className="font-light m-0 mb-5 tracking-[2px]">Live Video Stream</h1>

        <img
          id="video-stream"
          src={videoUrl}
          alt="Live Stream"
          className="max-h-[85%] rounded-lg object-cover transition-transform duration-300"
        />

        <div className="absolute bottom-5 left-1/2 -translate-x-1/2">
          <button
            onClick={zoomIn}
            className="w-[50px] h-[50px] text-2xl font-bold text-white bg-[#333] border border-[#555] rounded-full mx-2.5 cursor-pointer transition-all duration-200 hover:bg-[#555] active:scale-95 disabled:bg-[#2a2a2a] disabled:border-[#444] disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:bg-[#2a2a2a]"
          >
            +
          </button>
          <button
            onClick={zoomOut}
            className="w-[50px] h-[50px] text-2xl font-bold text-white bg-[#333] border border-[#555] rounded-full mx-2.5 cursor-pointer transition-all duration-200 hover:bg-[#555] active:scale-95 disabled:bg-[#2a2a2a] disabled:border-[#444] disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:bg-[#2a2a2a]"
          >
            -
          </button>
        </div>
      </div>

      <div className="flex-1 flex justify-center items-center border-l border-[#444] pl-5">
        <div className="w-full h-full flex flex-col gap-3 overflow-y-auto pr-1.5 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-[#1e1e1e] [&::-webkit-scrollbar-track]:rounded [&::-webkit-scrollbar-thumb]:bg-[#555] [&::-webkit-scrollbar-thumb]:rounded [&::-webkit-scrollbar-thumb:hover]:bg-[#666]">
          <Node
            title="Live Data"
            content={data ? data.message : "Waiting for server data..."}
          />
          <Node
            title="Speech Analysis"
            content="Speech pacing, clarity, and volume metrics will appear here."
          />
          <Node
            title="Visual Metrics"
            content="Eye contact, posture, and gesture analysis will appear here."
          />
          <Node
            title="Engagement Score"
            content="Overall engagement and performance metrics will appear here."
          />
        </div>
      </div>
    </div>
  );
}