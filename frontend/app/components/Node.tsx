"use client";

import { useState } from "react";

interface NodeProps {
  title: string;
  content: string;
  defaultExpanded?: boolean;
}

export default function Node({ title, content, defaultExpanded = false }: NodeProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className="bg-[#2a2a2a] rounded-lg overflow-hidden transition-all duration-300 hover:bg-[#333]">
      <div
        className="flex justify-between items-center px-5 py-[15px] cursor-pointer select-none"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h3 className="m-0 text-lg font-medium tracking-[0.5px]">{title}</h3>
        <span className="text-2xl font-light text-[#888] transition-transform duration-300">{isExpanded ? "−" : "+"}</span>
      </div>

      {isExpanded && (
        <div className="px-5 pb-[15px] animate-[slideDown_0.3s_ease]">
          <p className="m-0 font-['Courier_New',Courier,monospace] bg-[#1e1e1e] p-[15px] rounded-[5px] text-[0.95em] leading-[1.6]">{content}</p>
        </div>
      )}
    </div>
  );
}
