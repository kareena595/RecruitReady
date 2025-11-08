#voice.py
import pyaudio
import numpy as np
import whisper
import wave
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Generator, Tuple
import json
import threading
import queue

@dataclass
class SpeechMetrics:
    """Data class to store speech analysis metrics - RAW values only"""
    text: str
    words_per_minute: float
    volume_db: float
    clarity_score: float
    speech_duration: float
    timestamp: float
    
    def to_json(self) -> str:
        """Convert metrics to JSON string"""
        return json.dumps(asdict(self))
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary"""
        return asdict(self)


class VoiceAnalyzer:
    """Analyzes speech patterns using Whisper with text-based VAD"""
    
    def __init__(self, model_size="base"):
        """
        Initialize the voice analyzer
        model_size: 'tiny', 'base', 'small', 'medium', 'large'
        'base' is recommended for real-time performance
        """
        print("Loading Whisper model...")
        self.model = whisper.load_model(model_size)
        print("Model loaded!")
        
        # Audio configuration
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # Whisper expects 16kHz
        
        # Text-based VAD settings
        self.TRANSCRIBE_INTERVAL = 0.5  # Transcribe every 0.5 seconds for responsive detection
        self.NO_SPEECH_DURATION = 2.0  # If no new text for 2 seconds, consider speech ended
        
        # Audio interface
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # For tracking current speech segment
        self.full_audio_buffer = []  # Complete audio for final analysis
        self.recent_audio_buffer = []  # Recent audio for periodic transcription
        self.accumulated_text = ""
        self.last_text_time = None
        self.speech_start_time = None
        self.is_speaking = False
        
    def calculate_volume_db(self, audio_chunk: bytes) -> float:
        """Calculate volume in decibels for a single audio chunk"""
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_array**2))
        
        if rms > 0:
            db = 20 * np.log10(rms / 32768.0)
        else:
            db = -100
        
        return db
    
    def calculate_average_volume_db(self, audio_data: bytes) -> float:
        """Calculate average volume across entire audio segment"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_array**2))
        
        if rms > 0:
            db = 20 * np.log10(rms / 32768.0)
        else:
            db = -100
        
        return db
    
    def calculate_speech_rate(self, text: str, duration: float) -> float:
        """Calculate words per minute"""
        if duration == 0:
            return 0.0
        
        words = text.split()
        word_count = len(words)
        wpm = (word_count / duration) * 60
        return wpm
    
    def estimate_clarity(self, result) -> float:
        """Estimate clarity based on Whisper's confidence scores"""
        try:
            if hasattr(result, 'segments') and result.segments:
                avg_logprob = np.mean([seg['avg_logprob'] for seg in result.segments])
                clarity = np.exp(avg_logprob)
                return min(1.0, max(0.0, clarity))
            else:
                return 0.8 if result.get('text', '').strip() else 0.0
        except:
            return 0.8
    
    def transcribe_audio(self, audio_data: bytes) -> Optional[dict]:
        """Transcribe audio data using Whisper"""
        temp_filename = f"temp_audio_{int(time.time() * 1000)}.wav"
        
        try:
            # Save to temporary WAV file
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(audio_data)
            
            # Transcribe
            result = self.model.transcribe(
                temp_filename,
                language="en",
                fp16=False,
                verbose=False
            )
            
            os.remove(temp_filename)
            return result
            
        except Exception as e:
            print(f"Transcription error: {e}")
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return None
    
    def analyze_speech_segment(self, audio_data: bytes, duration: float, text: str) -> Optional[SpeechMetrics]:
        """Analyze a complete speech segment - returns RAW metrics only"""
        if not text or len(text.strip()) < 3:
            return None
        
        # Calculate RAW metrics only - no judgments
        volume_db = self.calculate_average_volume_db(audio_data)
        wpm = self.calculate_speech_rate(text, duration)
        
        # For clarity, re-transcribe the full segment to get confidence
        result = self.transcribe_audio(audio_data)
        clarity = self.estimate_clarity(result) if result else 0.8
        
        return SpeechMetrics(
            text=text.strip(),
            words_per_minute=wpm,
            volume_db=volume_db,
            clarity_score=clarity,
            speech_duration=duration,
            timestamp=time.time()
        )
    
    def start_stream(self):
        """Start audio stream"""
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
    
    def stop_stream(self):
        """Stop audio stream and clean up"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
    
    def read_audio_chunk(self) -> Tuple[bytes, float]:
        """Read a single audio chunk and return data + volume"""
        try:
            audio_chunk = self.stream.read(self.CHUNK, exception_on_overflow=False)
            volume_db = self.calculate_volume_db(audio_chunk)
            return audio_chunk, volume_db
        except Exception as e:
            print(f"Error reading audio: {e}")
            return b'', -100


# ============================================================================
# STREAMING INTERFACE FOR APP.PY WITH TEXT-BASED VAD
# ============================================================================

def stream_voice_with_text_vad(
    no_speech_duration: float = 2.0,
    transcribe_interval: float = 0.5,
    model_size: str = "base"
) -> Generator[dict, None, None]:
    """
    Generator that yields real-time updates and complete speech segments when detected.
    Uses TEXT-BASED Voice Activity Detection - stops when no new text for specified duration.
    
    This is much more reliable than volume-based detection in noisy environments!
    
    Args:
        no_speech_duration: Seconds of no new text before considering speech ended (default: 2.0)
        transcribe_interval: How often to transcribe recent audio (default: 0.5s)
        model_size: Whisper model size
        
    Yields:
        Dictionary containing:
        - Real-time status: {"type": "status", "text": "current text...", "is_speaking": True}
        - Speech started: {"type": "speech_started", "timestamp": ...}
        - Complete speech analysis: {"type": "speech_complete", "metrics": {...}}
        
        Metrics structure (RAW values only):
        {
            "text": "I have three years of experience...",
            "words_per_minute": 145.0,
            "volume_db": -52.0,
            "clarity_score": 0.85,
            "speech_duration": 8.5,
            "timestamp": 1699999999.123
        }
    
    Usage in app.py:
        for data in stream_voice_with_text_vad():
            if data["type"] == "status":
                print(f"Current text: {data['text']}")
                
            elif data["type"] == "speech_complete":
                # User finished speaking - send to agent!
                metrics = data["metrics"]
                send_to_agent(metrics)
    """
    analyzer = VoiceAnalyzer(model_size=model_size)
    analyzer.NO_SPEECH_DURATION = no_speech_duration
    analyzer.TRANSCRIBE_INTERVAL = transcribe_interval
    
    analyzer.start_stream()
    
    print("Voice analyzer streaming started (text-based VAD)...")
    print(f"No speech duration: {no_speech_duration} seconds")
    print(f"Transcribe interval: {transcribe_interval} seconds")
    
    frames_per_interval = int(analyzer.RATE / analyzer.CHUNK * transcribe_interval)
    frame_count = 0
    
    try:
        while True:
            audio_chunk, volume_db = analyzer.read_audio_chunk()
            current_time = time.time()
            
            # Add to buffers
            analyzer.full_audio_buffer.append(audio_chunk)
            analyzer.recent_audio_buffer.append(audio_chunk)
            frame_count += 1
            
            # Periodic transcription to detect new speech
            if frame_count >= frames_per_interval:
                frame_count = 0
                
                # Transcribe recent audio
                recent_audio_data = b''.join(analyzer.recent_audio_buffer)
                result = analyzer.transcribe_audio(recent_audio_data)
                
                if result:
                    new_text = result.get('text', '').strip()
                    
                    # Check if there's new meaningful text
                    if new_text and len(new_text) > 3:
                        # New speech detected!
                        if not analyzer.is_speaking:
                            # Speech just started
                            analyzer.is_speaking = True
                            analyzer.speech_start_time = current_time
                            analyzer.accumulated_text = new_text
                            analyzer.last_text_time = current_time
                            
                            yield {
                                "type": "speech_started",
                                "timestamp": current_time
                            }
                        else:
                            # Ongoing speech - check if text is actually new
                            # Compare with accumulated text to see if there's new content
                            if new_text not in analyzer.accumulated_text:
                                analyzer.accumulated_text += " " + new_text
                                analyzer.last_text_time = current_time
                        
                        # Yield status update
                        yield {
                            "type": "status",
                            "text": analyzer.accumulated_text,
                            "is_speaking": True,
                            "timestamp": current_time
                        }
                    
                    # Check for speech end (no new text)
                    elif analyzer.is_speaking and analyzer.last_text_time:
                        silence_duration = current_time - analyzer.last_text_time
                        
                        if silence_duration >= analyzer.NO_SPEECH_DURATION:
                            # Speech has ended!
                            speech_duration = current_time - analyzer.speech_start_time
                            full_audio_data = b''.join(analyzer.full_audio_buffer)
                            
                            # Analyze the complete segment
                            metrics = analyzer.analyze_speech_segment(
                                full_audio_data,
                                speech_duration,
                                analyzer.accumulated_text
                            )
                            
                            if metrics:
                                yield {
                                    "type": "speech_complete",
                                    "metrics": metrics.to_dict()
                                }
                            
                            # Reset for next segment
                            analyzer.is_speaking = False
                            analyzer.full_audio_buffer = []
                            analyzer.accumulated_text = ""
                            analyzer.last_text_time = None
                            analyzer.speech_start_time = None
                
                # Clear recent buffer for next interval
                analyzer.recent_audio_buffer = []
            
            # Yield periodic status even when not transcribing
            if analyzer.is_speaking and int(current_time * 2) % 2 == 0:  # Every 0.5s
                yield {
                    "type": "status",
                    "text": analyzer.accumulated_text,
                    "is_speaking": True,
                    "timestamp": current_time,
                    "volume_db": volume_db
                }
    
    finally:
        analyzer.stop_stream()
        print("Voice analyzer stopped.")


def listen_for_single_response(
    no_speech_duration: float = 2.0,
    model_size: str = "base",
    max_duration: float = 60.0
) -> Optional[dict]:
    """
    Listen for a single complete speech response and return RAW metrics.
    Uses text-based VAD - much more reliable in noisy environments!
    
    Args:
        no_speech_duration: Seconds of no new text before considering response complete
        model_size: Whisper model size
        max_duration: Maximum recording duration in seconds
        
    Returns:
        Dictionary with RAW speech metrics, or None if no speech detected
    """
    start_time = time.time()
    
    for data in stream_voice_with_text_vad(no_speech_duration=no_speech_duration, model_size=model_size):
        # Timeout check
        if time.time() - start_time > max_duration:
            print("Max duration exceeded")
            return None
        
        # Return when speech is complete
        if data["type"] == "speech_complete":
            return data["metrics"]
    
    return None


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_text_vad_streaming():
    """Test the text-based VAD streaming"""
    print("\n" + "="*70)
    print("Testing Text-Based VAD Voice Streaming (RAW metrics)")
    print("="*70)
    print("Speak naturally. System detects when you stop speaking (no new text).")
    print("Much more reliable in noisy environments!")
    print("Press Ctrl+C to stop\n")
    
    try:
        for data in stream_voice_with_text_vad():
            if data["type"] == "speech_started":
                print("\n🎤 Speech detected! Recording...")
            
            elif data["type"] == "status":
                # Show real-time transcription
                text = data["text"]
                display_text = text if len(text) <= 60 else text[:57] + "..."
                speaking_indicator = "🔴" if data["is_speaking"] else "⚪"
                print(f"\r{speaking_indicator} {display_text:<60}", end="", flush=True)
            
            elif data["type"] == "speech_complete":
                metrics = data["metrics"]
                print("\n\n" + "="*70)
                print("✅ SPEECH COMPLETE - RAW Metrics (Agent will analyze):")
                print("="*70)
                print(f"Transcript: {metrics['text']}")
                print(f"Duration: {metrics['speech_duration']:.1f}s")
                print(f"WPM: {metrics['words_per_minute']:.0f}")
                print(f"Volume: {metrics['volume_db']:.1f} dB")
                print(f"Clarity: {metrics['clarity_score']:.2f}")
                print("="*70 + "\n")
                print("Ready for next response...")
    
    except KeyboardInterrupt:
        print("\n\nStopped.")


def test_single_response():
    """Test listening for a single response"""
    print("\n" + "="*70)
    print("Testing Single Response Capture (Text-based VAD)")
    print("="*70)
    print("Speak your answer. Stop when you're done.\n")
    
    print("Listening...")
    metrics = listen_for_single_response(no_speech_duration=2.0)
    
    if metrics:
        print("\n" + "="*70)
        print("Response captured!")
        print("="*70)
        print(f"Transcript: {metrics['text']}")
        print(f"WPM: {metrics['words_per_minute']:.0f}")
        print(f"Volume: {metrics['volume_db']:.1f} dB")
        print(f"Duration: {metrics['speech_duration']:.1f}s")
        print("="*70)
    else:
        print("\nNo speech detected or timeout reached.")


def main():
    """Main test function"""
    print("\n" + "="*70)
    print("VOICE ANALYZER - Text-Based VAD (RAW Metrics)")
    print("="*70)
    print("\nChoose test mode:")
    print("1. Continuous streaming (recommended for interviews)")
    print("2. Single response capture")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_text_vad_streaming()
    elif choice == "2":
        test_single_response()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()