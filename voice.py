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
import webrtcvad

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
    """Analyzes speech patterns using Whisper with WebRTC VAD for silence detection"""
    
    def __init__(self, model_size="base"):
        """
        Initialize the voice analyzer
        model_size: 'tiny', 'base', 'small', 'medium', 'large'
        'base' is recommended for real-time performance
        """
        print("Loading Whisper model...")
        self.model = whisper.load_model(model_size)
        print("Model loaded!")
        
        # Audio configuration for WebRTC VAD (requires specific format)
        self.CHUNK = 480  # 30ms at 16kHz (WebRTC VAD requirement)
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # Whisper and WebRTC VAD both expect 16kHz
        
        # Initialize WebRTC VAD (mode 3 = most aggressive silence detection)
        self.vad = webrtcvad.Vad(3)
        
        # Transcription settings
        self.TRANSCRIBE_INTERVAL = 5.0  # 5 seconds for accurate transcription
        self.MIN_SPEECH_CHUNK_SIZE = 5.0  # Minimum 5 seconds before first transcription
        self.SILENCE_DURATION_THRESHOLD = 3.0  # 3 seconds of silence to end speech
        
        # Audio interface
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # For tracking current speech segment
        self.full_audio_buffer = []  # Complete audio for final analysis
        self.transcription_buffer = []  # Audio for next transcription
        self.accumulated_text = ""  # This stores ALL text across all chunks
        self.speech_start_time = None
        self.is_speaking = False
        self.first_transcription_done = False
        
        # WebRTC VAD silence tracking
        self.silence_frames = 0
        self.silence_threshold_frames = int((self.SILENCE_DURATION_THRESHOLD * self.RATE) / self.CHUNK)
        self.speech_frames = 0  # Consecutive speech frames
        self.speech_threshold_frames = 10  # Need 10 consecutive speech frames to start
        
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
    
    def transcribe_audio(self, audio_data: bytes, initial_prompt: str = None) -> Optional[dict]:
        """Transcribe audio data using Whisper with optional context"""
        temp_filename = f"temp_audio_{int(time.time() * 1000)}.wav"
        
        try:
            # Save to temporary WAV file
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(audio_data)
            
            # Transcribe with context from previous text
            transcribe_options = {
                "language": "en",
                "fp16": False,
                "verbose": False,
                "word_timestamps": True,
                "beam_size": 5,
                "best_of": 5,
                "temperature": 0.0
            }
            
            # Add initial prompt for context if we have previous text
            if initial_prompt:
                transcribe_options["initial_prompt"] = initial_prompt
            
            result = self.model.transcribe(temp_filename, **transcribe_options)
            
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
    
    def read_audio_chunk(self) -> Tuple[bytes, float, bool]:
        """Read a single audio chunk and return data + volume + is_speech"""
        try:
            audio_chunk = self.stream.read(self.CHUNK, exception_on_overflow=False)
            volume_db = self.calculate_volume_db(audio_chunk)
            
            # Use WebRTC VAD to detect speech
            is_speech = self.vad.is_speech(audio_chunk, self.RATE)
            
            return audio_chunk, volume_db, is_speech
        except Exception as e:
            print(f"Error reading audio: {e}")
            return b'', -100, False


# ============================================================================
# STREAMING INTERFACE WITH WEBRTC VAD
# ============================================================================

def stream_voice_with_text_vad(
    silence_duration: float = 3.0,
    transcribe_interval: float = 5.0,
    model_size: str = "base"
) -> Generator[dict, None, None]:
    """
    Generator that yields real-time updates and complete speech segments when detected.
    Uses WebRTC VAD for FAST and ACCURATE silence detection + large chunks for transcription.
    
    KEY FEATURES:
    - WebRTC VAD for real-time speech/silence detection (industry standard)
    - 5 second transcription intervals for accurate text
    - 3 second silence detection (much faster!)
    - Proper buffer clearing to avoid repeating text
    
    Args:
        silence_duration: Seconds of silence before considering speech ended (default: 3.0)
        transcribe_interval: How often to transcribe recent audio (default: 5.0s)
        model_size: Whisper model size
        
    Yields:
        Dictionary containing:
        - Real-time status: {"type": "status", "text": "current text...", "is_speaking": True}
        - Speech started: {"type": "speech_started", "timestamp": ...}
        - Complete speech analysis: {"type": "speech_complete", "metrics": {...}}
    """
    analyzer = VoiceAnalyzer(model_size=model_size)
    analyzer.SILENCE_DURATION_THRESHOLD = silence_duration
    analyzer.TRANSCRIBE_INTERVAL = transcribe_interval
    analyzer.silence_threshold_frames = int((silence_duration * analyzer.RATE) / analyzer.CHUNK)
    
    analyzer.start_stream()
    
    print("Voice analyzer streaming started (WebRTC VAD)...")
    print(f"Silence duration: {silence_duration} seconds")
    print(f"Transcribe interval: {transcribe_interval} seconds")
    print(f"Using WebRTC VAD for fast silence detection!")
    
    last_transcription_time = time.time()
    
    try:
        while True:
            audio_chunk, volume_db, is_speech = analyzer.read_audio_chunk()
            current_time = time.time()
            
            # Always add to full buffer for final analysis
            analyzer.full_audio_buffer.append(audio_chunk)
            
            # Track speech/silence using WebRTC VAD
            if is_speech:
                analyzer.speech_frames += 1
                analyzer.silence_frames = 0
                
                # Start speech after consistent speech detection
                if not analyzer.is_speaking and analyzer.speech_frames >= analyzer.speech_threshold_frames:
                    analyzer.is_speaking = True
                    analyzer.speech_start_time = current_time
                    print("\n🎤 Speech detected! Recording...")
                    
                    yield {
                        "type": "speech_started",
                        "timestamp": current_time
                    }
                
                # Add to transcription buffer while speaking
                if analyzer.is_speaking:
                    analyzer.transcription_buffer.append(audio_chunk)
            else:
                analyzer.speech_frames = 0
                
                if analyzer.is_speaking:
                    analyzer.silence_frames += 1
                    # Continue adding to transcription buffer during short silences
                    analyzer.transcription_buffer.append(audio_chunk)
                    
                    # Check if silence threshold exceeded
                    if analyzer.silence_frames >= analyzer.silence_threshold_frames and analyzer.first_transcription_done:
                        # Speech has ended!
                        speech_duration = current_time - analyzer.speech_start_time
                        full_audio_data = b''.join(analyzer.full_audio_buffer)
                        
                        print(f"\n✅ Silence detected ({silence_duration}s) - ending speech")
                        
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
                        analyzer.transcription_buffer = []
                        analyzer.accumulated_text = ""
                        analyzer.speech_start_time = None
                        analyzer.first_transcription_done = False
                        analyzer.silence_frames = 0
                        analyzer.speech_frames = 0
                        last_transcription_time = current_time
                        
                        continue
            
            # Periodic transcription while speaking
            if analyzer.is_speaking:
                time_since_last_transcription = current_time - last_transcription_time
                
                if time_since_last_transcription >= analyzer.TRANSCRIBE_INTERVAL:
                    # Check if we have enough audio
                    transcription_audio = b''.join(analyzer.transcription_buffer)
                    duration = len(analyzer.transcription_buffer) * analyzer.CHUNK / analyzer.RATE
                    
                    if duration >= analyzer.MIN_SPEECH_CHUNK_SIZE or analyzer.first_transcription_done:
                        print(f"\n[Transcribing {duration:.1f}s of audio...]")
                        
                        # Transcribe with context
                        initial_prompt = analyzer.accumulated_text if analyzer.accumulated_text else None
                        result = analyzer.transcribe_audio(transcription_audio, initial_prompt=initial_prompt)
                        
                        if result:
                            current_chunk_text = result.get('text', '').strip()
                            if current_chunk_text and len(current_chunk_text) > 3:
                                analyzer.first_transcription_done = True
                                # Only append new, non-repeated text
                                if not analyzer.accumulated_text:
                                    analyzer.accumulated_text = current_chunk_text
                                    print(f"[Initial text: '{current_chunk_text}']")
                                else:
                                    # Find the longest suffix of accumulated_text that matches the prefix of current_chunk_text
                                    overlap = 0
                                    max_overlap = min(len(analyzer.accumulated_text), len(current_chunk_text))
                                    for i in range(max_overlap, 0, -1):
                                        if analyzer.accumulated_text[-i:] == current_chunk_text[:i]:
                                            overlap = i
                                            break
                                    new_text = current_chunk_text[overlap:]
                                    if new_text:
                                        # Prevent repeated phrase addition
                                        last_sentences = analyzer.accumulated_text.split('. ')
                                        new_sentences = new_text.split('. ')
                                        # Only add sentences not already present at the end
                                        for sentence in new_sentences:
                                            sentence = sentence.strip()
                                            if sentence and (not analyzer.accumulated_text.endswith(sentence)):
                                                analyzer.accumulated_text += (" " if analyzer.accumulated_text else "") + sentence
                                                analyzer.accumulated_text = analyzer.accumulated_text.strip()
                                                print(f"[New text added: '{sentence}']")
                                            else:
                                                print(f"[Skipped repeated sentence: '{sentence}']")
                                    else:
                                        print(f"[Text unchanged - user may have paused]")
                                # Yield status update
                                yield {
                                    "type": "status",
                                    "text": analyzer.accumulated_text,
                                    "is_speaking": True,
                                    "timestamp": current_time,
                                    "word_count": len(analyzer.accumulated_text.split())
                                }
                        
                        # CRITICAL: Clear transcription buffer after transcribing!
                        analyzer.transcription_buffer = []
                        last_transcription_time = current_time
                    else:
                        print(f"[Collecting audio: {duration:.1f}s / {analyzer.MIN_SPEECH_CHUNK_SIZE:.1f}s]")
    
    finally:
        analyzer.stop_stream()
        print("Voice analyzer stopped.")


def listen_for_single_response(
    silence_duration: float = 3.0,
    model_size: str = "base",
    max_duration: float = 60.0
) -> Optional[dict]:
    """
    Listen for a single complete speech response and return RAW metrics.
    Uses WebRTC VAD for fast and accurate detection.
    
    Args:
        silence_duration: Seconds of silence before considering response complete
        model_size: Whisper model size
        max_duration: Maximum recording duration in seconds
        
    Returns:
        Dictionary with RAW speech metrics, or None if no speech detected
    """
    start_time = time.time()
    
    for data in stream_voice_with_text_vad(silence_duration=silence_duration, model_size=model_size):
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
    """Test the WebRTC VAD streaming"""
    print("\n" + "="*70)
    print("Testing WebRTC VAD Voice Streaming")
    print("="*70)
    print("Speak naturally - WebRTC VAD detects speech/silence!")
    print("5-second chunks for accuracy + 3-second silence detection!")
    print("Press Ctrl+C to stop\n")
    
    try:
        for data in stream_voice_with_text_vad():
            if data["type"] == "speech_started":
                print("\n🎤 Speech detected! Recording...")
            
            elif data["type"] == "status":
                # Show real-time transcription
                text = data["text"]
                word_count = data.get("word_count", 0)
                # Show more text in display
                display_text = text if len(text) <= 80 else "..." + text[-77:]
                speaking_indicator = "🔴"
                print(f"\r{speaking_indicator} [{word_count} words] {display_text:<80}", end="", flush=True)
            
            elif data["type"] == "speech_complete":
                metrics = data["metrics"]
                print("\n\n" + "="*70)
                print("✅ SPEECH COMPLETE - RAW Metrics:")
                print("="*70)
                print(f"Full Transcript: {metrics['text']}")
                print(f"Duration: {metrics['speech_duration']:.1f}s")
                print(f"WPM: {metrics['words_per_minute']:.0f}")
                print(f"Volume: {metrics['volume_db']:.1f} dB")
                print(f"Clarity: {metrics['clarity_score']:.2f}")
                print(f"Total Words: {len(metrics['text'].split())}")
                print("="*70 + "\n")
                print("Ready for next response...")
    
    except KeyboardInterrupt:
        print("\n\nStopped.")


def test_single_response():
    """Test listening for a single response"""
    print("\n" + "="*70)
    print("Testing Single Response Capture (WebRTC VAD)")
    print("="*70)
    print("Speak your answer. VAD detects when you stop!\n")
    
    print("Listening...")
    metrics = listen_for_single_response(silence_duration=3.0)
    
    if metrics:
        print("\n" + "="*70)
        print("Response captured!")
        print("="*70)
        print(f"Full Transcript: {metrics['text']}")
        print(f"Total Words: {len(metrics['text'].split())}")
        print(f"WPM: {metrics['words_per_minute']:.0f}")
        print(f"Volume: {metrics['volume_db']:.1f} dB")
        print(f"Duration: {metrics['speech_duration']:.1f}s")
        print("="*70)
    else:
        print("\nNo speech detected or timeout reached.")


def main():
    """Main test function"""
    print("\n" + "="*70)
    print("VOICE ANALYZER - WebRTC VAD (Industry Standard)")
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