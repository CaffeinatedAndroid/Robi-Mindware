#!/usr/bin/env python3
"""
Local AI Voice Assistant with Piper TTS
Modes: constant_listen (always on) or sleep/wake (wake phrase required)
"""

import os
import re
import sys
import time
import wave
import threading
import tempfile
import subprocess
import argparse
from dataclasses import dataclass
from typing import List, Optional

import pyaudio
import webrtcvad
import whisper
from piper.voice import PiperVoice
from llama_cpp import Llama


@dataclass
class Personality:
    """Configure your assistant's behavior"""
    name: str = "Atlas"
    role: str = "helpful AI assistant"
    tone: str = "a bit blunt"
    max_tokens: int = 150
    temperature: float = 0.7
    system_prompt: Optional[str] = None

    def get_system_prompt(self) -> str:
        if self.system_prompt:
            return self.system_prompt
        return (
            f"You are {self.name}, a {self.tone} {self.role}. "
            f"Keep responses brief (1-2 sentences) for voice conversation. "
            f"Be warm, helpful, and natural."
        )


class VoiceAssistant:
    STATE_SLEEPING = "sleeping"
    STATE_ACTIVE = "active"

    SAMPLE_RATE = 16000
    CHUNK_DURATION_MS = 30
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
    VAD_AGGRESSIVENESS = 2

    def __init__(
        self,
        personality: Personality,
        model_path: str,
        piper_model_path: str,
        piper_config_path: str,
        whisper_model: str = "base",
        n_threads: int = 4,
        wake_phrases: Optional[List[str]] = None,
        sleep_timeout: int = 60,
        constant_listen: bool = False,
    ):
        self.personality = personality
        self.n_threads = n_threads
        self.sleep_timeout = sleep_timeout
        self.constant_listen = constant_listen

        self.wake_phrases = [p.lower() for p in (wake_phrases or ["hey atlas", "atlas", "okay atlas"])]
        self.state = self.STATE_ACTIVE if constant_listen else self.STATE_SLEEPING
        self.speaking = False
        self.last_active_time = time.time() if constant_listen else 0

        print(f"[INIT] Mode: {'CONSTANT LISTEN' if constant_listen else 'SLEEP/WAKE'}")
        print(f"[INIT] Loading Whisper ({whisper_model})...")
        self.whisper = whisper.load_model(whisper_model)

        print(f"[INIT] Loading LLM from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=n_threads,
            verbose=False,
        )

        print(f"[INIT] Loading Piper voice ({os.path.basename(piper_model_path)})...")
        self.piper_voice = PiperVoice.load(piper_model_path, piper_config_path)
        self.piper_rate = getattr(self.piper_voice.config, 'sample_rate', 22050)
        print(f"[INIT] Piper sample rate: {self.piper_rate} Hz")

        self.vad = webrtcvad.Vad(self.VAD_AGGRESSIVENESS)
        self.audio = pyaudio.PyAudio()

        self.history: List[dict] = []
        self.max_history = 6

        print(f"[INIT] {personality.name} is ready.")
        if not constant_listen:
            print(f"[INIT] Wake phrases: {self.wake_phrases}")

    def _record_utterance(self, max_seconds: int = 20, pause_ms: int = 1500, label: str = "") -> Optional[str]:
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE,
        )

        if label:
            print(f"\n{label}", end="", flush=True)
        else:
            print("  [Listening...]", end="", flush=True)

        frames = []
        is_speaking = False
        silence_chunks = 0
        pause_chunks = int(pause_ms / self.CHUNK_DURATION_MS)
        max_chunks = int(max_seconds * 1000 / self.CHUNK_DURATION_MS)

        try:
            for _ in range(max_chunks):
                data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                is_speech = self.vad.is_speech(data, self.SAMPLE_RATE)

                if is_speech:
                    if not is_speaking:
                        is_speaking = True
                        print(" 🎤", flush=True)
                    silence_chunks = 0
                    frames.append(data)
                elif is_speaking:
                    frames.append(data)
                    silence_chunks += 1
                    if silence_chunks > pause_chunks:
                        if label:
                            print("  (done)")
                        else:
                            print(f"  (done, {len(frames)} chunks)")
                        break
        finally:
            stream.stop_stream()
            stream.close()

        if not frames:
            if label:
                pass
            else:
                print("  (no speech)")
            return None

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        return tmp.name

    def _transcribe(self, wav_path: str) -> str:
        result = self.whisper.transcribe(wav_path, fp16=False)
        os.unlink(wav_path)
        text = result["text"].strip()
        return text

    def _contains_wake_phrase(self, text: str) -> bool:
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in self.wake_phrases)

    def _strip_wake_phrases(self, text: str) -> str:
        pattern = re.compile(
            r'^(' + '|'.join(re.escape(p) for p in self.wake_phrases) + r')[,;\s]*',
            re.IGNORECASE
        )
        return pattern.sub('', text).strip()

    def _generate_response(self, user_text: str) -> str:
        messages = [{"role": "system", "content": self.personality.get_system_prompt()}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})

        prompt = self._format_chat_prompt(messages)

        output = self.llm(
            prompt,
            max_tokens=self.personality.max_tokens,
            temperature=self.personality.temperature,
            stop=["</s>", "User:", f"{self.personality.name}:"],
            echo=False,
        )

        response = output["choices"][0]["text"].strip()

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": response})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return response

    def _format_chat_prompt(self, messages: List[dict]) -> str:
        prompt = ""
        for m in messages:
            if m["role"] == "system":
                prompt += f"<|system|>\n{m['content']}</s>\n"
            elif m["role"] == "user":
                prompt += f"<|user|>\n{m['content']}</s>\n"
            elif m["role"] == "assistant":
                prompt += f"<|assistant|>\n{m['content']}</s>\n"
        prompt += "<|assistant|>\n"
        return prompt

    def _synthesize_to_wav(self, text: str) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.piper_rate)
            self.piper_voice.synthesize_wav(text, wf)
        return tmp.name

    def _play_wav(self, wav_path: str):
        try:
            result = subprocess.run(
                ["aplay", wav_path],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                return
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"  [aplay error: {e}]")

        wf = wave.open(wav_path, "rb")
        try:
            stream = self.audio.open(
                format=self.audio.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
            )
            chunk_size = 1024
            data = wf.readframes(chunk_size)
            while data:
                stream.write(data)
                data = wf.readframes(chunk_size)
            stream.stop_stream()
            stream.close()
        finally:
            wf.close()

    def _speak(self, text: str):
        print(f"[{self.personality.name}] {text}")
        wav_path = self._synthesize_to_wav(text)
        self._play_wav(wav_path)
        os.unlink(wav_path)

    def _speak_with_interrupt(self, text: str):
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            return True, None

        self.speaking = True
        interrupt_text = [None]

        def listener():
            time.sleep(0.3)
            wav = self._record_utterance(max_seconds=20, pause_ms=1000)
            if wav and self.speaking:
                txt = self._transcribe(wav)
                if txt.strip():
                    interrupt_text[0] = txt
                    self.speaking = False

        listener_thread = threading.Thread(target=listener)
        listener_thread.start()

        for sentence in sentences:
            if not self.speaking:
                print("  [Interrupted mid-speech]")
                break

            print(f"[{self.personality.name}] {sentence}")
            wav_path = self._synthesize_to_wav(sentence)
            self._play_wav(wav_path)
            os.unlink(wav_path)

        was_interrupted = interrupt_text[0] is not None
        self.speaking = False
        listener_thread.join(timeout=3)

        return not was_interrupted, interrupt_text[0]

    def _handle_active_turn(self):
        """One full conversation turn in active mode"""
        print(f"\n[🎙 {self.personality.name} is listening...]")

        wav = self._record_utterance(max_seconds=25, pause_ms=1500, label="  Listening")
        if not wav:
            return

        self.last_active_time = time.time()
        text = self._transcribe(wav)
        print(f"[You] {text}")

        if not text:
            return

        # Sleep commands (only in sleep/wake mode)
        if not self.constant_listen and any(cmd in text.lower() for cmd in ["goodbye", "go to sleep", "sleep now", "that's all"]):
            self._speak("Going to sleep. Call me when you need me.")
            self.state = self.STATE_SLEEPING
            self.history.clear()
            return

        response = self._generate_response(text)
        completed, interrupt_text = self._speak_with_interrupt(response)

        if not completed and interrupt_text:
            print(f"[🚨 Interrupted] {interrupt_text}")
            response2 = self._generate_response(interrupt_text)
            self._speak_with_interrupt(response2)

    def run(self):
        if self.constant_listen:
            greeting = f"Hello, I'm {self.personality.name}. I'm in constant listen mode. Just talk to me."
        else:
            wake_str = ", ".join(f'"{p}"' for p in self.wake_phrases[:3])
            greeting = f"Hello, I'm {self.personality.name}. Say {wake_str} to wake me."

        self._speak(greeting)

        try:
            while True:
                # === CONSTANT LISTEN MODE ===
                if self.constant_listen:
                    self._handle_active_turn()

                # === SLEEP/WAKE MODE ===
                else:
                    if self.state == self.STATE_SLEEPING:
                        print(f"\n[😴 {self.personality.name} is sleeping... Waiting for wake phrase]")

                        wav = self._record_utterance(max_seconds=15, pause_ms=1000)
                        if not wav:
                            continue

                        text = self._transcribe(wav)
                        print(f"[Heard] '{text}'")

                        if self._contains_wake_phrase(text):
                            self.state = self.STATE_ACTIVE
                            self.last_active_time = time.time()
                            clean_text = self._strip_wake_phrases(text)

                            if clean_text:
                                print(f"[You] {clean_text}")
                                response = self._generate_response(clean_text)
                                completed, interrupt_text = self._speak_with_interrupt(response)
                                if not completed and interrupt_text:
                                    print(f"[🚨 Interrupted] {interrupt_text}")
                                    response2 = self._generate_response(interrupt_text)
                                    self._speak_with_interrupt(response2)
                            else:
                                self._speak("Yes? I'm listening.")
                        else:
                            print(f"  (ignored — no wake phrase)")

                    elif self.state == self.STATE_ACTIVE:
                        # Check sleep timeout
                        if time.time() - self.last_active_time > self.sleep_timeout:
                            self._speak("Going to sleep due to inactivity.")
                            self.state = self.STATE_SLEEPING
                            self.history.clear()
                            continue

                        self._handle_active_turn()

        except KeyboardInterrupt:
            print("\n[Shutdown] Interrupted by user.")
        finally:
            self.audio.terminate()


def main():
    parser = argparse.ArgumentParser(description="Local AI Voice Assistant")
    parser.add_argument(
        "--constant",
        action="store_true",
        help="Constant listen mode — no wake phrase, always active",
    )
    args = parser.parse_args()

    personality = Personality(
        name="Atlas",
        role="helpful AI assistant",
        tone="blunt",
        temperature=0.7,
        max_tokens=150,
    )

    wake_phrases = ["hey atlas", "atlas", "okay atlas", "yo atlas"]

    import multiprocessing
    cpu_count = multiprocessing.cpu_count()

    assistant = VoiceAssistant(
        personality=personality,
        model_path=os.path.expanduser("~/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
        piper_model_path=os.path.expanduser("~/piper-voices/en_US-lessac-medium.onnx"),
        piper_config_path=os.path.expanduser("~/piper-voices/en_US-lessac-medium.onnx.json"),
        whisper_model="base",
        n_threads=cpu_count,
        wake_phrases=wake_phrases,
        sleep_timeout=30,
        constant_listen=args.constant,
    )

    assistant.run()


if __name__ == "__main__":
    main()
