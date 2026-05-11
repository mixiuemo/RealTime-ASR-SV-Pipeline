#!/usr/bin/env python3
import sherpa_onnx
import sys
import queue
import threading
import time
import numpy as np
import pyaudio
import shutil 
import wave
import os
import json
from websockets.sync.server import serve
import subprocess
from pathlib import Path

# ========================= 核心配置 =========================
MODEL_DIR = "./models/sherpa-onnx-funasr-nano-int8-2025-12-30"
VAD_MODEL = "./models/silero_vad.onnx"
SV_MODEL = "./models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
SPEAKER_DB_DIR = "./speaker_db"
SAVE_DIR = "./captured_audio"

# 策略参数 
HARD_TRUNCATE_SEC = 5.0 
HARD_TRUNCATE_SAMPLES = int(16000 * HARD_TRUNCATE_SEC)
SV_WINDOW_SAMPLES = int(2.0 * 16000)
SILENCE_THRESHOLD_S = 0.85 # 闭嘴 850ms 判定结算
SV_THRESHOLD = 0.48        # 声纹判定阈值

# 全局通信与同步
audio_queue = queue.Queue()       
ws_queue = queue.Queue()          
connected_clients = set()
clients_lock = threading.Lock()
buffer_lock = threading.Lock() 
print_lock = threading.Lock()

killed = False

# ========================= 1. 基础工具函数 =========================
def save_wav(audio_data, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        samples = np.array(audio_data)
        # 归一化，防止破音
        max_val = np.max(np.abs(samples))
        if max_val > 0.01: samples *= (0.9 / max_val)
        
        pcm_samples = (samples * 32767).astype(np.int16)
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
            wf.writeframes(pcm_samples.tobytes())
    except Exception as e:
        print(f"写入文件失败: {e}")

def load_audio_ffmpeg(file_path):
    cmd = ['ffmpeg', '-i', str(file_path), '-f', 'f32le', '-acodec', 'pcm_f32le', 
           '-ac', '1', '-ar', '16000', '-loglevel', 'quiet', '-']
    try:
        pipe = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
        return np.frombuffer(pipe.stdout, dtype=np.float32)
    except: return None

# ========================= 2. ChannelManager (核心逻辑类) =========================
class ChannelManager:
    def __init__(self, recognizer, extractor, speaker_db):
        self.recognizer = recognizer
        self.extractor = extractor
        self.speaker_db = speaker_db
        
        # --- 核心状态位  ---
        self.locked_speaker = None
        self.locked_score = 0.0
        self.sv_confirmed_this_turn = False
        self.settled_final_text = ""      
        self.accumulated_session_text = "" 
        self.last_speech_time = time.time()
        
        self.preview_audio_buffer = []    
        self.last_sv_check_time = 0
        self.last_preview_time = 0
        
    def reset(self, deep=False):
        """ deepReset"""
        self.settled_final_text = ""
        self.accumulated_session_text = ""
        with buffer_lock: self.preview_audio_buffer = []
        if deep:
            self.locked_speaker = None
            self.locked_score = 0.0
            self.sv_confirmed_this_turn = False
            self.last_speech_time = time.time()
            with print_lock:
                print("\n\033[94m♻️ [系统] 深度重置，等待下一轮对话...\033[0m")

    def push_samples(self, samples):
        now = time.time()
        self.last_speech_time = now
        
        with buffer_lock:
            self.preview_audio_buffer.extend(samples.tolist())
            buf_size = len(self.preview_audio_buffer)

        # 1. 硬截断积木 (5s)
        if buf_size >= HARD_TRUNCATE_SAMPLES:
            self.process_buffer(is_truncated=True)
            return

        # 2. 滑动声纹识别 (1.5s 窗口)
        if not self.sv_confirmed_this_turn and buf_size >= SV_WINDOW_SAMPLES:
            if (now - self.last_sv_check_time) > 1.2:
                self.last_sv_check_time = now
                snapshot = np.array(self.preview_audio_buffer[-SV_WINDOW_SAMPLES:])
                threading.Thread(target=self.identify_speaker_task, args=(snapshot,)).start()

        # 3. 实时预览推送
        if (now - self.last_preview_time) > 0.6:
            self.last_preview_time = now
            with buffer_lock: snapshot = np.array(self.preview_audio_buffer)
            threading.Thread(target=self.do_preview, args=(snapshot,)).start()

    def identify_speaker_task(self, audio):
        """识别并透传分值"""
        if not self.speaker_db: return
        try:
            s = self.extractor.create_stream(); s.accept_waveform(16000, audio)
            emb = np.array(self.extractor.compute(s))
            
            best_name, max_s = "陌生人", 0.0
            for name, db_emb in self.speaker_db.items():
                score = np.dot(emb, db_emb) / (np.linalg.norm(emb) * np.linalg.norm(db_emb))
                if score > max_s: max_s = score; best_name = name
            
            final_score = round(float(max_s), 2)
            
            # 如果分值达到门槛，执行锁定
            if max_s >= SV_THRESHOLD:
                self.locked_speaker = best_name
                self.locked_score = final_score
                if not best_name.startswith("OP_"): # 非话务员即锁定正主
                    self.sv_confirmed_this_turn = True
                self.send_to_ws("speaker_id", "", best_name, final_score)
            else:
                self.send_to_ws("speaker_id", "", "陌生人", final_score)
        except Exception as e:
            print(f"声纹识别线程异常: {e}")

    def do_preview(self, audio):
        """执行预览 ASR 并推送到 WebSocket"""
        try:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(16000, audio)
            self.recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            
            if text and text not in ["嗯", "啊", "呃", "。"]:
                self.accumulated_session_text = text
                full_display = self.settled_final_text + self.accumulated_session_text
                spk = self.locked_speaker if self.locked_speaker else "识别中..."
                score = self.locked_score
                
                # WebSocket 实时推
                self.send_to_ws("preview", full_display, spk, score)
                
                # 控制台实时显
                with print_lock:
                    sys.stdout.write(f"\r\033[K\033[90m[预览 - {spk}({score})] {full_display}\033[0m")
                    sys.stdout.flush()
        except: pass

    def process_buffer(self, is_truncated):
        """核心修复：区分‘片段结算’与‘最终结案’"""
        with buffer_lock:
            if not self.preview_audio_buffer: return
            segment = np.array(self.preview_audio_buffer)
            self.preview_audio_buffer = []
        
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, segment)
        self.recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        
        if text and text not in ["嗯", "啊", "呃", "。"]:
            self.settled_final_text += text
            spk, score = (self.locked_speaker, self.locked_score) if self.locked_speaker else ("识别中...", 0.0)

            # 🚩 修改点：如果是 VAD 断句，发 preview 以便前端更新同一行，不准起新行
            if is_truncated:
                msg_type = "truncate"
            else:
                msg_type = "preview" 

            self.send_to_ws(msg_type, self.settled_final_text, spk, score)
            
            # 存音逻辑不变
            save_wav(segment, f"{SAVE_DIR}/{spk}_{int(time.time()*1000)}.wav")
            
            with print_lock:
                sys.stdout.write("\r\033[K")
                color = "33" if is_truncated else "36" # 片段结算用青色区别
                label = "积木切片" if is_truncated else "片段结算"
                print(f"\033[1;{color}m[{label} - {spk}({score})] {self.settled_final_text}\033[0m")

    def send_to_ws(self, msg_type, text, speaker, score=0.0):
        ws_queue.put({
            "type": msg_type,
            "channel": "Python-Pro-ASR",
            "speaker": speaker,
            "score": score,
            "text": text
        })

# ========================= 3. 服务线程组 =========================
def ws_server_thread():
    """WebSocket 连接管理 (同步 Server 模式)"""
    def handler(ws):
        with clients_lock: connected_clients.add(ws)
        try:
            for _ in ws: pass
        except: pass
        finally:
            with clients_lock: connected_clients.discard(ws)

    print("🌐 [WS] WebSocket 服务正在启动 (Port: 8081)")
    with serve(handler, "0.0.0.0", 8081) as server:
        server.serve_forever()

def ws_broadcaster():
    """WebSocket 消息分发员"""
    while not killed:
        try:
            msg = ws_queue.get(timeout=0.1)
            msg_str = json.dumps(msg)
            with clients_lock:
                for c in list(connected_clients):
                    try: c.send(msg_str)
                    except: connected_clients.discard(c)
        except queue.Empty: continue

def watchdog_thread(manager):
    """只有看门狗负责发 FINAL 信号"""
    while not killed:
        time.sleep(0.1)
        silence_duration = time.time() - manager.last_speech_time
        
        # 闭嘴 850ms 且已经有确定的文字积木了
        if silence_duration > SILENCE_THRESHOLD_S and manager.settled_final_text:
            # 1. 检查 buffer 里是否还有最后一点“尾巴”没结算
            manager.process_buffer(is_truncated=False)
            
            # 2. 🚩 唯一发送 FINAL 信号的地方
            if manager.settled_final_text:
                final_spk = manager.locked_speaker if manager.locked_speaker else "未知"
                final_score = manager.locked_score
                
                # 发送结案通知
                manager.send_to_ws("final", manager.settled_final_text, final_spk, final_score)
                
                with print_lock:
                    print(f"\033[1;32m[✅ 最终结案 - {final_spk}({final_score})] {manager.settled_final_text}\033[0m")
            
            # 3. 彻底重置
            manager.reset(deep=True)

# ========================= 4. 主程序入口 =========================
def main():
    print("正在初始化 FunASR + Campplus 语音引擎...")
    asr_conf = {"encoder_adaptor": f"{MODEL_DIR}/encoder_adaptor.int8.onnx", "llm": f"{MODEL_DIR}/llm.int8.onnx", 
                "embedding": f"{MODEL_DIR}/embedding.int8.onnx", "tokenizer": f"{MODEL_DIR}/Qwen3-0.6B"}
    
    try:
        recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(**asr_conf, num_threads=4)
        extractor = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=SV_MODEL, num_threads=2))
        vad = sherpa_onnx.VoiceActivityDetector(sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(model=VAD_MODEL, threshold=0.45, max_speech_duration=30.0), 
            sample_rate=16000))
    except Exception as e:
        print(f"❌ 引擎加载失败，请检查模型路径: {e}"); return

    # 加载声纹底库
    path = Path(SPEAKER_DB_DIR)
    speaker_db = {}
    if path.exists():
        for folder in path.iterdir():
            if folder.is_dir():
                embs = []
                for f in folder.iterdir():
                    audio = load_audio_ffmpeg(f)
                    if audio is not None:
                        s = extractor.create_stream(); s.accept_waveform(16000, audio)
                        embs.append(np.array(extractor.compute(s)))
                if embs: 
                    speaker_db[folder.name] = np.mean(embs, axis=0)
                    print(f"  ✅ 声纹注册: {folder.name}")

    manager = ChannelManager(recognizer, extractor, speaker_db)

    # 启动后台任务
    threading.Thread(target=ws_server_thread, daemon=True).start()
    threading.Thread(target=ws_broadcaster, daemon=True).start()
    threading.Thread(target=watchdog_thread, args=(manager,), daemon=True).start()
    
    # 麦克风流
    pa = pyaudio.PyAudio()
    mic = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    print("\n" + "━"*65 + "\n  Python ChannelManager Pro 已上线 (监听中...)\n" + "━"*65 + "\n")

    try:
        while not killed:
            data = mic.read(1024, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            vad.accept_waveform(samples)

            if vad.is_speech_detected():
                manager.push_samples(samples)

            while not vad.empty():
                # VAD 回调结算
                manager.process_buffer(is_truncated=False)
                vad.pop()
                
    except KeyboardInterrupt: pass
    finally: mic.stop_stream(); mic.close(); pa.terminate()

if __name__ == "__main__": main()