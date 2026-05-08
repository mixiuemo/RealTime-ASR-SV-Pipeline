#!/usr/bin/env python3
import sherpa_onnx
import sys
import os
import queue
import threading
import time
import numpy as np
import pyaudio
import shutil 

# ========================= 全局队列与状态 =========================
audio_queue = queue.Queue()       
final_task_queue = queue.Queue()  
print_lock = threading.Lock()     
killed = False

shared_preview_lines = 0  
NOISE_WORDS = {"/sil", "嗯", "嗯。", "啊", "啊。", "。", "，", "呃", ""}

def record_audio_thread():
    """后台录音工：永远不阻塞地获取音频"""
    pa = pyaudio.PyAudio()
    chunk_size = 1024 
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=chunk_size
    )
    while not killed:
        try:
            data = stream.read(chunk_size, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.float32)
            audio_queue.put(samples)
        except Exception:
            pass
    stream.stop_stream()
    stream.close()
    pa.terminate()

def final_decode_thread(recognizer):
    """重装步兵：后台线程，负责结算完整句子"""
    global shared_preview_lines
    accumulated_final_text = ""  # 【新增】：跨切片的最终文本缝合器
    
    while not killed:
        try:
            pure_audio, is_truncated = final_task_queue.get(timeout=1)
            
            f_stream = recognizer.create_stream()
            f_stream.accept_waveform(16000, pure_audio)
            recognizer.decode_stream(f_stream)
            
            f_text = f_stream.result.text.strip()
            
            if f_text and (f_text not in NOISE_WORDS):
                # 把每次切片算出的小段结果拼起来
                accumulated_final_text += f_text
                
                with print_lock:
                    if shared_preview_lines > 0:
                        sys.stdout.write(f"\033[{shared_preview_lines}A")
                    
                    sys.stdout.write("\r\033[J")
                    
                    if is_truncated:
                        # 打印当前拼接的全部内容，加上省略号表示未完待续
                        print(f"\033[1;33m  [超长切片] {accumulated_final_text}...\033[0m\n")
                    else:
                        # 打印最终的完整长句
                        print(f"\033[1;32m  [本句完毕] {accumulated_final_text}\033[0m\n")
                    
                    shared_preview_lines = 0
            
            # 【关键逻辑】：只要是正常停顿（非截断），不管有没有识别出字，都必须清空缝合器，准备迎接下一次开口
            if not is_truncated:
                accumulated_final_text = ""
                
        except queue.Empty:
            pass

def main():
    MODEL_DIR = "./models/sherpa-onnx-funasr-nano-int8-2025-12-30"
    VAD_MODEL = "./models/silero_vad.onnx"
    
    PREVIEW_INTERVAL = 1.0     
    MAX_SPEECH_SEC = 8.0       

    asr_config = {
        "encoder_adaptor": f"{MODEL_DIR}/encoder_adaptor.int8.onnx",
        "llm": f"{MODEL_DIR}/llm.int8.onnx",
        "embedding": f"{MODEL_DIR}/embedding.int8.onnx",
        "tokenizer": f"{MODEL_DIR}/Qwen3-0.6B",
    }

    print("正在初始化双轨语音引擎...")
    
    vad_config = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=VAD_MODEL,
            threshold=0.6,             
            min_speech_duration=0.25,  
            min_silence_duration=1.0,  
            max_speech_duration=MAX_SPEECH_SEC,   
        ),
        sample_rate=16000,
    )
    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)

    recognizer_preview = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
        **asr_config,
        num_threads=2, 
        system_prompt="查号台助手，仅转写文字。",
    )

    recognizer_final = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
        **asr_config,
        num_threads=4, 
        system_prompt="查号台助手，仅转写文字，不要解释。",
    )

    global killed
    threading.Thread(target=record_audio_thread, daemon=True).start()
    threading.Thread(target=final_decode_thread, args=(recognizer_final,), daemon=True).start()

    full_audio_buffer = []           
    preview_chunk_buffer = []        
    session_preview_text = ""        
    
    last_preview_time = time.time()
    global shared_preview_lines

    print("\n" + "━"*65)
    print("  查号系统 ASR (O(1)极速增量 + 超长句无缝拼接版) 已启动")
    print("  黄字 [超长切片]：底层防崩溃自动截断，UI 层自动拼接")
    print("  绿字 [本句完毕]：正常停顿，输出完整全句")
    print("━"*65 + "\n")

    try:
        while True:
            try:
                samples = audio_queue.get(timeout=0.05)
                full_audio_buffer = np.concatenate([full_audio_buffer, samples])
                preview_chunk_buffer = np.concatenate([preview_chunk_buffer, samples])
                vad.accept_waveform(samples)
            except queue.Empty:
                pass

            if not vad.is_speech_detected():
                if len(full_audio_buffer) > 16000:
                    full_audio_buffer = full_audio_buffer[-16000:]
                if len(preview_chunk_buffer) > 16000:
                    preview_chunk_buffer = preview_chunk_buffer[-16000:]

            # ================= [轻骑兵：纯增量极速预览] =================
            if vad.is_speech_detected():
                curr_time = time.time()
                if curr_time - last_preview_time > PREVIEW_INTERVAL:
                    
                    if len(preview_chunk_buffer) > 0:
                        p_stream = recognizer_preview.create_stream()
                        p_stream.accept_waveform(16000, np.array(preview_chunk_buffer))
                        recognizer_preview.decode_stream(p_stream)
                        p_text = p_stream.result.text.strip()
                        
                        if p_text and (p_text not in NOISE_WORDS):
                            session_preview_text += p_text
                            
                            term_width = shutil.get_terminal_size().columns
                            display_width = sum(2 if ord(c) > 127 else 1 for c in session_preview_text) + 12
                            current_lines = (display_width - 1) // term_width if display_width > 0 else 0
                            
                            with print_lock:
                                if shared_preview_lines > 0:
                                    sys.stdout.write(f"\033[{shared_preview_lines}A")
                                sys.stdout.write("\r\033[J")
                                print(f"\033[90m  [预览中] {session_preview_text}\033[0m", end="", flush=True)
                                shared_preview_lines = current_lines
                        
                        preview_chunk_buffer = []
                            
                    last_preview_time = curr_time

            # ================= [一句话结束/被截断：清理战场] =================
            while not vad.empty():
                segment = vad.front 
                
                duration_sec = len(segment.samples) / 16000.0
                is_truncated = duration_sec >= (MAX_SPEECH_SEC - 0.1)
                
                final_task_queue.put((segment.samples, is_truncated))
                
                vad.pop()
                full_audio_buffer = [] 
                preview_chunk_buffer = [] 
                
                # 【新增】：只有真正停顿断句时，才清空灰色的预览文本。
                # 如果只是被底层切片了，保留记忆继续拼，保证视觉流畅！
                if not is_truncated:
                    session_preview_text = "" 

    except KeyboardInterrupt:
        killed = True
        print("\n\n[系统消息] 退出程序")

if __name__ == "__main__":
    main()