#!/usr/bin/env python3
import sherpa_onnx
import sys
import queue
import threading
import time
import numpy as np
import pyaudio
import shutil 
import subprocess  # 【新增】：用于调用 ffmpeg
from pathlib import Path

# ========================= 全局队列与状态 =========================
audio_queue = queue.Queue()       
final_task_queue = queue.Queue()  
print_lock = threading.Lock()     
killed = False

shared_preview_lines = 0  
NOISE_WORDS = {"/sil", "嗯", "嗯。", "啊", "啊。", "。", "，", "呃", ""}

# ========================= 音频万能解码器 =========================
def load_audio_with_ffmpeg(file_path, target_sr=16000, normalize=True):
    """
    加强版万能解码器：支持格式转换 + 重采样 + 自动音量归一化
    """
    command = [
        'ffmpeg',
        '-i', str(file_path),
        '-f', 'f32le',         
        '-acodec', 'pcm_f32le',
        '-ac', '1',            
        '-ar', str(target_sr), 
        '-loglevel', 'quiet',  
        '-'                    
    ]
    try:
        pipe = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        audio_data = np.frombuffer(pipe.stdout, dtype=np.float32)
        
        # ==========================================
        # 【新增】：极速矩阵级峰值归一化 (Peak Normalization)
        # ==========================================
        if normalize and len(audio_data) > 0:
            max_amp = np.max(np.abs(audio_data))
            # 只有当音量不是完全静音，且确实需要调整时才处理
            if max_amp > 0.0:
                # 0.9 是安全余量(Headroom)，防止大模型底层计算时溢出破音
                audio_data = audio_data * (0.9 / max_amp) 
                
        return audio_data, target_sr
    except Exception as e:
        print(f"  [错误] 无法解码音频文件 {file_path}，详细: {e}")
        return None, None

# ========================= 声纹特征算法 =========================
def compute_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def init_speaker_db(extractor, db_dir):
    """启动时：读取文件夹，按人物提取平均声纹特征"""
    speaker_db = {}
    db_path = Path(db_dir)
    if not db_path.exists():
        print(f"\n[系统提示] 找不到声纹底库文件夹 {db_dir}，当前为无底库模式。\n")
        return speaker_db
        
    print(f"正在从 {db_dir} 加载声纹底库...")
    
    # 【改动】：遍历一级目录（按人名分类的文件夹）
    for speaker_folder in db_path.iterdir():
        if not speaker_folder.is_dir():
            continue
            
        speaker_name = speaker_folder.name
        embeddings = []
        
        # 遍历该人名文件夹下的所有音频文件
        for audio_file in speaker_folder.iterdir():
            if audio_file.name.startswith('.'): # 忽略隐藏文件（如 Mac 的 .DS_Store）
                continue
                
            # 使用万能解码器读取
            audio, sample_rate = load_audio_with_ffmpeg(audio_file)
            if audio is not None and len(audio) > 0:
                stream = extractor.create_stream()
                stream.accept_waveform(sample_rate, audio)
                emb = extractor.compute(stream)
                embeddings.append(np.array(emb))
        
        # 【核心逻辑】：如果这个人有多段音频，将它们的特征相加求平均，得到最稳健的灵魂声纹
        if embeddings:
            mean_embedding = np.mean(embeddings, axis=0)
            speaker_db[speaker_name] = mean_embedding
            print(f"  - 注册成功: {speaker_name} (综合了 {len(embeddings)} 段音频)")
        else:
            print(f"  - 注册跳过: {speaker_name} (文件夹内没有有效音频)")
            
    return speaker_db

def identify_speaker(extractor, speaker_db, audio_data, threshold=0.45):
    """实战期：识别当前说话人"""
    if not speaker_db:
        return "未知说话人"
        
    stream = extractor.create_stream()
    stream.accept_waveform(16000, audio_data)
    curr_embedding = np.array(extractor.compute(stream))
    
    best_name = "未知说话人"
    best_score = -1.0
    
    for name, db_embedding in speaker_db.items():
        score = compute_similarity(curr_embedding, db_embedding)
        if score > best_score:
            best_score = score
            best_name = name
            
    if best_score < threshold:
        return f"陌生人(相似度:{best_score:.2f})"
        
    return f"{best_name}({best_score:.2f})"

# ========================= 以下为主程序 (基本不变) =========================
def record_audio_thread():
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

def final_decode_thread(recognizer, extractor, speaker_db):
    global shared_preview_lines
    accumulated_final_text = ""  
    
    while not killed:
        try:
            pure_audio, is_truncated = final_task_queue.get(timeout=1)
            
            f_stream = recognizer.create_stream()
            f_stream.accept_waveform(16000, pure_audio)
            recognizer.decode_stream(f_stream)
            f_text = f_stream.result.text.strip()
            
            speaker_identity = "未知"
            if len(pure_audio) > 8000:
                speaker_identity = identify_speaker(extractor, speaker_db, pure_audio)
            
            if f_text and (f_text not in NOISE_WORDS):
                accumulated_final_text += f_text
                
                with print_lock:
                    if shared_preview_lines > 0:
                        sys.stdout.write(f"\033[{shared_preview_lines}A")
                    
                    sys.stdout.write("\r\033[J")
                    
                    if is_truncated:
                        print(f"\033[1;33m  [自动切片 - {speaker_identity}] {accumulated_final_text}...\033[0m\n")
                    else:
                        print(f"\033[1;32m  [本句完毕 - {speaker_identity}] {accumulated_final_text}\033[0m\n")
                    
                    shared_preview_lines = 0
            
            if not is_truncated:
                accumulated_final_text = ""
                
        except queue.Empty:
            pass

def main():
    MODEL_DIR = "./models/sherpa-onnx-funasr-nano-int8-2025-12-30"
    VAD_MODEL = "./models/silero_vad.onnx"
    SV_MODEL = "./models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
    SPEAKER_DB_DIR = "./speaker_db"
    
    PREVIEW_INTERVAL = 1.0     
    MAX_SPEECH_SEC = 8.0       

    asr_config = {
        "encoder_adaptor": f"{MODEL_DIR}/encoder_adaptor.int8.onnx",
        "llm": f"{MODEL_DIR}/llm.int8.onnx",
        "embedding": f"{MODEL_DIR}/embedding.int8.onnx",
        "tokenizer": f"{MODEL_DIR}/Qwen3-0.6B",
    }

    print("正在初始化 双轨ASR + 声纹SV 融合引擎...")
    
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
    
    official_hotwords = "财务处,张处长,李主任,新殿光,张三,李四"
    
    recognizer_preview = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
        **asr_config,
        num_threads=2, 
        system_prompt="查号台助手，仅转写文字。",
        hotwords=official_hotwords,
    )

    recognizer_final = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
        **asr_config,
        num_threads=4, 
        system_prompt="查号台助手，仅转写文字，不要解释。",
        hotwords=official_hotwords,
    )

    sv_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=SV_MODEL,
        num_threads=2,
        debug=False,
    )
    speaker_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(sv_config)
    
    speaker_db = init_speaker_db(speaker_extractor, SPEAKER_DB_DIR)

    global killed
    threading.Thread(target=record_audio_thread, daemon=True).start()
    threading.Thread(target=final_decode_thread, args=(recognizer_final, speaker_extractor, speaker_db), daemon=True).start()

    full_audio_buffer = []           
    preview_chunk_buffer = []        
    session_preview_text = ""        
    
    last_preview_time = time.time()
    global shared_preview_lines

    print("\n" + "━"*65)
    print("  查号系统 ASR + SV (语音识别 + 听音辨人版) 已启动")
    print(f"  已挂载声纹库人数: {len(speaker_db)} 人")
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

            while not vad.empty():
                segment = vad.front 
                
                duration_sec = len(segment.samples) / 16000.0
                is_truncated = duration_sec >= (MAX_SPEECH_SEC - 0.1)
                
                final_task_queue.put((segment.samples, is_truncated))
                
                vad.pop()
                full_audio_buffer = [] 
                preview_chunk_buffer = [] 
                
                if not is_truncated:
                    session_preview_text = "" 

    except KeyboardInterrupt:
        killed = True
        print("\n\n[系统消息] 退出程序")

if __name__ == "__main__":
    main()