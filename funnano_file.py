import sys
from pathlib import Path
import soundfile as sf
import sherpa_onnx
import numpy as np

def main():
    # ================= 配置区 =================
    MODEL_DIR = "./models/sherpa-onnx-funasr-nano-int8-2025-12-30"
    config = {
        "encoder_adaptor": f"{MODEL_DIR}/encoder_adaptor.int8.onnx",
        "llm": f"{MODEL_DIR}/llm.int8.onnx",
        "embedding": f"{MODEL_DIR}/embedding.int8.onnx",
        "tokenizer": f"{MODEL_DIR}/Qwen3-0.6B",
        "wav_file": f"{MODEL_DIR}/test_wavs/raokouling.wav",
    }
    # 每段切分的长度（秒）
    SEGMENT_SECONDS = 10 
    # ==========================================

    # 1. 初始化识别器
    print("正在初始化引擎...")
    recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
        encoder_adaptor=config["encoder_adaptor"],
        llm=config["llm"],
        embedding=config["embedding"],
        tokenizer=config["tokenizer"],
        num_threads=4,
        system_prompt="", # 留空以节省 Token 空间
        user_prompt="转写:",
        max_new_tokens=512,
    )

    # 2. 读取完整音频
    audio, sample_rate = sf.read(config["wav_file"], dtype="float32", always_2d=True)
    audio = audio[:, 0]
    total_samples = len(audio)
    segment_samples = SEGMENT_SECONDS * sample_rate

    # 3. 分段逻辑
    full_result = []
    print(f"音频总长: {total_samples/sample_rate:.2f} 秒，将分为 {int(np.ceil(total_samples/segment_samples))} 段识别...")

    for i in range(0, total_samples, segment_samples):
        # 截取当前分段
        chunk = audio[i : i + segment_samples]
        
        # 识别当前段
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, chunk)
        recognizer.decode_stream(stream)
        
        text = stream.result.text.strip()
        if text:
            print(f"[段落 {i//segment_samples + 1}]: {text}")
            full_result.append(text)

    # 4. 汇总输出
    print("\n" + "="*20 + " 完整拼接结果 " + "="*20)
    print("".join(full_result))
    print("="*54)

if __name__ == "__main__":
    main()