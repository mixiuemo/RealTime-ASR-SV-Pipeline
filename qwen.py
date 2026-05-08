import sys
from pathlib import Path
import soundfile as sf
import sherpa_onnx

def main():
    # ================= 1. 路径与参数配置 =================
    MODEL_DIR = Path("./models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25")
    
    # 官方命令行中使用的“必中”热词
    # 针对你的查号系统，你可以把这里替换成“财务处,张处长,李主任”
    official_hotwords = "骨质疏松症患者,咬死山前,紫色柿子树,年年恋牛娘,灰黑灰化肥,走出香港"

    config = {
        "conv_frontend": str(MODEL_DIR / "conv_frontend.onnx"),
        "encoder": str(MODEL_DIR / "encoder.int8.onnx"),
        "decoder": str(MODEL_DIR / "decoder.int8.onnx"),
        "tokenizer": str(MODEL_DIR / "tokenizer"),
        "wav_file": str(MODEL_DIR / "test_wavs/raokouling.wav"),
    }

    # ================= 2. 初始化识别器 =================
    # 复刻官方命令行的所有核心参数
    print("正在以官方最强配置初始化 Qwen3-ASR...")
    recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=config["conv_frontend"],
        encoder=config["encoder"],
        decoder=config["decoder"],
        tokenizer=config["tokenizer"],
        # 注入热词
        hotwords=official_hotwords,
        # 官方推荐：Qwen3 离线版必须是 128
        feature_dim=128,
        # 对应命令行的 --qwen3-asr-max-new-tokens=512
        max_new_tokens=512,
        # 脑容量，确保长音频不截断
        max_total_len=512,
        num_threads=4,
        sample_rate=16000,
        temperature=1e-6, # 越小越严谨
        top_p=0.8,
    )

    # ================= 3. 读取并识别 =================
    if not Path(config["wav_file"]).exists():
        print(f"找不到音频文件: {config['wav_file']}")
        return

    audio, sample_rate = sf.read(config["wav_file"], dtype="float32", always_2d=True)
    audio = audio[:, 0]

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    
    print("正在执行官方级热词偏置解码...")
    recognizer.decode_stream(stream)
    
    # ================= 4. 输出结果 =================
    print("\n" + "★"*40)
    print("【官方热词增强版识别结果】")
    print(stream.result.text)
    print("★"*40)

if __name__ == "__main__":
    main()