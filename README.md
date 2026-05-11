本项目是一个基于 sherpa-onnx 框架实现的工业级语音调度系统。它集成了 ASR (自动语音识别) 与 SV (声纹识别) 技术，采用了类似 Java 高并发架构的状态机管理逻辑。

系统能够实时识别麦克风输入的语音，将其转化为文字，并根据预录入的声纹底库精准锁定说话人身份，同时通过 WebSocket 实时推送到监控大屏。

🌟 核心特性
    搭积木式切片逻辑 (Building Block Logic)：采用 5 秒硬截断技术，解决长语音识别的延迟与内存堆积问题，实现文字的“流式生长”。

    状态锁定机制 (State Locking)：复刻 Java 通道管理器逻辑，一旦声纹相似度突破阈值（默认 0.48），自动锁定身份，防止识别结果在多人环境中跳变。

    静音看门狗 (Silence Watchdog)：独立线程监控 850ms 静音判定，实现精准的“结案”与系统自动深度重置。

    置信度分级系统：实时计算声纹相似度分数并推送到前端，支持从“陌生人”到“极高置信度”的多级视觉反馈。

    全自动语料捕获：系统自动将识别到的语音片段按人名命名并保存为 16k/16bit WAV 文件，方便后续算法自学习。

🛠️ 技术栈
    ASR 模型：SenseVoice / FunASR Nano (由 sherpa-onnx 提供推理支持)。

    声纹模型：Campplus (3D Speaker)。

    推理引擎：ONNX Runtime (高效、跨平台)。

    实时通信：WebSockets (同步 Server 架构)。

    音频处理：PyAudio & NumPy & ffmpeg。



📂 项目结构
Plaintext
.
├── models/                  # 存放导出的 ONNX 模型文件
├── speaker_db/              # 声纹底库（每个文件夹一个名字，内存放 WAV 片段）
├── captured_audio/          # 系统运行过程中自动捕获的语音
├── funnano_vad_speaker.py   # Python 核心后端脚本
└── index.html               # WebSocket 实时监控大屏前端

🚀 快速开始
1. 环境配置
    建议使用 Conda 创建独立环境：
    Bash
    conda create -n sheronnx python=3.10
    conda activate sheronnx
    pip install -r requirements.txt

    注：需确保系统已安装 ffmpeg 并添加到环境变量。

2. 放置模型
    将模型文件放置在 models 目录下，确保路径与脚本内配置一致：
    sherpa-onnx-funasr-nano-int8-2025-12-30 
    silero_vad.onnx
    3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx

3. 运行系统
    Bash
    python funnano_vad_speaker.py

4. 开启监控大屏
    直接用浏览器打开 index.html。系统默认监听 8081 端口。

⚙️ 关键参数
    BLOCK_SIZE_SEC	5.0	强制切片时间，控制实时积木的长度
    SV_THRESHOLD	0.48	声纹相似度门槛，高于此值才进行身份锁定
    SILENCE_THRESHOLD_S	0.85	判定一句话彻底结束的静音时长

声纹更新：若需添加新成员，只需在 speaker_db 下新建文件夹并放入其 16k 采样率的 WAV 文件，重启系统即可自动加载，后续考虑增加数据库，直接从数据库读取。

日志说明：控制台绿色代表 [最终结案]，黄色代表 [积木切片]。
