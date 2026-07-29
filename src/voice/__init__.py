"""voice — KunPeng 声纹识别与语音交互管道

全链路本地语音处理：麦克风采集 → VAD → KWS 唤醒词 → ASR → 声纹识别 → 身份决策

参考: 声纹识别方案_v2.docx (FunASR + CAM++ + sherpa-onnx)
"""

from .voiceprint import VoiceprintDB

__all__ = ["VoiceprintDB"]
