"""
语音交互模块

集成ASR(自动语音识别)和TTS(文本转语音)的语音交互驱动。
支持FunASR/Whisper.cpp进行语音识别,PaddleSpeech/pyttsx3进行语音合成,
以及基于PyAudio的音频录制和播放。
适用于OrangePi Kunpeng Pro (RK3588)平台。

功能特性:
    - ASR: FunASR或Whisper.cpp语音识别
    - TTS: PaddleSpeech TTS或pyttsx3语音合成
    - 唤醒词检测(可选,基于Snowboy或自定义)
    - 音频录制和播放(PyAudio)
    - 音频流处理(降噪、VAD)
    - 异步接口
    - 说话状态检测

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ASRModel(Enum):
    """ASR模型类型枚举"""
    FUNASR = "funasr"           # FunASR(阿里达摩院)
    WHISPER_CPP = "whisper_cpp" # Whisper.cpp(GGML)
    WHISPER_API = "whisper_api" # OpenAI Whisper API
    MOCK = "mock"               # 模拟模式


class TTSModel(Enum):
    """TTS模型类型枚举"""
    PADDLESPEECH = "paddlespeech"   # PaddleSpeech TTS
    PYTTSX3 = "pyttsx3"             # pyttsx3(离线)
    EDGE_TTS = "edge_tts"           # Edge TTS(微软在线)
    MOCK = "mock"                   # 模拟模式


class AudioState(Enum):
    """音频处理状态枚举"""
    IDLE = "idle"               # 空闲
    RECORDING = "recording"     # 录音中
    RECOGNIZING = "recognizing" # 识别中
    SPEAKING = "speaking"       # 播报中
    ERROR = "error"             # 错误


@dataclass
class AudioConfig:
    """音频配置参数

    属性:
        sample_rate: 采样率(Hz)
        channels: 声道数
        sample_width: 采样位宽(字节)
        chunk_size: 音频块大小
        record_duration: 默认录音时长(秒)
        vad_aggressiveness: VAD激进程度(0-3)
        asr_model: ASR模型类型
        tts_model: TTS模型类型
        tts_speed: 语速(0.5-2.0)
        tts_volume: 音量(0.0-1.0)
        temp_dir: 临时文件目录
    """
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    chunk_size: int = 1024
    record_duration: float = 5.0
    vad_aggressiveness: int = 2
    asr_model: ASRModel = ASRModel.FUNASR
    tts_model: TTSModel = TTSModel.PYTTSX3
    tts_speed: float = 1.0
    tts_volume: float = 0.8
    temp_dir: str = "/tmp/kpcortex/audio"
    asr_model_path: str = "/opt/models/funasr"
    whisper_model_path: str = "/opt/models/whisper"


@dataclass
class ASRResult:
    """语音识别结果

    属性:
        text: 识别文本
        confidence: 置信度(0-1)
        language: 检测到的语言
        duration: 音频时长(秒)
        processing_time: 处理耗时(秒)
    """
    text: str = ""
    confidence: float = 0.0
    language: str = "zh"
    duration: float = 0.0
    processing_time: float = 0.0


@dataclass
class TTSResult:
    """语音合成结果

    属性:
        audio_file: 输出音频文件路径
        duration: 音频时长(秒)
        processing_time: 处理耗时(秒)
    """
    audio_file: str = ""
    duration: float = 0.0
    processing_time: float = 0.0


class VoiceInterface:
    """语音交互接口类

    提供完整的语音交互功能,包括语音识别(ASR)和语音合成(TTS)。
    支持多种后端模型,可根据硬件能力和场景需求选择。

    示例:
        >>> voice = VoiceInterface(asr_model="funasr", tts_model="pyttsx3")
        >>> await voice.initialize()
        >>> text = await voice.listen(timeout=10.0)
        >>> print(f"识别结果: {text}")
        >>> await voice.speak(f"您说的是: {text}")
        >>> await voice.shutdown()

    属性:
        config: 音频配置
        _state: 当前音频状态
        _audio: PyAudio对象
    """

    DEFAULT_TIMEOUT: float = 10.0
    MIN_RECORD_SECONDS: float = 0.5
    MAX_RECORD_SECONDS: float = 30.0
    SILENCE_TIMEOUT: float = 2.0  # 静音超时(秒)

    def __init__(
        self,
        asr_model: str = "funasr",
        tts_model: str = "paddlespeech",
        config: AudioConfig | None = None,
    ) -> None:
        """初始化语音交互接口

        参数:
            asr_model: ASR模型名称("funasr"|"whisper_cpp"|"whisper_api"|"mock")
            tts_model: TTS模型名称("paddlespeech"|"pyttsx3"|"edge_tts"|"mock")
            config: 音频配置,None则使用默认配置
        """
        self.config: AudioConfig = config or AudioConfig()

        # 解析模型类型
        try:
            self.config.asr_model = ASRModel(asr_model)
        except ValueError:
            logger.warning(f"未知ASR模型: {asr_model},使用mock模式")
            self.config.asr_model = ASRModel.MOCK

        try:
            self.config.tts_model = TTSModel(tts_model)
        except ValueError:
            logger.warning(f"未知TTS模型: {tts_model},使用mock模式")
            self.config.tts_model = TTSModel.MOCK

        # 状态
        self._state: AudioState = AudioState.IDLE
        self._initialized: bool = False
        self._speaking: bool = False
        self._recording: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()

        # PyAudio对象
        self._pyaudio: Any = None
        self._vad: Any = None

        # 回调
        self._state_callbacks: list[Callable[[AudioState], None]] = []

        # 确保临时目录存在
        Path(self.config.temp_dir).mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> bool:
        """初始化语音交互模块

        初始化PyAudio、VAD和TTS引擎。

        返回:
            bool: 初始化成功返回True
        """
        async with self._lock:
            if self._initialized:
                return True

            try:
                # 初始化PyAudio
                try:
                    import pyaudio
                    self._pyaudio = pyaudio.PyAudio()
                    logger.debug("PyAudio初始化成功")
                except ImportError:
                    logger.warning("PyAudio未安装,录音/播放功能不可用")
                    self._pyaudio = None

                # 初始化VAD(语音活动检测)
                try:
                    import webrtcvad
                    self._vad = webrtcvad.Vad(self.config.vad_aggressiveness)
                    logger.debug("WebRTC VAD初始化成功")
                except ImportError:
                    logger.warning("webrtcvad未安装,VAD功能不可用")
                    self._vad = None

                # 预初始化TTS引擎
                await self._init_tts()

                self._initialized = True
                logger.info(
                    f"语音交互初始化成功: "
                    f"ASR={self.config.asr_model.value}, "
                    f"TTS={self.config.tts_model.value}"
                )
                return True

            except Exception as e:
                logger.error(f"语音交互初始化失败: {e}")
                return False

    async def _init_tts(self) -> bool:
        """初始化TTS引擎(内部方法)

        返回:
            bool: 初始化成功返回True
        """
        if self.config.tts_model == TTSModel.PYTTSX3:
            try:
                import pyttsx3
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty("rate", int(200 * self.config.tts_speed))
                self._tts_engine.setProperty("volume", self.config.tts_volume)
                logger.debug("pyttsx3 TTS引擎初始化成功")
                return True
            except ImportError:
                logger.warning("pyttsx3未安装")
                return False

        return True

    def _update_state(self, new_state: AudioState) -> None:
        """更新音频状态并通知回调(内部方法)

        参数:
            new_state: 新状态
        """
        if self._state != new_state:
            self._state = new_state
            logger.debug(f"音频状态变更: {new_state.value}")

            for cb in self._state_callbacks:
                try:
                    cb(new_state)
                except Exception as e:
                    logger.error(f"状态回调异常: {e}")

    async def listen(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        use_vad: bool = True,
    ) -> str:
        """录制音频并进行语音识别

        录制用户语音,使用配置的ASR模型进行识别,返回识别文本。
        支持VAD自动停止和超时保护。

        参数:
            timeout: 录音超时时间(秒),默认10.0
            use_vad: 是否使用VAD自动检测语音结束

        返回:
            str: 识别文本,空字符串表示识别失败或无语音

        示例:
            >>> text = await voice.listen(timeout=10.0)
            >>> if text:
            ...     print(f"识别结果: {text}")
        """
        if not self._initialized:
            logger.error("语音模块未初始化")
            return ""

        self._update_state(AudioState.RECORDING)
        self._recording = True

        try:
            # 录制音频
            audio_file = await self._record_audio(timeout, use_vad)

            if not audio_file or not os.path.exists(audio_file):
                logger.warning("录音失败或未检测到语音")
                return ""

            # 检查音频时长
            duration = self._get_audio_duration(audio_file)
            if duration < self.MIN_RECORD_SECONDS:
                logger.debug(f"录音太短: {duration:.2f}s < {self.MIN_RECORD_SECONDS}s")
                return ""

            self._update_state(AudioState.RECOGNIZING)

            # 语音识别
            result = await self._asr_recognize(audio_file)

            return result.text

        except Exception as e:
            logger.error(f"语音识别异常: {e}")
            return ""
        finally:
            self._recording = False
            self._update_state(AudioState.IDLE)

    async def _record_audio(
        self,
        timeout: float,
        use_vad: bool = True,
    ) -> str:
        """录制音频到文件(内部方法)

        使用PyAudio录制音频,支持VAD自动停止。

        参数:
            timeout: 最大录音时间
            use_vad: 是否使用VAD

        返回:
            str: 录音文件路径
        """
        audio_file = os.path.join(
            self.config.temp_dir, f"record_{int(time.time())}.wav"
        )

        if self._pyaudio is None:
            # 模拟模式
            await asyncio.sleep(1.0)
            return audio_file

        # 在线程池中执行阻塞的音频录制
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._record_audio_sync, audio_file, timeout, use_vad
        )

    def _record_audio_sync(
        self,
        audio_file: str,
        timeout: float,
        use_vad: bool,
    ) -> str:
        """同步录音(在线程池中执行)

        参数:
            audio_file: 输出文件路径
            timeout: 超时时间
            use_vad: 是否使用VAD

        返回:
            str: 录音文件路径
        """
        import pyaudio

        chunk = self.config.chunk_size
        fmt = pyaudio.paInt16
        channels = self.config.channels
        rate = self.config.sample_rate

        stream = self._pyaudio.open(
            format=fmt,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
        )

        frames: list[bytes] = []
        start_time = time.time()
        silence_start: float | None = None

        try:
            while time.time() - start_time < timeout:
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)

                # VAD检测
                if use_vad and self._vad and len(data) >= 320:
                    is_speech = self._vad.is_speech(data[:320], rate)

                    if not is_speech:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > self.SILENCE_TIMEOUT:
                            logger.debug("VAD检测到静音,停止录音")
                            break
                    else:
                        silence_start = None

                # 最大录音时长限制
                total_duration = len(frames) * chunk / rate
                if total_duration >= self.MAX_RECORD_SECONDS:
                    break
        finally:
            stream.stop_stream()
            stream.close()

        # 保存WAV文件
        with wave.open(audio_file, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(self._pyaudio.get_sample_size(fmt))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))

        return audio_file

    async def _asr_recognize(self, audio_file: str) -> ASRResult:
        """执行语音识别(内部方法)

        根据配置的ASR模型调用对应的识别引擎。

        参数:
            audio_file: 音频文件路径

        返回:
            ASRResult: 识别结果
        """
        start_time = time.time()

        if self.config.asr_model == ASRModel.MOCK:
            # 模拟模式
            await asyncio.sleep(0.5)
            return ASRResult(
                text="模拟语音识别结果",
                confidence=0.95,
                processing_time=time.time() - start_time,
            )

        elif self.config.asr_model == ASRModel.FUNASR:
            return await self._asr_funasr(audio_file)

        elif self.config.asr_model == ASRModel.WHISPER_CPP:
            return await self._asr_whisper_cpp(audio_file)

        elif self.config.asr_model == ASRModel.WHISPER_API:
            return await self._asr_whisper_api(audio_file)

        return ASRResult()

    async def _asr_funasr(self, audio_file: str) -> ASRResult:
        """FunASR语音识别(内部方法)

        通过subprocess调用FunASR CLI进行识别。

        参数:
            audio_file: 音频文件路径

        返回:
            ASRResult: 识别结果
        """
        start_time = time.time()

        try:
            cmd = [
                "python", "-m", "funasr",
                "+model=paraformer-zh",
                "+vad_model=fsmn-vad",
                "+punc_model=ct-punc",
                f"+input={audio_file}",
                "--output_dir", self.config.temp_dir,
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=30.0
            )

            # 解析输出
            output = stdout.decode("utf-8", errors="ignore")

            # 提取识别文本(简化处理)
            text = output.strip().split("\n")[-1] if output else ""

            return ASRResult(
                text=text,
                confidence=0.85,
                processing_time=time.time() - start_time,
            )

        except asyncio.TimeoutError:
            logger.error("FunASR识别超时")
            return ASRResult()
        except FileNotFoundError:
            logger.error("FunASR未安装或不可用")
            return ASRResult(text="[ASR引擎不可用]")
        except Exception as e:
            logger.error(f"FunASR识别异常: {e}")
            return ASRResult()

    async def _asr_whisper_cpp(self, audio_file: str) -> ASRResult:
        """Whisper.cpp语音识别(内部方法)

        通过subprocess调用whisper.cpp进行识别。

        参数:
            audio_file: 音频文件路径

        返回:
            ASRResult: 识别结果
        """
        start_time = time.time()

        try:
            model_path = os.path.join(
                self.config.whisper_model_path, "ggml-base.bin"
            )

            cmd = [
                "./whisper-cli",
                "-m", model_path,
                "-f", audio_file,
                "-l", "zh",
                "--no-timestamps",
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=60.0
            )

            text = stdout.decode("utf-8", errors="ignore").strip()

            return ASRResult(
                text=text,
                confidence=0.80,
                processing_time=time.time() - start_time,
            )

        except asyncio.TimeoutError:
            logger.error("Whisper.cpp识别超时")
            return ASRResult()
        except FileNotFoundError:
            logger.error("whisper-cli未找到")
            return ASRResult(text="[Whisper未安装]")
        except Exception as e:
            logger.error(f"Whisper.cpp识别异常: {e}")
            return ASRResult()

    async def _asr_whisper_api(self, audio_file: str) -> ASRResult:
        """OpenAI Whisper API识别(内部方法)

        调用OpenAI API进行在线语音识别。需要配置API密钥。

        参数:
            audio_file: 音频文件路径

        返回:
            ASRResult: 识别结果
        """
        start_time = time.time()

        try:
            import aiohttp

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                logger.error("未设置OPENAI_API_KEY环境变量")
                return ASRResult(text="[API密钥未配置]")

            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {api_key}"}

            data = aiohttp.FormData()
            data.add_field("model", "whisper-1")
            data.add_field("language", "zh")
            data.add_field("file", open(audio_file, "rb"))

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=headers, data=data, timeout=30
                ) as resp:
                    result = await resp.json()
                    text = result.get("text", "")

            return ASRResult(
                text=text,
                confidence=0.90,
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Whisper API异常: {e}")
            return ASRResult()

    async def speak(
        self,
        text: str,
        speed: float | None = None,
        volume: float | None = None,
    ) -> bool:
        """语音合成并播报

        将文本转换为语音并播放。

        参数:
            text: 要播报的文本
            speed: 语速(0.5-2.0),None使用默认值
            volume: 音量(0.0-1.0),None使用默认值

        返回:
            bool: 播报成功返回True

        示例:
            >>> await voice.speak("您好,有什么可以帮您?")
            >>> await voice.speak("请稍等", speed=0.8)
        """
        if not self._initialized:
            logger.error("语音模块未初始化")
            return False

        if not text:
            return True

        self._update_state(AudioState.SPEAKING)
        self._speaking = True

        try:
            # 语音合成
            tts_result = await self._tts_synthesize(text, speed, volume)

            if not tts_result.audio_file:
                logger.error("语音合成失败")
                return False

            # 播放音频
            await self._play_audio(tts_result.audio_file)

            return True

        except Exception as e:
            logger.error(f"语音播报异常: {e}")
            return False
        finally:
            self._speaking = False
            self._update_state(AudioState.IDLE)

    async def _tts_synthesize(
        self,
        text: str,
        speed: float | None = None,
        volume: float | None = None,
    ) -> TTSResult:
        """语音合成(内部方法)

        根据配置的TTS模型调用对应的合成引擎。

        参数:
            text: 要合成的文本
            speed: 语速
            volume: 音量

        返回:
            TTSResult: 合成结果
        """
        speed = speed or self.config.tts_speed
        volume = volume or self.config.tts_volume

        audio_file = os.path.join(
            self.config.temp_dir, f"tts_{int(time.time())}.wav"
        )

        if self.config.tts_model == TTSModel.MOCK:
            await asyncio.sleep(0.5)
            return TTSResult(audio_file=audio_file, duration=1.0)

        elif self.config.tts_model == TTSModel.PYTTSX3:
            return await self._tts_pyttsx3(text, audio_file, speed, volume)

        elif self.config.tts_model == TTSModel.PADDLESPEECH:
            return await self._tts_paddlespeech(text, audio_file, speed)

        elif self.config.tts_model == TTSModel.EDGE_TTS:
            return await self._tts_edge(text, audio_file, speed)

        return TTSResult()

    async def _tts_pyttsx3(
        self, text: str, output_file: str, speed: float, volume: float
    ) -> TTSResult:
        """pyttsx3语音合成(内部方法)

        参数:
            text: 文本
            output_file: 输出文件
            speed: 语速
            volume: 音量

        返回:
            TTSResult: 合成结果
        """
        start_time = time.time()

        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", int(200 * speed))
            engine.setProperty("volume", volume)

            # pyttsx3不能直接保存到文件,需要变通
            # 使用runAndWait在事件循环中可能阻塞
            loop = asyncio.get_event_loop()

            def _synthesize():
                engine.save_to_file(text, output_file)
                engine.runAndWait()

            await loop.run_in_executor(None, _synthesize)

            duration = self._get_audio_duration(output_file)

            return TTSResult(
                audio_file=output_file,
                duration=duration,
                processing_time=time.time() - start_time,
            )

        except ImportError:
            logger.error("pyttsx3未安装")
            return TTSResult()
        except Exception as e:
            logger.error(f"pyttsx3合成异常: {e}")
            return TTSResult()

    async def _tts_paddlespeech(
        self, text: str, output_file: str, speed: float
    ) -> TTSResult:
        """PaddleSpeech语音合成(内部方法)

        通过subprocess调用paddlespeech CLI。

        参数:
            text: 文本
            output_file: 输出文件
            speed: 语速

        返回:
            TTSResult: 合成结果
        """
        start_time = time.time()

        try:
            cmd = [
                "paddlespeech", "tts",
                "--input", text,
                "--output", output_file,
                "--am", "fastspeech2_csmsc",
                "--voc", "pwgan_csmsc",
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=30.0
            )

            if proc.returncode == 0 and os.path.exists(output_file):
                duration = self._get_audio_duration(output_file)
                return TTSResult(
                    audio_file=output_file,
                    duration=duration,
                    processing_time=time.time() - start_time,
                )
            else:
                logger.error(f"PaddleSpeech合成失败: {stderr.decode()}")
                return TTSResult()

        except asyncio.TimeoutError:
            logger.error("PaddleSpeech合成超时")
            return TTSResult()
        except FileNotFoundError:
            logger.error("paddlespeech命令未找到")
            return TTSResult()
        except Exception as e:
            logger.error(f"PaddleSpeech合成异常: {e}")
            return TTSResult()

    async def _tts_edge(
        self, text: str, output_file: str, speed: float
    ) -> TTSResult:
        """Edge TTS语音合成(内部方法)

        通过edge-tts库调用微软在线TTS服务。

        参数:
            text: 文本
            output_file: 输出文件
            speed: 语速

        返回:
            TTSResult: 合成结果
        """
        start_time = time.time()

        try:
            import edge_tts

            voice = "zh-CN-XiaoxiaoNeural"
            rate = f"{int((speed - 1.0) * 100):+d}%"

            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(output_file)

            duration = self._get_audio_duration(output_file)

            return TTSResult(
                audio_file=output_file,
                duration=duration,
                processing_time=time.time() - start_time,
            )

        except ImportError:
            logger.error("edge-tts未安装")
            return TTSResult()
        except Exception as e:
            logger.error(f"Edge TTS合成异常: {e}")
            return TTSResult()

    async def _play_audio(self, audio_file: str) -> bool:
        """播放音频文件(内部方法)

        参数:
            audio_file: 音频文件路径

        返回:
            bool: 播放成功返回True
        """
        if self._pyaudio is None:
            logger.debug("模拟播放音频")
            duration = self._get_audio_duration(audio_file)
            await asyncio.sleep(duration)
            return True

        try:
            # 使用aplay播放
            proc = await asyncio.create_subprocess_exec(
                "aplay", "-q", audio_file,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=60.0
            )

            return proc.returncode == 0

        except FileNotFoundError:
            # 回退到PyAudio播放
            return await self._play_with_pyaudio(audio_file)
        except Exception as e:
            logger.error(f"音频播放异常: {e}")
            return False

    async def _play_with_pyaudio(self, audio_file: str) -> bool:
        """使用PyAudio播放(内部方法)

        参数:
            audio_file: 音频文件路径

        返回:
            bool: 播放成功返回True
        """
        try:
            import pyaudio

            with wave.open(audio_file, "rb") as wf:
                stream = self._pyaudio.open(
                    format=self._pyaudio.get_format_from_width(
                        wf.getsampwidth()
                    ),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                )

                chunk = 1024
                data = wf.readframes(chunk)

                while data:
                    stream.write(data)
                    data = wf.readframes(chunk)

                stream.stop_stream()
                stream.close()

            return True

        except Exception as e:
            logger.error(f"PyAudio播放异常: {e}")
            return False

    def _get_audio_duration(self, audio_file: str) -> float:
        """获取音频文件时长(内部方法)

        参数:
            audio_file: 音频文件路径

        返回:
            float: 时长(秒)
        """
        try:
            with wave.open(audio_file, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
        except Exception:
            return 0.0

    def is_speaking(self) -> bool:
        """检查是否正在播报

        返回:
            bool: 正在播报返回True
        """
        return self._speaking

    def is_recording(self) -> bool:
        """检查是否正在录音

        返回:
            bool: 正在录音返回True
        """
        return self._recording

    @property
    def state(self) -> AudioState:
        """当前音频状态"""
        return self._state

    def register_state_callback(
        self, callback: Callable[[AudioState], None]
    ) -> None:
        """注册状态变更回调

        参数:
            callback: 状态回调函数
        """
        if callback not in self._state_callbacks:
            self._state_callbacks.append(callback)

    async def shutdown(self) -> None:
        """关闭语音交互模块"""
        async with self._lock:
            try:
                if self._pyaudio:
                    self._pyaudio.terminate()
                    self._pyaudio = None

                self._initialized = False
                logger.info("语音交互模块已关闭")

            except Exception as e:
                logger.error(f"关闭语音模块异常: {e}")

    def __repr__(self) -> str:
        return (
            f"VoiceInterface(asr={self.config.asr_model.value}, "
            f"tts={self.config.tts_model.value}, "
            f"state={self._state.value})"
        )

    async def __aenter__(self) -> VoiceInterface:
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.shutdown()
