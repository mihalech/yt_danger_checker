from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import yt_dlp
import yt_dlp_ejs  # noqa: F401
from scipy.io import wavfile


try:
    import tensorflow_hub as hub
except ImportError:
    hub = None


try:
    from nudenet import NudeDetector
except ImportError:
    NudeDetector = None


try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# =========================
# Настройки
# =========================

COOKIES_PATH = Path(__file__).resolve().parent / "cookies.txt"

MAX_SIZE_MB = 80
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

# Для анализа опасности 360p/480p достаточно.
MIN_HEIGHT = 144
MAX_HEIGHT = 480
TARGET_HEIGHT = 360

AUDIO_ABR_KBPS = 64
MAX_FINAL_SIZE_MULTIPLIER = 1.35

MAX_NSFW_FRAMES = 80
MAX_YOLO_FRAMES = 80


# =========================
# Результат
# =========================

@dataclass
class DangerResult:
    url: str
    video_id: str
    title: str
    channel: str
    duration_seconds: int

    title_score: float
    visual_nsfw_score: float
    visual_violence_score: float
    speech_score: float
    audio_event_score: float

    danger_score: float
    label: str

    original_speech: str = ""
    translated_speech: str = ""
    speech_reasons: list[str] = field(default_factory=list)


# =========================
# Основной анализатор
# =========================

class YouTubeDangerAnalyzer:
    def __init__(self) -> None:
        self.nude_detector = self._init_nude_detector()
        self.whisper_model = self._init_whisper()
        self.yolo_model = self._init_yolo()
        self.yamnet_model = self._init_yamnet()
        self.yamnet_class_names = (
            self._load_yamnet_class_names()
            if self.yamnet_model is not None
            else []
        )

        self.danger_audio_keywords: dict[str, float] = {
            "gunshot": 0.45,
            "gunfire": 0.45,
            "explosion": 0.50,
            "scream": 0.30,
            "screaming": 0.30,
            "siren": 0.20,
            "glass": 0.20,
            "breaking": 0.20,
            "fire": 0.20,
            "alarm": 0.15,
        }

        self.danger_keywords: dict[str, float] = {
            "pistol": 0.12,
            "gun": 0.12,
            "rifle": 0.14,
            "shotgun": 0.15,
            "knife": 0.10,
            "blood": 0.12,
            "murder": 0.22,
            "kill": 0.14,
            "killed": 0.14,
            "killing": 0.14,
            "death": 0.12,
            "dead": 0.10,
            "corpse": 0.22,
            "suicide": 0.35,
            "fight": 0.08,
            "beating": 0.14,
            "abuse": 0.16,
            "violence": 0.18,
            "torture": 0.30,
            "gore": 0.35,
            "explosion": 0.14,
            "bomb": 0.25,
            "grenade": 0.25,
            "war": 0.10,
            "terror": 0.30,
            "terrorist": 0.35,

            "пистолет": 0.12,
            "ружье": 0.14,
            "ружьё": 0.14,
            "винтовка": 0.14,
            "нож": 0.10,
            "кровь": 0.12,
            "убийство": 0.22,
            "убил": 0.14,
            "убили": 0.14,
            "убивать": 0.14,
            "смерть": 0.12,
            "мертв": 0.10,
            "мёртв": 0.10,
            "труп": 0.22,
            "суицид": 0.35,
            "самоубийство": 0.35,
            "драка": 0.08,
            "избиение": 0.14,
            "насилие": 0.18,
            "пытки": 0.30,
            "жесть": 0.06,
            "взрыв": 0.14,
            "бомба": 0.25,
            "граната": 0.25,
            "война": 0.10,
            "террор": 0.30,
            "террорист": 0.35,
        }

    # -------------------------
    # Инициализация моделей
    # -------------------------

    def _init_nude_detector(self):
        if NudeDetector is None:
            print("[NSFW] nudenet не установлен, NSFW-анализ отключен")
            return None

        try:
            return NudeDetector()
        except Exception as exc:
            print(f"[NSFW] Не удалось загрузить NudeDetector: {exc}")
            return None

    def _init_whisper(self):
        if WhisperModel is None:
            print("[Whisper] faster-whisper не установлен, анализ речи отключен")
            return None

        try:
            return WhisperModel("base", device="cpu", compute_type="int8")
        except Exception as exc:
            print(f"[Whisper] Не удалось загрузить WhisperModel: {exc}")
            return None

    def _init_yolo(self):
        if YOLO is None:
            print("[YOLO] ultralytics не установлен, анализ объектов отключен")
            return None

        try:
            return YOLO("yolo11n.pt")
        except Exception as exc:
            print(f"[YOLO] Не удалось загрузить YOLO: {exc}")
            return None

    def _init_yamnet(self):
        if hub is None:
            print("[YAMNet] tensorflow_hub не установлен, анализ звуков отключен")
            return None

        try:
            return hub.load("https://tfhub.dev/google/yamnet/1")
        except Exception as exc:
            print(f"[YAMNet] Не удалось загрузить YAMNet: {exc}")
            return None

    # -------------------------
    # Главный метод
    # -------------------------

    def analyze(self, video_url: str) -> DangerResult:
        if not self._is_youtube_url(video_url):
            raise ValueError("Нужна именно ссылка на YouTube-видео")

        normalized_url = self._normalize_youtube_url(video_url)
        metadata = self._get_youtube_metadata(normalized_url)

        print("\n=== Видео ===")
        print(f"Название: {metadata['title']}")
        print(f"Канал: {metadata['channel']}")
        print(f"Длительность: {metadata['duration_seconds']} сек.")
        print(f"URL: {normalized_url}")

        title_score = self._analyze_text_keywords(metadata["title"])

        visual_nsfw_score = 0.0
        visual_violence_score = 0.0
        speech_score = 0.0
        audio_event_score = 0.0

        original_speech = ""
        translated_speech = ""
        speech_reasons: list[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            video_path = self._download_video(normalized_url, tmp_path)

            if video_path is None:
                print("[download] Видео не скачалось, считаю только title_score")
            else:
                frames_dir = tmp_path / "frames"
                frames_dir.mkdir(exist_ok=True)

                fps_value = self._choose_sampling_fps(metadata["duration_seconds"])

                self._extract_frames(
                    video_path=video_path,
                    frames_dir=frames_dir,
                    fps_value=fps_value,
                )

                visual_nsfw_score = self._analyze_nsfw_frames(frames_dir)
                visual_violence_score = self._analyze_visual_violence_frames(frames_dir)

                audio_path = tmp_path / "audio.wav"

                try:
                    self._extract_audio(video_path, audio_path)
                    audio_event_score = self._analyze_audio_events(audio_path)

                    original_speech, translated_speech = self._transcribe_audio(audio_path)

                    speech_score, speech_reasons = self._analyze_text_keywords_with_reasons(
                        translated_speech
                    )

                except Exception as exc:
                    print(f"[audio] Не удалось проанализировать звук/речь: {exc}")

        danger_score = max(
            title_score,
            visual_nsfw_score,
            visual_violence_score,
            speech_score,
            audio_event_score,
        )

        if danger_score >= 0.7:
            label = "18+"
        elif danger_score >= 0.3:
            label = "suspicious"
        else:
            label = "safe"

        return DangerResult(
            url=normalized_url,
            video_id=metadata["video_id"],
            title=metadata["title"],
            channel=metadata["channel"],
            duration_seconds=metadata["duration_seconds"],
            title_score=round(title_score, 3),
            visual_nsfw_score=round(visual_nsfw_score, 3),
            visual_violence_score=round(visual_violence_score, 3),
            speech_score=round(speech_score, 3),
            audio_event_score=round(audio_event_score, 3),
            danger_score=round(danger_score, 3),
            label=label,
            original_speech=original_speech,
            translated_speech=translated_speech,
            speech_reasons=speech_reasons,
        )

    # -------------------------
    # YouTube URL / metadata
    # -------------------------

    def _is_youtube_url(self, url: str) -> bool:
        parsed = urlparse(url.strip())
        host = parsed.netloc.lower().replace("www.", "")

        return host in {
            "youtube.com",
            "m.youtube.com",
            "music.youtube.com",
            "youtu.be",
            "youtube-nocookie.com",
        }

    def _normalize_youtube_url(self, url: str) -> str:
        url = url.strip()
        parsed = urlparse(url)
        host = parsed.netloc.lower().replace("www.", "")

        if host == "youtu.be":
            video_id = parsed.path.strip("/").split("/")[0]
            if video_id:
                return f"https://www.youtube.com/watch?v={video_id}"

        if parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/shorts/", 1)[1].split("/", 1)[0]
            if video_id:
                return f"https://www.youtube.com/watch?v={video_id}"

        if parsed.path.startswith("/embed/"):
            video_id = parsed.path.split("/embed/", 1)[1].split("/", 1)[0]
            if video_id:
                return f"https://www.youtube.com/watch?v={video_id}"

        query = parse_qs(parsed.query)
        video_id = query.get("v", [""])[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"

        return url

    def _base_ytdlp_opts(self) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "quiet": False,
            "no_warnings": False,
            "noplaylist": True,
            "remote_components": {"ejs": "github"},
            "js_runtimes": {"deno": {}},
        }

        if COOKIES_PATH.exists():
            opts["cookiefile"] = str(COOKIES_PATH)
            print(f"[cookies] Использую cookies.txt: {COOKIES_PATH}")
        else:
            print(f"[cookies] cookies.txt не найден: {COOKIES_PATH}")

        return opts

    def _get_youtube_metadata(self, video_url: str) -> dict[str, Any]:
        opts = {
            **self._base_ytdlp_opts(),
            "quiet": True,
            "skip_download": True,
        }

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

        if not info:
            raise RuntimeError("yt-dlp не смог получить метаданные видео")

        return {
            "video_id": info.get("id") or "",
            "title": info.get("title") or "",
            "channel": info.get("channel") or info.get("uploader") or "",
            "duration_seconds": int(info.get("duration") or 0),
        }

    # -------------------------
    # Скачивание видео
    # -------------------------

    def _download_video(self, video_url: str, output_dir: Path) -> Path | None:
        output_template = str(output_dir / "%(id)s.%(ext)s")

        opts = {
            **self._base_ytdlp_opts(),
            "outtmpl": output_template,
            "merge_output_format": "mp4",
            "postprocessor_args": {
                "ffmpeg": ["-movflags", "faststart"],
            },
        }

        def estimate_format_size(fmt: dict[str, Any], duration: int) -> int | None:
            if fmt.get("filesize"):
                return int(fmt["filesize"])

            if fmt.get("filesize_approx"):
                return int(fmt["filesize_approx"])

            bitrate = fmt.get("tbr") or fmt.get("vbr")
            if bitrate and duration > 0:
                return int((float(bitrate) * 1000 / 8) * duration)

            return None

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(video_url, download=False)

                if not info:
                    print("[download] Не удалось получить информацию о видео")
                    return None

                duration = int(info.get("duration") or 0)
                formats = info.get("formats", [])

                video_formats: list[dict[str, Any]] = []

                for fmt in formats:
                    height = int(fmt.get("height") or 0)
                    vcodec = fmt.get("vcodec")

                    if vcodec == "none":
                        continue

                    if height < MIN_HEIGHT or height > MAX_HEIGHT:
                        continue

                    if fmt.get("protocol") in {"mhtml"}:
                        continue

                    estimated_size = estimate_format_size(fmt, duration)

                    if fmt.get("acodec") == "none" and duration > 0:
                        audio_size = int((AUDIO_ABR_KBPS * 1000 / 8) * duration)
                        estimated_size = (estimated_size or 0) + audio_size

                    fmt["_estimated_size"] = estimated_size
                    video_formats.append(fmt)

                if not video_formats:
                    print(f"[download] Нет форматов в диапазоне {MIN_HEIGHT}p-{MAX_HEIGHT}p")
                    return None

                def sort_key(fmt: dict[str, Any]) -> tuple[int, int, int, int]:
                    height = int(fmt.get("height") or 0)
                    estimated_size = fmt.get("_estimated_size")

                    size_priority = (
                        0
                        if estimated_size is not None and estimated_size <= MAX_SIZE_BYTES
                        else 1
                    )

                    height_distance = abs(height - TARGET_HEIGHT)
                    size_value = int(estimated_size) if estimated_size is not None else 10**18
                    bitrate = int(fmt.get("tbr") or fmt.get("vbr") or 10**9)

                    return size_priority, height_distance, size_value, bitrate

                video_formats.sort(key=sort_key)

                print("\n[download] Подходящие форматы:")
                for fmt in video_formats[:8]:
                    size = fmt.get("_estimated_size")
                    size_text = (
                        f"~{size / (1024 * 1024):.1f} MB"
                        if size
                        else "размер неизвестен"
                    )

                    print(
                        f"  {fmt.get('format_id')} | "
                        f"{fmt.get('height')}p | "
                        f"{fmt.get('ext')} | "
                        f"{size_text}"
                    )

                selected_fmt = video_formats[0]
                selected_size = selected_fmt.get("_estimated_size")
                selected_mb = selected_size / (1024 * 1024) if selected_size else 0.0

                if selected_fmt.get("acodec") and selected_fmt.get("acodec") != "none":
                    format_selector = str(selected_fmt["format_id"])
                else:
                    format_selector = (
                        f"{selected_fmt['format_id']}+"
                        f"bestaudio[abr<={AUDIO_ABR_KBPS}]/"
                        "bestaudio[abr<=96]/worstaudio"
                    )

                ydl.params["format"] = format_selector

                print(
                    f"\n[download] Скачиваю для анализа: "
                    f"format={format_selector}, "
                    f"height={selected_fmt.get('height')}p, "
                    f"estimate=~{selected_mb:.1f} MB"
                )

                downloaded = ydl.extract_info(video_url, download=True)

                if not downloaded:
                    print("[download] Скачивание не удалось")
                    return None

                video_files = sorted(
                    [
                        p
                        for p in output_dir.iterdir()
                        if p.is_file()
                        and p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
                    ],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )

                if not video_files:
                    prepared_path = Path(ydl.prepare_filename(downloaded))
                    if prepared_path.exists():
                        video_files = [prepared_path]

                if not video_files:
                    print("[download] Файл после скачивания не найден")
                    return None

                video_path = video_files[0]
                actual_size = video_path.stat().st_size
                actual_mb = actual_size / (1024 * 1024)

                print(f"[download] Скачано: {actual_mb:.1f} MB")

                max_final_size = int(MAX_SIZE_BYTES * MAX_FINAL_SIZE_MULTIPLIER)

                if actual_size > max_final_size:
                    print(
                        f"[download] Файл слишком большой: {actual_mb:.1f} MB. "
                        f"Лимит: {max_final_size / (1024 * 1024):.1f} MB. Удаляю."
                    )

                    try:
                        video_path.unlink()
                    except Exception as exc:
                        print(f"[download] Не удалось удалить файл: {exc}")

                    return None

                return video_path

        except Exception as exc:
            print(f"[download] Не удалось скачать видео: {exc}")
            return None

    # -------------------------
    # Кадры
    # -------------------------

    def _choose_sampling_fps(self, duration_seconds: int | None) -> float:
        if duration_seconds is None or duration_seconds <= 0:
            return 0.5

        if duration_seconds <= 180:
            return 1.0

        if duration_seconds <= 600:
            return 0.5

        return 0.2

    def _extract_frames(self, video_path: Path, frames_dir: Path, fps_value: float) -> None:
        output_pattern = str(frames_dir / "frame_%06d.jpg")

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps_value}",
            "-q:v",
            "3",
            output_pattern,
        ]

        print(f"[frames] Извлекаю кадры: fps={fps_value}")
        subprocess.run(cmd, check=True)

    def _sample_frame_paths(self, frames_dir: Path, max_frames: int) -> list[Path]:
        frame_paths = sorted(frames_dir.glob("*.jpg"))

        if not frame_paths:
            return []

        if len(frame_paths) <= max_frames:
            return frame_paths

        step = max(1, len(frame_paths) // max_frames)
        return frame_paths[::step][:max_frames]

    # -------------------------
    # NSFW
    # -------------------------

    def _analyze_nsfw_frames(self, frames_dir: Path) -> float:
        if self.nude_detector is None:
            return 0.0

        frame_paths = self._sample_frame_paths(frames_dir, MAX_NSFW_FRAMES)

        if not frame_paths:
            return 0.0

        print(f"[NSFW] Анализирую кадров: {len(frame_paths)}")

        scores: list[float] = []

        for frame_path in frame_paths:
            try:
                detections = self.nude_detector.detect(str(frame_path))
            except Exception:
                continue

            frame_score = 0.0

            for det in detections:
                class_name = str(det.get("class", "")).lower()
                score = float(det.get("score", 0.0))

                if "exposed" in class_name and score > 0.6:
                    frame_score = max(frame_score, score)

            scores.append(frame_score)

        if not scores:
            return 0.0

        scores.sort(reverse=True)
        top_scores = scores[: min(5, len(scores))]

        return sum(top_scores) / len(top_scores)

    # -------------------------
    # YOLO / насилие по объектам
    # -------------------------

    def _analyze_visual_violence_frames(self, frames_dir: Path) -> float:
        if self.yolo_model is None:
            return 0.0

        danger_classes = {
            "knife": 0.35,
            "gun": 0.45,
            "pistol": 0.45,
            "rifle": 0.50,
            "weapon": 0.40,
            "blood": 0.45,
        }

        frame_paths = self._sample_frame_paths(frames_dir, MAX_YOLO_FRAMES)

        if not frame_paths:
            return 0.0

        print(f"[YOLO] Анализирую кадров: {len(frame_paths)}")

        frame_scores: list[float] = []

        for frame_path in frame_paths:
            try:
                results = self.yolo_model.predict(
                    source=str(frame_path),
                    conf=0.35,
                    verbose=False,
                )
            except Exception:
                continue

            frame_score = 0.0

            for result in results:
                names = result.names

                for box in result.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = str(names[class_id]).lower()

                    for danger_name, weight in danger_classes.items():
                        if danger_name in class_name:
                            frame_score = max(frame_score, conf * weight)

            frame_scores.append(frame_score)

        if not frame_scores:
            return 0.0

        frame_scores.sort(reverse=True)
        top_scores = frame_scores[: min(5, len(frame_scores))]

        return min(sum(top_scores), 1.0)

    # -------------------------
    # Аудио
    # -------------------------

    def _extract_audio(self, video_path: Path, output_audio_path: Path) -> None:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_audio_path),
        ]

        print("[audio] Извлекаю аудио 16000 Hz mono")
        subprocess.run(cmd, check=True)

    def _load_yamnet_class_names(self) -> list[str]:
        if self.yamnet_model is None:
            return []

        class_map_path = self.yamnet_model.class_map_path().numpy().decode("utf-8")
        class_names: list[str] = []

        with open(class_map_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                class_names.append(row["display_name"])

        return class_names

    def _analyze_audio_events(self, audio_path: Path) -> float:
        if self.yamnet_model is None:
            return 0.0

        print("[YAMNet] Анализирую опасные звуки")

        sample_rate, wav_data = wavfile.read(str(audio_path))

        if sample_rate != 16000:
            print(f"[YAMNet] Неверный sample_rate: {sample_rate}, ожидалось 16000")
            return 0.0

        if wav_data.ndim > 1:
            wav_data = np.mean(wav_data, axis=1)

        waveform = wav_data.astype(np.float32)

        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))

        scores, embeddings, spectrogram = self.yamnet_model(waveform)
        scores_np = scores.numpy()

        class_scores = np.max(scores_np, axis=0)

        matched_scores: list[float] = []

        for class_index, class_score in enumerate(class_scores):
            class_name = self.yamnet_class_names[class_index].lower()

            for keyword, weight in self.danger_audio_keywords.items():
                if keyword in class_name:
                    matched_scores.append(float(class_score) * weight)

        if not matched_scores:
            return 0.0

        matched_scores.sort(reverse=True)
        top_scores = matched_scores[: min(5, len(matched_scores))]

        return min(sum(top_scores), 1.0)

    # -------------------------
    # Речь
    # -------------------------

    def _transcribe_audio(self, audio_path: Path) -> tuple[str, str]:
        if self.whisper_model is None:
            return "", ""

        print("[Whisper] Распознаю речь на исходном языке")

        original_segments, original_info = self.whisper_model.transcribe(
            str(audio_path),
            task="transcribe",
        )

        original_parts: list[str] = []

        for segment in original_segments:
            original_parts.append(segment.text.strip())

        original_text = " ".join(original_parts).strip()

        print("[Whisper] Перевожу речь на английский")

        translated_segments, translated_info = self.whisper_model.transcribe(
            str(audio_path),
            task="translate",
        )

        translated_parts: list[str] = []

        for segment in translated_segments:
            translated_parts.append(segment.text.strip())

        translated_text = " ".join(translated_parts).strip()

        return original_text, translated_text

    # -------------------------
    # Текстовые ключевые слова
    # -------------------------

    def _analyze_text_keywords(self, text: str) -> float:
        score, _ = self._analyze_text_keywords_with_reasons(text)
        return score

    def _analyze_text_keywords_with_reasons(self, text: str) -> tuple[float, list[str]]:
        if not text:
            return 0.0, []

        text_lower = text.lower()
        text_clean = re.sub(r"[^a-zа-яё0-9 ]", " ", text_lower)
        words = text_clean.split()

        total_score = 0.0
        matched_keywords: set[str] = set()

        for keyword, weight in self.danger_keywords.items():
            for word in words:
                if keyword in word and keyword not in matched_keywords:
                    total_score += weight
                    matched_keywords.add(keyword)

        return min(total_score, 1.0), sorted(matched_keywords)


# =========================
# Вывод
# =========================

def print_report(result: DangerResult) -> None:
    print("\n=== DANGER MODERATION RESULT ===")
    print(f"Видео: {result.title}")
    print(f"Канал: {result.channel}")
    print(f"URL: {result.url}")
    print(f"Длительность: {result.duration_seconds} сек.")

    print("\n--- Оценки опасности ---")
    print(f"Категория: {result.label}")
    print(f"Общий рейтинг опасности: {result.danger_score}")
    print(f"Рейтинг опасности названия: {result.title_score}")
    print(f"Рейтинг NSFW по кадрам: {result.visual_nsfw_score}")
    print(f"Рейтинг насилия по видеоряду: {result.visual_violence_score}")
    print(f"Рейтинг опасности речи: {result.speech_score}")
    print(f"Рейтинг опасных звуков: {result.audio_event_score}")

    if result.speech_reasons:
        print(f"Опасные слова в речи: {', '.join(result.speech_reasons)}")
    else:
        print("Опасные слова в речи: не найдены")

    print("\n--- Речь ---")

    if result.original_speech:
        print(f"Оригинал: {result.original_speech[:1000]}")
    else:
        print("Оригинал: речь не обнаружена")

    if result.translated_speech:
        print(f"Перевод EN: {result.translated_speech[:1000]}")
    else:
        print("Перевод EN: не получен")


# =========================
# CLI
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Один файл для анализа опасности YouTube-видео"
    )

    parser.add_argument("url", help="Ссылка на YouTube-видео")

    parser.add_argument(
        "--json",
        action="store_true",
        help="Вывести результат в JSON",
    )

    args = parser.parse_args()

    try:
        analyzer = YouTubeDangerAnalyzer()
        result = analyzer.analyze(args.url)

        if args.json:
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        else:
            print_report(result)

    except KeyboardInterrupt:
        print("\nОстановлено пользователем")
        sys.exit(130)

    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()