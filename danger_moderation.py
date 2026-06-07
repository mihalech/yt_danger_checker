from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yt_dlp
import yt_dlp_ejs
import re

import csv
import numpy as np
from scipy.io import wavfile

from ultralytics import YOLO

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

# Для модерации не нужно скачивать 1080p/1440p.
# 360p обычно хватает для NSFW/YOLO-проверок, а размер файла становится в разы меньше.
MAX_SIZE_MB = 80
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024
MAX_HEIGHT = 480
MIN_HEIGHT = 144
TARGET_HEIGHT = 360
MAX_FINAL_SIZE_MULTIPLIER = 1.35
AUDIO_ABR_KBPS = 64

@dataclass
class DangerResult:
    title_score: float
    visual_nsfw_score: float
    visual_violence_score: float
    speech_score: float
    audio_event_score: float
    danger_score: float
    label: str
    original_speech: str = ""
    translated_speech: str = ""
    speech_reasons: list[str] = None


class VideoDangerAnalyzer:
    def __init__(self) -> None:
        self.nude_detector = NudeDetector() if NudeDetector is not None else None

        self.whisper_model = (
            WhisperModel("base", device="cpu", compute_type="int8")
            if WhisperModel is not None
            else None
        )
        self.yolo_model = YOLO("yolo11n.pt") if YOLO is not None else None
        self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1") if hub is not None else None
        self.yamnet_class_names = self._load_yamnet_class_names() if self.yamnet_model is not None else []

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

            # Русские варианты
            "пистолет": 0.12,
            "ружье": 0.14,
            "винтовка": 0.14,
            "нож": 0.10,
            "кровь": 0.12,
            "убийство": 0.22,
            "убил": 0.14,
            "убили": 0.14,
            "смерть": 0.12,
            "труп": 0.22,
            "суицид": 0.35,
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

    def analyze_video_url(
        self,
        video_url: str,
        title: str = "",
        duration_seconds: int | None = None,
    ) -> DangerResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            video_path = self._download_video(video_url, tmp_path)

            title_score = self._analyze_title_keywords(title)

            visual_nsfw_score = 0.0
            visual_violence_score = 0.0
            speech_score = 0.0
            audio_event_score = 0.0

            original_speech = ""
            translated_speech = ""
            speech_reasons: list[str] = []
            if video_path is not None:
                frames_dir = tmp_path / "frames"
                frames_dir.mkdir(exist_ok=True)

                fps_value = self._choose_sampling_fps(duration_seconds)
                self._extract_frames(video_path, frames_dir, fps_value)

                visual_nsfw_score = self._analyze_nsfw_frames(frames_dir)

                # Пока заглушка. Позже сюда подключим VideoMAE / MoViNet.
                visual_violence_score = self._analyze_visual_violence_frames(frames_dir)

                # Пока заглушки
                audio_path = tmp_path / "audio.wav"

                try:
                    self._extract_audio(video_path, audio_path)
                    audio_event_score = self._analyze_audio_events(audio_path)
                    original_speech, translated_speech = self._transcribe_audio(audio_path)

                    print("\n[Речь в видео — оригинал]")
                    print(original_speech if original_speech else "Речь не распознана.")

                    print("\n[Речь в видео — перевод на английский]")
                    print(translated_speech if translated_speech else "Перевод не получен.")

                    speech_score, speech_reasons = self._analyze_text_keywords_with_reasons(translated_speech)

                except Exception as e:
                    print(f"Не удалось проанализировать речь: {e}")
                    speech_score = 0.0

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

    def _analyze_title_keywords(self, title: str) -> float:
        if not title:
            return 0.0

        title_lower = title.lower()
        title_clean = re.sub(r"[^a-zа-я0-9 ]", " ", title_lower)
        words = title_clean.split()

        total_score = 0.0
        matched_keywords: set[str] = set()

        for keyword, weight in self.danger_keywords.items():
            for word in words:
                if keyword in word:
                    if keyword not in matched_keywords:
                        total_score += weight
                        matched_keywords.add(keyword)

        return min(total_score, 1.0)

    def _analyze_text_keywords_with_reasons(self, text: str) -> tuple[float, list[str]]:
        if not text:
            return 0.0, []

        text_lower = text.lower()
        text_clean = re.sub(r"[^a-zа-я0-9 ]", " ", text_lower)
        words = text_clean.split()

        total_score = 0.0
        matched_keywords: set[str] = set()

        for keyword, weight in self.danger_keywords.items():
            for word in words:
                if keyword in word and keyword not in matched_keywords:
                    total_score += weight
                    matched_keywords.add(keyword)

        return min(total_score, 1.0), sorted(matched_keywords)

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
            print("YAMNet не установлен")
            return 0.0

        sample_rate, wav_data = wavfile.read(str(audio_path))

        if sample_rate != 16000:
            print(f"Неверный sample_rate: {sample_rate}, ожидалось 16000")
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

    def _analyze_visual_violence_frames(self, frames_dir: Path) -> float:
        if self.yolo_model is None:
            print("YOLO не установлен")
            return 0.0

        danger_classes = {
            "knife": 0.35,
            "gun": 0.45,
            "pistol": 0.45,
            "rifle": 0.50,
            "weapon": 0.40,
            "blood": 0.45,
        }

        frame_scores: list[float] = []

        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            return 0.0

        max_frames = 40
        step = max(1, len(frame_paths) // max_frames)
        sampled_frames = frame_paths[::step][:max_frames]

        for frame_path in sampled_frames:
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

    def _download_video(self, video_url: str, output_dir: Path) -> Path | None:
        output_template = str(output_dir / "%(id)s.%(ext)s")
        cookies_path = Path(__file__).resolve().parent / "cookies.txt"

        base_opts = {
            "quiet": False,
            "no_warnings": False,
            "verbose": False,
            "outtmpl": output_template,
            "noplaylist": True,

            "remote_components": {"ejs": "github"},
            "js_runtimes": {"deno": {}},

            "merge_output_format": "mp4",
            "postprocessor_args": {
                "ffmpeg": ["-movflags", "faststart"],
            },
        }

        def estimate_format_size(fmt: dict, duration: int) -> int | None:
            """
            Примерная оценка размера конкретного формата.
            YouTube не всегда отдает filesize, поэтому fallback — расчет через bitrate.
            """
            if fmt.get("filesize"):
                return int(fmt["filesize"])

            if fmt.get("filesize_approx"):
                return int(fmt["filesize_approx"])

            bitrate = fmt.get("tbr") or fmt.get("vbr")
            if bitrate and duration > 0:
                return int((float(bitrate) * 1000 / 8) * duration)

            return None

        def download_attempt(use_cookies: bool) -> Path | None:
            current_opts = dict(base_opts)

            if use_cookies:
                if not cookies_path.exists():
                    print(f"Файл cookies не найден: {cookies_path}")
                    return None
                current_opts["cookiefile"] = str(cookies_path)

            try:
                with yt_dlp.YoutubeDL(current_opts) as ydl:
                    info = ydl.extract_info(video_url, download=False)

                    if not info:
                        print("Не удалось получить информацию о видео")
                        return None

                    duration = int(info.get("duration") or 0)
                    formats = info.get("formats", [])

                    video_formats: list[dict] = []

                    for fmt in formats:
                        height = fmt.get("height") or 0
                        vcodec = fmt.get("vcodec")

                        if vcodec == "none":
                            continue

                        if not height or height < MIN_HEIGHT or height > MAX_HEIGHT:
                            continue

                        # Отсекаем странные служебные форматы/storyboards.
                        if fmt.get("protocol") in {"mhtml"}:
                            continue

                        estimated_size = estimate_format_size(fmt, duration)

                        # Если формат без звука, добавляем примерный размер дешевой аудиодорожки.
                        if fmt.get("acodec") == "none" and duration > 0:
                            estimated_size = (estimated_size or 0) + int((AUDIO_ABR_KBPS * 1000 / 8) * duration)

                        # Если yt-dlp вообще не дал данных по размеру, оставляем формат как запасной,
                        # но при сортировке он будет менее приоритетным.
                        fmt["_estimated_size"] = estimated_size
                        video_formats.append(fmt)

                    if not video_formats:
                        print(f"Нет форматов в диапазоне {MIN_HEIGHT}p-{MAX_HEIGHT}p")
                        return None

                    def sort_key(fmt: dict) -> tuple[int, int, int, int]:
                        height = int(fmt.get("height") or 0)
                        estimated_size = fmt.get("_estimated_size")

                        # 1) Сначала форматы, которые точно влезают в лимит.
                        size_priority = 0 if estimated_size is not None and estimated_size <= MAX_SIZE_BYTES else 1

                        # 2) Предпочитаем около 360p: нормально для анализа, но не слишком жирно.
                        height_distance = abs(height - TARGET_HEIGHT)

                        # 3) Если размеры известны, выбираем меньший.
                        size_value = int(estimated_size) if estimated_size is not None else 10**18

                        # 4) При равенстве берем более низкий bitrate.
                        bitrate = int(fmt.get("tbr") or fmt.get("vbr") or 10**9)

                        return (size_priority, height_distance, size_value, bitrate)

                    video_formats.sort(key=sort_key)

                    selected_fmt = video_formats[0]
                    selected_size = selected_fmt.get("_estimated_size")
                    selected_mb = selected_size / (1024 * 1024) if selected_size else 0.0

                    print("\n[Доступные форматы для модерации]")
                    for fmt in video_formats[:8]:
                        size = fmt.get("_estimated_size")
                        size_text = f"~{size / (1024 * 1024):.1f} MB" if size else "размер неизвестен"
                        print(
                            f"Формат {fmt.get('format_id')} "
                            f"({fmt.get('height')}p, {fmt.get('ext')}): {size_text}"
                        )

                    if selected_size and selected_size > MAX_SIZE_BYTES:
                        print(
                            f"Предупреждение: даже самый подходящий формат примерно "
                            f"{selected_mb:.1f} MB, это выше лимита {MAX_SIZE_MB} MB. "
                            f"Пробую скачать, но после скачивания размер всё равно проверю."
                        )

                    # Если выбранный формат уже содержит звук, не добавляем отдельную аудиодорожку.
                    if selected_fmt.get("acodec") and selected_fmt.get("acodec") != "none":
                        format_selector = str(selected_fmt["format_id"])
                    else:
                        # Берем дешевую аудиодорожку, потому что для Whisper/YAMNet качество 16000 Hz mono
                        # всё равно будет пережато через ffmpeg.
                        format_selector = (
                            f"{selected_fmt['format_id']}+"
                            f"bestaudio[abr<={AUDIO_ABR_KBPS}]/"
                            "bestaudio[abr<=96]/worstaudio"
                        )

                    ydl.params["format"] = format_selector

                    print(
                        f"\nСкачиваю низкое качество для анализа: "
                        f"format={format_selector}, height={selected_fmt.get('height')}p, "
                        f"estimate=~{selected_mb:.1f} MB"
                    )

                    downloaded = ydl.extract_info(video_url, download=True)

                    if not downloaded:
                        print("Скачивание не удалось")
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
                        print("Файл после скачивания не найден")
                        return None

                    video_path = video_files[0]
                    actual_size = video_path.stat().st_size
                    actual_mb = actual_size / (1024 * 1024)

                    print(f"Скачано: {actual_mb:.1f} MB")

                    max_final_size = int(MAX_SIZE_BYTES * MAX_FINAL_SIZE_MULTIPLIER)
                    if actual_size > max_final_size:
                        print(
                            f"Файл всё равно слишком большой: {actual_mb:.1f} MB. "
                            f"Лимит: {max_final_size / (1024 * 1024):.1f} MB. Удаляю."
                        )
                        try:
                            video_path.unlink()
                        except Exception as delete_error:
                            print(f"Не удалось удалить файл: {delete_error}")
                        return None

                    return video_path

            except Exception as e:
                print(f"Не удалось скачать видео: {e}")
                return None

        if cookies_path.exists():
            result = download_attempt(use_cookies=True)
            if result is not None:
                return result

        return download_attempt(use_cookies=False)

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
            "2",
            output_pattern,
        ]

        subprocess.run(cmd, check=True)

    def _analyze_nsfw_frames(self, frames_dir: Path) -> float:
        if self.nude_detector is None:
            print("self.nude_detector is None")
            return 0.0

        scores: list[float] = []

        for frame_path in sorted(frames_dir.glob("*.jpg")):
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

        subprocess.run(cmd, check=True)


    def _transcribe_audio(self, audio_path: Path) -> tuple[str, str]:
        if self.whisper_model is None:
            print("WhisperModel is None")
            return "", ""

        # 1. Речь на исходном языке
        original_segments, original_info = self.whisper_model.transcribe(
            str(audio_path),
            task="transcribe",
        )

        original_parts: list[str] = []
        for segment in original_segments:
            original_parts.append(segment.text.strip())

        original_text = " ".join(original_parts).strip()

        # 2. Перевод речи на английский
        translated_segments, translated_info = self.whisper_model.transcribe(
            str(audio_path),
            task="translate",
        )

        translated_parts: list[str] = []
        for segment in translated_segments:
            translated_parts.append(segment.text.strip())

        translated_text = " ".join(translated_parts).strip()

        return original_text, translated_text


    def _analyze_speech_keywords(self, transcript: str) -> float:
        return self._analyze_title_keywords(transcript)