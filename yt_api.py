import csv
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import requests
from dateutil import parser as dt_parser

from videoFormat import probe_video_format
from danger_moderation import VideoDangerAnalyzer
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY", "")
BASE_URL = "https://www.googleapis.com/youtube/v3"


# =========================
# Конфиг
# =========================

@dataclass
class SearchFilters:
    query: str
    content_format: str = "all"
    video_length: str = "any"
    max_results: int = 25
    published_after_hours: Optional[int] = None
    check_dangerous_content: bool = False


@dataclass
class VideoResult:
    video_id: str
    title: str
    channel_title: str
    published_at: str
    hours_since_publish: float
    duration: str
    view_count: int
    like_count: int
    comment_count: int
    trend_score: float
    url: str
    danger_checked: bool = False
    title_score: float = 0.0
    visual_nsfw_score: float = 0.0
    visual_violence_score: float = 0.0
    speech_score: float = 0.0
    audio_event_score: float = 0.0
    danger_score: float = 0.0
    danger_label: str = "unknown"
    original_speech: str = ""
    translated_speech: str = ""
    speech_reasons: list[str] | None = None

# =========================
# Вспомогательные функции
# =========================

def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_iso_datetime(value: str) -> datetime:
    return dt_parser.isoparse(value).astimezone(timezone.utc)


def hours_since(dt: datetime) -> float:
    now = datetime.now(timezone.utc)
    diff = now - dt
    return max(diff.total_seconds() / 3600.0, 0.01)


def parse_duration_to_seconds(duration: str) -> int:
    """
    Упрощенный разбор ISO 8601 duration вида PT15M33S, PT2H1M, PT59S
    """
    if not duration or not duration.startswith("P"):
        return 0

    total = 0
    num = ""
    in_time = False

    for ch in duration:
        if ch == "T":
            in_time = True
            continue

        if ch.isdigit():
            num += ch
            continue

        if not num:
            continue

        value = int(num)
        num = ""

        if ch == "D":
            total += value * 86400
        elif ch == "H":
            total += value * 3600
        elif ch == "M" and in_time:
            total += value * 60
        elif ch == "S":
            total += value

    return total


def format_duration(seconds: int) -> str:
    if seconds <= 0:
        return "0:00"

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02}:{secs:02}"
    return f"{minutes}:{secs:02}"


def calculate_trend_score(
    views: int,
    likes: int,
    comments: int,
    hours_from_publish: float,
) -> float:
    """
    Это не официальный рейтинг YouTube.
    Это наша эвристика: чем больше просмотров в единицу времени,
    чем лучше вовлеченность и чем свежее ролик — тем выше score.
    """
    views_per_hour = views / max(hours_from_publish, 1.0)

    like_ratio = likes / views if views > 0 else 0.0
    comment_ratio = comments / views if views > 0 else 0.0

    freshness_bonus = 1 / max(math.log(hours_from_publish + 2), 1.0)

    score = (
        views_per_hour * 0.65
        + like_ratio * 100000 * 0.20
        + comment_ratio * 100000 * 0.10
        + freshness_bonus * 100 * 0.05
    )

    return round(score, 2)


# =========================
# YouTube API клиент
# =========================

class YouTubeTrendAnalyzer:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.session = requests.Session()
        self.video_format_cache: dict[str, Any] = {}
        self.danger_analyzer = VideoDangerAnalyzer()

    def _request(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "key": self.api_key}
        response = self.session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_format_info(self, video_url: str):
        if video_url not in self.video_format_cache:
            self.video_format_cache[video_url] = probe_video_format(video_url)
        return self.video_format_cache[video_url]

    def search_videos(self, filters: SearchFilters) -> list[str]:
        search_pool_size = max(filters.max_results * 10, 50)

        params: dict[str, Any] = {
            "part": "snippet",
            "q": filters.query,
            "type": "video",
            "maxResults": min(search_pool_size, 50),
            "order": "viewCount",
        }

        if filters.video_length != "any":
            params["videoDuration"] = filters.video_length

        if filters.published_after_hours is not None:
            published_after_dt = datetime.now(timezone.utc) - timedelta(hours=filters.published_after_hours)
            params["publishedAfter"] = published_after_dt.isoformat().replace("+00:00", "Z")

        data = self._request("search", params)

        video_ids: list[str] = []
        for item in data.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            if video_id:
                video_ids.append(video_id)

        return video_ids

    def get_video_details(self, video_ids: list[str]) -> list[dict[str, Any]]:
        if not video_ids:
            return []

        all_items: list[dict[str, Any]] = []

        for i in range(0, len(video_ids), 50):
            chunk = video_ids[i:i + 50]
            params = {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(chunk),
            }
            data = self._request("videos", params)
            all_items.extend(data.get("items", []))

        return all_items

    def _duration_str_to_seconds(self, value: str) -> int:
        parts = value.split(":")
        try:
            if len(parts) == 2:
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            if len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            return 0
        return 0
    
    def _apply_danger_analysis(self, results: list[VideoResult], filters: SearchFilters) -> None:

        if not filters.check_dangerous_content:
            return

        for idx, item in enumerate(results, start=1):
            print(f"\n[Danger check #{idx}] Анализирую: {item.title}")

            duration_seconds = self._duration_str_to_seconds(item.duration)

            verdict = self.danger_analyzer.analyze_video_url(
                video_url=item.url,
                title=item.title,  # ВАЖНО
                duration_seconds=duration_seconds,
            )

            item.danger_checked = True

            item.title_score = verdict.title_score
            item.visual_nsfw_score = verdict.visual_nsfw_score
            item.visual_violence_score = verdict.visual_violence_score
            item.speech_score = verdict.speech_score
            item.audio_event_score = verdict.audio_event_score
            item.danger_score = verdict.danger_score
            item.danger_label = verdict.label

            item.original_speech = verdict.original_speech
            item.translated_speech = verdict.translated_speech
            item.speech_reasons = verdict.speech_reasons

    def analyze(self, filters: SearchFilters) -> list[VideoResult]:
        video_ids = self.search_videos(filters)
        details = self.get_video_details(video_ids)

        all_results: list[VideoResult] = []
        short_candidates: list[VideoResult] = []
        regular_uncertain_candidates: list[VideoResult] = []

        for item in details:
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})
            content_details = item.get("contentDetails", {})

            published_at = snippet.get("publishedAt")
            if not published_at:
                continue

            published_dt = parse_iso_datetime(published_at)
            video_hours_since_publish = hours_since(published_dt)

            if (
                filters.published_after_hours is not None
                and video_hours_since_publish > filters.published_after_hours
            ):
                continue

            duration_seconds = parse_duration_to_seconds(
                content_details.get("duration", "")
            )

            video_id = item.get("id", "")
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            views = safe_int(statistics.get("viewCount"))
            likes = safe_int(statistics.get("likeCount"))
            comments = safe_int(statistics.get("commentCount"))

            trend_score = calculate_trend_score(
                views=views,
                likes=likes,
                comments=comments,
                hours_from_publish=video_hours_since_publish,
            )

            result = VideoResult(
                video_id=video_id,
                title=snippet.get("title", ""),
                channel_title=snippet.get("channelTitle", ""),
                published_at=published_at,
                hours_since_publish=round(video_hours_since_publish, 2),
                duration=format_duration(duration_seconds),
                view_count=views,
                like_count=likes,
                comment_count=comments,
                trend_score=trend_score,
                url=video_url,
            )

            if filters.content_format == "all":
                all_results.append(result)

            elif filters.content_format == "shorts":
                if duration_seconds <= 180:
                    short_candidates.append(result)

            elif filters.content_format == "regular":
                if duration_seconds > 180:
                    all_results.append(result)
                else:
                    regular_uncertain_candidates.append(result)

        all_results.sort(key=lambda x: x.trend_score, reverse=True)
        short_candidates.sort(key=lambda x: x.trend_score, reverse=True)
        regular_uncertain_candidates.sort(key=lambda x: x.trend_score, reverse=True)

        if filters.content_format == "all":
            final_results = all_results[:filters.max_results]
            self._apply_danger_analysis(final_results, filters)
            return final_results

        if filters.content_format == "shorts":
            final_results: list[VideoResult] = []

            for candidate in short_candidates:
                format_info = self.get_format_info(candidate.url)

                if format_info.is_vertical:
                    final_results.append(candidate)
                    print_found_result(candidate, len(final_results))

                if len(final_results) >= filters.max_results:
                    break

            self._apply_danger_analysis(final_results, filters)
            return final_results

        if filters.content_format == "regular":
            final_results: list[VideoResult] = []

            for candidate in all_results:
                final_results.append(candidate)
                print_found_result(candidate, len(final_results))

                if len(final_results) >= filters.max_results:
                    self._apply_danger_analysis(final_results, filters)
                    return final_results

            for candidate in regular_uncertain_candidates:
                format_info = self.get_format_info(candidate.url)

                if not format_info.is_vertical:
                    final_results.append(candidate)
                    print_found_result(candidate, len(final_results))

                if len(final_results) >= filters.max_results:
                    break

            final_results.sort(key=lambda x: x.trend_score, reverse=True)
            final_results = final_results[:filters.max_results]
            self._apply_danger_analysis(final_results, filters)
            return final_results

        return all_results[:filters.max_results]

# =========================
# Вывод / CSV
# =========================

def print_found_result(item: VideoResult, idx: int) -> None:
    print(f"\n[Найдено #{idx}] {item.title}")
    print(f"   Канал: {item.channel_title}")
    print(f"   Опубликовано: {item.published_at}")
    print(f"   Часов с публикации: {item.hours_since_publish}")
    print(f"   Длительность: {item.duration}")
    print(f"   Просмотры: {item.view_count}")
    print(f"   Лайки: {item.like_count}")
    print(f"   Комменты: {item.comment_count}")
    print(f"   Trend score: {item.trend_score}")
    print(f"   URL: {item.url}")
    if item.danger_checked:
        print(f"   Категория: {item.danger_label}")
        print(f"   Контент 18+ (NSFW): {item.nsfw_score}")
        print(f"   Опасные слова в названии: {item.violence_score}")

def print_results(results: list[VideoResult]) -> None:
    if not results:
        print("Ничего не найдено под заданные фильтры.")
        return

    print("\n=== РЕЗУЛЬТАТЫ ===\n")
    for idx, item in enumerate(results, start=1):
        print(f"{idx}. {item.title}")
        print(f"   Канал: {item.channel_title}")
        print(f"   Опубликовано: {item.published_at}")
        print(f"   Часов с публикации: {item.hours_since_publish}")
        print(f"   Длительность: {item.duration}")
        print(f"   Просмотры: {item.view_count}")
        print(f"   Лайки: {item.like_count}")
        print(f"   Комменты: {item.comment_count}")
        print(f"   Trend score: {item.trend_score}")
        print(f"   URL: {item.url}")
        if item.danger_checked:
            print(f"   Категория опасности: {item.danger_label}")
            print(f"   Общий рейтинг опасности: {item.danger_score}")
            print(f"   Рейтинг опасности названия: {item.title_score}")
            print(f"   Рейтинг NSFW по кадрам: {item.visual_nsfw_score}")
            print(f"   Рейтинг насилия по видеоряду: {item.visual_violence_score}")
            print(f"   Рейтинг опасности речи: {item.speech_score}")
            print(f"   Рейтинг опасных звуков: {item.audio_event_score}")

            if item.original_speech:
                print(f"   Речь в видео: {item.original_speech[:500]}")
            else: 
                print("   Речь в видео не обнаружена!")

            if item.translated_speech:
                print(f"   Перевод речи EN: {item.translated_speech[:500]}")
            
            if item.speech_reasons:
                print(f"   Опасные слова в речи: {', '.join(item.speech_reasons)}")
        print()


def save_to_csv(results: list[VideoResult], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "title",
                "channel_title",
                "published_at",
                "hours_since_publish",
                "duration",
                "view_count",
                "like_count",
                "comment_count",
                "trend_score",
                "url",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


# =========================
# CLI
# =========================

def ask_optional(prompt: str) -> Optional[str]:
    value = input(prompt).strip()
    return value or None


def ask_int(prompt: str, default: int = 0) -> int:
    raw = input(prompt).strip()
    if not raw:
        return default
    return int(raw)


def ask_float_optional(prompt: str) -> Optional[float]:
    raw = input(prompt).strip()
    if not raw:
        return None
    return float(raw)


def build_filters_from_input() -> SearchFilters:
    print("=== YouTube Trend Analyzer ===")
    print("Оставь пустым, если фильтр не нужен.\n")

    query = input("Запрос: ").strip()
    if not query:
        raise ValueError("Запрос не может быть пустым.")

    content_format = input(
        "Формат [all/shorts/regular] (по умолчанию all): "
    ).strip().lower() or "all"

    video_length = input(
        "Длина видео [any/short/medium/long] (по умолчанию any): "
    ).strip().lower() or "any"

    max_results = int(input("Сколько результатов [25]: ").strip() or "25")

    raw_hours = input("За последние N часов (например 24, 48): ").strip()
    published_after_hours = int(raw_hours) if raw_hours else None

    # 👉 ВОТ СЮДА ДОБАВЛЯЕМ
    raw_check = input("Проверять содержимое видео на опасный контент? [y/N]: ").strip().lower()
    check_dangerous_content = raw_check == "y"

    return SearchFilters(
        query=query,
        content_format=content_format,
        video_length=video_length,
        max_results=max_results,
        published_after_hours=published_after_hours,
        check_dangerous_content=check_dangerous_content,
)


def main() -> None:
    api_key = API_KEY.strip()
    if not api_key:
        print("Ошибка: API_KEY пустой. Вставь свой ключ в код.")
        sys.exit(1)

    try:
        filters = build_filters_from_input()
        analyzer = YouTubeTrendAnalyzer(api_key)
        results = analyzer.analyze(filters)

        print_results(results)

    except requests.HTTPError as e:
        print("HTTP ошибка при запросе к YouTube API:")
        print(e)
        try:
            print(e.response.text)
        except Exception:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()