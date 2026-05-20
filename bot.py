"""
Football News Telegram Agent
=============================
Sources:
  1. football.ua  — RSS feed, «головне за добу», skip video/review
  2. onefootball.com — last 10 news via Playwright, translate to Ukrainian, skip video/review
AI:      Groq API (llama-3.3-70b-versatile)
Hosting: Railway — all secrets come from environment variables (no .env needed)
"""

import os
import re
import time
import hashlib
import sqlite3
import logging
import asyncio
import datetime
from pathlib import Path
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from openai import OpenAI
from telegram import Bot
from telegram.constants import ParseMode

# ─── CONFIG (all values come from Railway environment variables) ─────────────
# Variables are read lazily inside run_pipeline() — NOT at module level.
# This prevents KeyError crashes when Railway imports the module before
# injecting the environment variables.
def _env(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set.\n"
            "Go to Railway → your service → Variables and add it."
        )
    return val

# DB path — resolved lazily in init_db()
DB_PATH: Path | None = None

LOG_LEVEL = logging.INFO

# Keywords that mark video/review articles — skip these
SKIP_KEYWORDS = [
    "відео", "огляд", "video", "review", "highlights",
    "highlight", "дивіться", "трансляці", "обзор", "watch",
]

# ─── LOGGING ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("football-agent")

# ─── STARTUP CHECK ──────────────────────────────────────────────────────────
def check_env() -> None:
    """Fail fast with a clear message if any required variable is missing."""
    for key in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHANNEL", "GROQ_API_KEY"]:
        _env(key)  # raises EnvironmentError if missing

# ─── DATABASE ───────────────────────────────────────────────────────────────
def init_db() -> sqlite3.Connection:
    data_dir = Path(os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "/tmp"))
    db_path  = data_dir / "published.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS published (
            url_hash  TEXT PRIMARY KEY,
            url       TEXT,
            title     TEXT,
            posted_at TEXT
        )"""
    )
    conn.commit()
    log.info("DB at %s", db_path)
    return conn


def is_published(conn: sqlite3.Connection, url: str) -> bool:
    h = hashlib.md5(url.encode()).hexdigest()
    return conn.execute(
        "SELECT 1 FROM published WHERE url_hash=?", (h,)
    ).fetchone() is not None


def mark_published(conn: sqlite3.Connection, url: str, title: str) -> None:
    h = hashlib.md5(url.encode()).hexdigest()
    conn.execute(
        "INSERT OR IGNORE INTO published(url_hash, url, title, posted_at) VALUES(?,?,?,?)",
        (h, url, title, datetime.datetime.now().isoformat()),
    )
    conn.commit()

# ─── HELPERS ────────────────────────────────────────────────────────────────
def is_skip(title: str) -> bool:
    """Return True if the article looks like video/review content."""
    lower = title.lower()
    return any(kw in lower for kw in SKIP_KEYWORDS)


def url_to_image_bytes(url: str) -> bytes | None:
    """Download image bytes; return None on any error."""
    if not url:
        return None
    try:
        r = requests.get(
            url, timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FootballBot/1.0)"},
        )
        r.raise_for_status()
        return r.content
    except Exception as e:
        log.warning("Image download failed %s: %s", url, e)
        return None

# ─── AI: SUMMARISE & TRANSLATE via Groq ─────────────────────────────────────
_groq_client: OpenAI | None = None

def get_groq_client() -> OpenAI:
    global _groq_client
    if _groq_client is None:
        _groq_client = OpenAI(
            api_key=_env("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
    return _groq_client


def ai_summarise(title: str, body: str, translate_to_uk: bool = False) -> str:
    """
    Summarise a news article in 2-3 sentences using Groq (llama-3.3-70b).
    If translate_to_uk=True, output is always in Ukrainian regardless of input language.
    """
    lang_instruction = (
        "Відповідь надай ТІЛЬКИ українською мовою."
        if translate_to_uk
        else "Відповідь надай українською мовою."
    )

    prompt = (
        "Ти редактор спортивного Telegram-каналу про футбол.\n"
        f"{lang_instruction}\n\n"
        "Зроби стислий дайджест новини (2-3 речення, максимум 200 символів) "
        "на основі заголовку та тексту нижче.\n"
        "Не використовуй лапки на початку/кінці. "
        "Не починай з «Коротко:» або подібних міток. "
        "Будь живим і динамічним у подачі.\n\n"
        f"Заголовок: {title}\n\n"
        f"Текст: {body[:3000]}"
    )

    try:
        response = get_groq_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            temperature=0.6,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error("Groq API error: %s", e)
        # Fallback — return the raw title so the post still goes out
        return title

# ─── SOURCE 1: FOOTBALL.UA ──────────────────────────────────────────────────
FOOTBALL_UA_RSS = "https://football.ua/rss2.ashx"
HOURS_WINDOW    = 24   # look back this many hours for «головне за добу»


def fetch_football_ua() -> list[dict]:
    log.info("Fetching football.ua RSS…")
    feed   = feedparser.parse(FOOTBALL_UA_RSS)
    cutoff = time.time() - HOURS_WINDOW * 3600
    results: list[dict] = []

    for entry in feed.entries:
        pub = entry.get("published_parsed")
        if pub and time.mktime(pub) < cutoff:
            continue

        title = entry.get("title", "").strip()
        if not title or is_skip(title):
            log.debug("  SKIP: %s", title)
            continue

        url         = entry.get("link", "")
        description = entry.get("description", "")

        # Image from RSS <enclosure>
        image_url = None
        for enc in entry.get("enclosures", []):
            if enc.get("type", "").startswith("image"):
                image_url = enc.get("url")
                break

        # Fallback: og:image from article page
        if not image_url:
            image_url = _scrape_og_image(url)

        results.append({
            "title":     title,
            "url":       url,
            "image_url": image_url,
            "description": description,
            "source":    "football.ua",
        })

    log.info("  football.ua: %d articles after filter", len(results))
    return results


def _scrape_og_image(url: str) -> str | None:
    try:
        r   = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        tag = BeautifulSoup(r.text, "html.parser").find("meta", property="og:image")
        return tag["content"] if tag else None  # type: ignore[index]
    except Exception:
        return None


def fetch_football_ua_article_body(url: str) -> str:
    try:
        r    = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        for selector in [".news-text", ".article-content", "article"]:
            block = soup.select_one(selector)
            if block:
                return block.get_text(" ", strip=True)
        return ""
    except Exception:
        return ""

# ─── SOURCE 2: ONEFOOTBALL ──────────────────────────────────────────────────
async def fetch_onefootball() -> list[dict]:
    log.info("Fetching onefootball.com via Playwright…")
    results: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],  # required on Railway
        )
        page = await browser.new_page(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        )
        await page.goto(
            "https://onefootball.com/en/home",
            wait_until="networkidle",
            timeout=30_000,
        )
        await page.wait_for_timeout(3000)
        html = await page.content()
        await browser.close()

    soup     = BeautifulSoup(html, "html.parser")
    seen_urls: set[str] = set()
    cards    = soup.find_all("a", href=re.compile(r"/en/news/"))

    for a in cards:
        href     = a.get("href", "")
        full_url = urljoin("https://onefootball.com", href)

        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        # Extract title
        heading = a.find(["h1", "h2", "h3", "h4"])
        title = (
            a.get("aria-label")
            or a.get("title")
            or (heading.get_text(strip=True) if heading else "")
            or a.get_text(" ", strip=True)[:120]
        ).strip()

        if not title or is_skip(title):
            log.debug("  SKIP: %s", full_url)
            continue

        # Image
        img_tag   = a.find("img")
        image_url = None
        if img_tag:
            image_url = img_tag.get("src") or img_tag.get("data-src")
            if image_url and image_url.startswith("//"):
                image_url = "https:" + image_url

        results.append({
            "title":     title,
            "url":       full_url,
            "image_url": image_url,
            "description": "",
            "source":    "onefootball",
        })

        if len(results) >= 10:
            break

    log.info("  onefootball: %d articles after filter", len(results))
    return results


async def fetch_onefootball_article(url: str) -> tuple[str, str | None]:
    """Fetch body text + better og:image from an individual OneFootball article."""
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            page = await browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                )
            )
            await page.goto(url, wait_until="networkidle", timeout=25_000)
            await page.wait_for_timeout(2000)
            html = await page.content()
            await browser.close()

        soup      = BeautifulSoup(html, "html.parser")
        og        = soup.find("meta", property="og:image")
        image_url = og["content"] if og else None  # type: ignore[index]

        body = ""
        for sel in ["article", ".article-body", "main", "[class*='article']"]:
            block = soup.select_one(sel)
            if block:
                body = block.get_text(" ", strip=True)
                break

        return body, image_url
    except Exception as e:
        log.warning("Could not load article %s: %s", url, e)
        return "", None

# ─── TELEGRAM POSTING ───────────────────────────────────────────────────────
async def post_to_telegram(bot: Bot, article: dict, channel: str) -> None:
    title   = article["title"]
    url     = article["url"]
    summary = article.get("summary") or article.get("description") or ""
    source  = article["source"]
    img_url = article.get("image_url")

    source_tag = "🇺🇦 football.ua" if source == "football.ua" else "🌍 OneFootball"

    caption = (
        f"<b>{title}</b>\n\n"
        f"{summary}\n\n"
        f"🔗 <a href='{url}'>Читати повністю</a>\n"
        f"{source_tag}"
    )

    if len(caption) > 1024:
        caption = caption[:1020] + "…"

    img_bytes = url_to_image_bytes(img_url)

    try:
        if img_bytes:
            await bot.send_photo(
                chat_id=channel,
                photo=img_bytes,
                caption=caption,
                parse_mode=ParseMode.HTML,
            )
        else:
            await bot.send_message(
                chat_id=channel,
                text=caption,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=False,
            )
        log.info("  ✅ Posted: %s", title[:70])
    except Exception as e:
        log.error("  ❌ Telegram error: %s", e)

# ─── MAIN PIPELINE ──────────────────────────────────────────────────────────
async def run_pipeline() -> None:
    check_env()
    conn = init_db()
    bot  = Bot(token=_env("TELEGRAM_BOT_TOKEN"))
    channel = _env("TELEGRAM_CHANNEL")

    # ── 1. Football.ua ──────────────────────────────────────────────────────
    fu_articles = fetch_football_ua()
    new_fu      = [a for a in fu_articles if not is_published(conn, a["url"])]
    log.info("football.ua: %d new articles to post", len(new_fu))

    for article in new_fu:
        body = fetch_football_ua_article_body(article["url"])
        article["summary"] = ai_summarise(
            article["title"],
            body or article["description"],
            translate_to_uk=False,
        )
        await post_to_telegram(bot, article, channel)
        mark_published(conn, article["url"], article["title"])
        await asyncio.sleep(2)

    # ── 2. OneFootball ──────────────────────────────────────────────────────
    of_articles = await fetch_onefootball()
    new_of      = [a for a in of_articles if not is_published(conn, a["url"])]
    log.info("onefootball: %d new articles to post", len(new_of))

    for article in new_of:
        body, better_img = await fetch_onefootball_article(article["url"])
        if better_img and not article["image_url"]:
            article["image_url"] = better_img
        article["summary"] = ai_summarise(
            article["title"],
            body,
            translate_to_uk=True,
        )
        await post_to_telegram(bot, article, channel)
        mark_published(conn, article["url"], article["title"])
        await asyncio.sleep(3)

    conn.close()
    log.info("Pipeline done.")


if __name__ == "__main__":
    asyncio.run(run_pipeline())
