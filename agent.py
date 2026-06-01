"""
Football News Telegram Agent
Sources:
  1. football.ua  — scrapes «ГОЛОВНЕ ЗА ДОБУ» block, posts only last 20 min
  2. onefootball.com — RSS feed, last 10 news, translated to Ukrainian
AI:      Groq API (llama-3.3-70b-versatile)
Hosting: Railway
"""

import os
import re
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
from openai import OpenAI
from telegram import Bot
from telegram.constants import ParseMode

# ─── CONFIG ──────────────────────────────────────────────────────────────────
def _env(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        raise EnvironmentError(f"Missing env variable: '{key}'")
    return val

MINUTES_WINDOW   = 20
FOOTBALL_UA_HOME = "https://football.ua/"
OF_RSS           = "https://onefootball.com/rss/news/en"

SKIP_KEYWORDS = [
    "відео", "огляд", "video", "review", "highlights", "highlight",
    "дивіться", "трансляці", "обзор", "watch", "анонс", "прогноз",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("football-agent")

# ─── DATABASE ────────────────────────────────────────────────────────────────
def init_db() -> sqlite3.Connection:
    data_dir = Path(os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "/tmp"))
    db_path  = data_dir / "published.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS published (
        url_hash TEXT PRIMARY KEY, url TEXT, title TEXT, posted_at TEXT
    )""")
    conn.commit()
    log.info("DB at %s", db_path)
    return conn

def is_published(conn, url):
    h = hashlib.md5(url.encode()).hexdigest()
    return conn.execute("SELECT 1 FROM published WHERE url_hash=?", (h,)).fetchone() is not None

def mark_published(conn, url, title):
    h = hashlib.md5(url.encode()).hexdigest()
    conn.execute(
        "INSERT OR IGNORE INTO published(url_hash,url,title,posted_at) VALUES(?,?,?,?)",
        (h, url, title, datetime.datetime.now().isoformat()),
    )
    conn.commit()

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def is_skip(title: str) -> bool:
    return any(kw in title.lower() for kw in SKIP_KEYWORDS)

def cutoff_dt() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=MINUTES_WINDOW)

def get_image_bytes(url: str) -> bytes | None:
    if not url:
        return None
    try:
        r = requests.get(url, timeout=15, headers=HEADERS)
        r.raise_for_status()
        return r.content
    except Exception as e:
        log.warning("Image download failed: %s", e)
        return None

def http_get(url: str) -> str:
    r = requests.get(url, timeout=10, headers=HEADERS)
    r.raise_for_status()
    return r.text

# ─── AI ──────────────────────────────────────────────────────────────────────
_groq: OpenAI | None = None

def groq() -> OpenAI:
    global _groq
    if _groq is None:
        _groq = OpenAI(api_key=_env("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    return _groq

def ai_summarise(title: str, body: str, translate: bool = False) -> str:
    lang = "Відповідь надай ТІЛЬКИ українською мовою." if translate else "Відповідь надай українською мовою."
    prompt = (
        f"Ти редактор футбольного Telegram-каналу. {lang}\n"
        "Зроби стислий дайджест (2-3 речення, до 200 символів). "
        "Без лапок, без «Коротко:», живо і динамічно.\n\n"
        f"Заголовок: {title}\nТекст: {body[:3000]}"
    )
    try:
        r = groq().chat.completions.create(
            model="llama-3.3-70b-versatile", max_tokens=300, temperature=0.6,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        log.error("Groq error: %s", e)
        return title

# ─── SOURCE 1: FOOTBALL.UA ───────────────────────────────────────────────────
def fetch_football_ua() -> list[dict]:
    """Scrape «ГОЛОВНЕ ЗА ДОБУ» links, open each article, filter by date."""
    log.info("Fetching football.ua…")
    try:
        soup = BeautifulSoup(http_get(FOOTBALL_UA_HOME), "html.parser")
    except Exception as e:
        log.error("football.ua home error: %s", e)
        return []

    # Find «головне за добу» heading and the <ul> after it
    heading = soup.find(string=re.compile(r"головне за добу", re.IGNORECASE))
    if not heading:
        log.warning("«головне за добу» block not found")
        return []

    parent = heading.find_parent()
    ul = None
    for _ in range(6):
        if not parent:
            break
        ul = parent.find("ul")
        if ul:
            break
        parent = parent.find_parent()

    if not ul:
        log.warning("No <ul> found in «головне за добу»")
        return []

    links = [urljoin(FOOTBALL_UA_HOME, a["href"])
             for li in ul.find_all("li") for a in [li.find("a", href=True)] if a]
    log.info("  %d links in «головне за добу»", len(links))

    results = []
    for url in links:
        article = _fetch_fu_article(url)
        if article:
            results.append(article)
    return results

def _fetch_fu_article(url: str) -> dict | None:
    try:
        soup = BeautifulSoup(http_get(url), "html.parser")
    except Exception as e:
        log.warning("football.ua article error %s: %s", url, e)
        return None

    # Title
    og = soup.find("meta", property="og:title")
    title = str(og["content"]).strip() if og and og.get("content") else ""
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""
    if not title or is_skip(title):
        return None

    # Date — must be within last MINUTES_WINDOW
    pub_dt = None
    time_tag = soup.find("time")
    if time_tag and time_tag.get("datetime"):
        try:
            pub_dt = datetime.datetime.fromisoformat(str(time_tag["datetime"]).replace("Z", "+00:00"))
        except ValueError:
            pass
    if not pub_dt:
        meta = soup.find("meta", property="article:published_time")
        if meta and meta.get("content"):
            try:
                pub_dt = datetime.datetime.fromisoformat(str(meta["content"]).replace("Z", "+00:00"))
            except ValueError:
                pass

    if not pub_dt:
        log.debug("  SKIP (no date): %s", title[:60])
        return None

    pub_aware = pub_dt if pub_dt.tzinfo else pub_dt.replace(tzinfo=datetime.timezone.utc)
    if pub_aware < cutoff_dt():
        log.debug("  SKIP (old %s): %s", pub_dt.strftime("%H:%M"), title[:60])
        return None

    # Image (og:image = full res)
    image_url = None
    og_img = soup.find("meta", property="og:image")
    if og_img and og_img.get("content"):
        image_url = str(og_img["content"])

    # Body
    body = ""
    for sel in [".news-text", ".article-content", "article"]:
        block = soup.select_one(sel)
        if block:
            body = block.get_text(" ", strip=True)
            break

    return {"title": title, "url": url, "image_url": image_url, "body": body, "source": "football.ua"}

# ─── SOURCE 2: ONEFOOTBALL (RSS) ─────────────────────────────────────────────
def fetch_onefootball() -> list[dict]:
    """Parse OneFootball RSS feed — no browser needed."""
    log.info("Fetching OneFootball RSS…")
    try:
        feed = feedparser.parse(OF_RSS)
    except Exception as e:
        log.error("OneFootball RSS error: %s", e)
        return []

    import time as time_module
    cutoff_ts = time_module.time() - MINUTES_WINDOW * 60
    results = []

    for entry in feed.entries:
        title = entry.get("title", "").strip()
        if not title or is_skip(title):
            continue

        # Date filter
        pub = entry.get("published_parsed")
        if pub and time_module.mktime(pub) < cutoff_ts:
            continue

        url = entry.get("link", "")

        # Image from enclosure or media
        image_url = None
        for enc in entry.get("enclosures", []):
            if enc.get("type", "").startswith("image"):
                image_url = enc.get("url")
                break
        if not image_url:
            media = entry.get("media_content", [])
            if media:
                image_url = media[0].get("url")
        if not image_url:
            # try og:image from article page
            try:
                soup = BeautifulSoup(http_get(url), "html.parser")
                og_img = soup.find("meta", property="og:image")
                if og_img and og_img.get("content"):
                    image_url = str(og_img["content"])
            except Exception:
                pass

        body = entry.get("summary", "") or entry.get("description", "")

        results.append({
            "title": title, "url": url, "image_url": image_url,
            "body": body, "source": "onefootball",
        })

        if len(results) >= 10:
            break

    log.info("  OneFootball: %d articles after filter", len(results))
    return results

# ─── TELEGRAM ────────────────────────────────────────────────────────────────
async def post_to_telegram(bot: Bot, article: dict, channel: str) -> None:
    caption = f"<b>{article['title']}</b>\n\n{article.get('summary', '')}"
    if len(caption) > 1024:
        caption = caption[:1020] + "…"

    img = get_image_bytes(article.get("image_url"))
    try:
        if img:
            await bot.send_photo(chat_id=channel, photo=img,
                                 caption=caption, parse_mode=ParseMode.HTML)
        else:
            await bot.send_message(chat_id=channel, text=caption,
                                   parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        log.info("  ✅ %s", article["title"][:70])
    except Exception as e:
        log.error("  ❌ Telegram: %s", e)

# ─── PIPELINE ────────────────────────────────────────────────────────────────
async def run_pipeline() -> None:
    for key in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHANNEL", "GROQ_API_KEY"]:
        _env(key)

    conn    = init_db()
    bot     = Bot(token=_env("TELEGRAM_BOT_TOKEN"))
    channel = _env("TELEGRAM_CHANNEL")
    log.info("Pipeline started. Window: last %d min.", MINUTES_WINDOW)

    # 1. Football.ua
    for article in fetch_football_ua():
        if is_published(conn, article["url"]):
            continue
        article["summary"] = ai_summarise(article["title"], article["body"])
        await post_to_telegram(bot, article, channel)
        mark_published(conn, article["url"], article["title"])
        await asyncio.sleep(2)

    # 2. OneFootball
    posted = 0
    for article in fetch_onefootball():
        if is_published(conn, article["url"]):
            continue
        article["summary"] = ai_summarise(article["title"], article["body"], translate=True)
        await post_to_telegram(bot, article, channel)
        mark_published(conn, article["url"], article["title"])
        posted += 1
        await asyncio.sleep(2)
        if posted >= 10:
            break

    log.info("Done. OF posted: %d", posted)
    conn.close()

if __name__ == "__main__":
    asyncio.run(run_pipeline())
