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

# ─── CONFIG ──────────────────────────────────────────────────────────────────
def _env(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set.\n"
            "Go to Railway → your service → Variables and add it."
        )
    return val

LOG_LEVEL      = logging.INFO
MINUTES_WINDOW = 20
FOOTBALL_UA_RSS = "https://football.ua/rss2.ashx"

SKIP_KEYWORDS = [
    "відео", "огляд", "video", "review", "highlights",
    "highlight", "дивіться", "трансляці", "обзор", "watch",
]

# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("football-agent")

# ─── STARTUP CHECK ───────────────────────────────────────────────────────────
def check_env() -> None:
    for key in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHANNEL", "GROQ_API_KEY"]:
        _env(key)

# ─── DATABASE ────────────────────────────────────────────────────────────────
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

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def is_skip(title: str) -> bool:
    lower = title.lower()
    return any(kw in lower for kw in SKIP_KEYWORDS)

def url_to_image_bytes(url: str) -> bytes | None:
    """Download image; try to get highest-res version."""
    if not url:
        return None
    # football.ua thumbnails look like: /img/news/123_100x75.jpg
    # replace known thumb patterns with full-size
    url = re.sub(r'_\d+x\d+(\.\w+)$', r'\1', url)
    # remove common width query params
    url = re.sub(r'[?&](w|width|size|thumb)=\d+', '', url)
    try:
        r = requests.get(
            url, timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            },
        )
        r.raise_for_status()
        return r.content
    except Exception as e:
        log.warning("Image download failed %s: %s", url, e)
        return None

# ─── AI ──────────────────────────────────────────────────────────────────────
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
        return title

# ─── SOURCE 1: FOOTBALL.UA ───────────────────────────────────────────────────
def _best_image_from_page(url: str) -> str | None:
    """Scrape the article page and return the best available image URL."""
    try:
        r    = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")

        # 1. og:image — usually the full-res cover image
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            return str(og["content"])

        # 2. Largest <img> inside article body
        for sel in [".news-text", ".article-content", "article"]:
            block = soup.select_one(sel)
            if block:
                imgs = block.find_all("img")
                for img in imgs:
                    src = img.get("src") or img.get("data-src") or ""
                    if src and not src.endswith(".gif"):
                        if src.startswith("//"):
                            src = "https:" + src
                        elif src.startswith("/"):
                            src = "https://football.ua" + src
                        return src
    except Exception:
        pass
    return None

def fetch_football_ua() -> list[dict]:
    log.info("Fetching football.ua RSS…")
    feed   = feedparser.parse(FOOTBALL_UA_RSS)
    cutoff = time.time() - MINUTES_WINDOW * 60
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

        # Always prefer full-res image from the article page
        image_url = _best_image_from_page(url)

        # Fallback to RSS enclosure thumbnail only if page scrape fails
        if not image_url:
            for enc in entry.get("enclosures", []):
                if enc.get("type", "").startswith("image"):
                    image_url = enc.get("url")
                    break

        results.append({
            "title":       title,
            "url":         url,
            "image_url":   image_url,
            "description": description,
            "source":      "football.ua",
        })

    log.info("  football.ua: %d articles after filter", len(results))
    return results

def fetch_football_ua_article_body(url: str) -> str:
    try:
        r    = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        for sel in [".news-text", ".article-content", "article"]:
            block = soup.select_one(sel)
            if block:
                return block.get_text(" ", strip=True)
        return ""
    except Exception:
        return ""

# ─── SOURCE 2: ONEFOOTBALL ───────────────────────────────────────────────────
# OneFootball renders via JS. We open the home page, collect article links,
# then open each article individually to get title + image + date + body.

OF_HOME = "https://onefootball.com/en/home"

async def _playwright_get(url: str, wait: int = 3000) -> str:
    """Launch headless Chromium, load url, return page HTML."""
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
        try:
            await page.goto(url, wait_until="networkidle", timeout=35_000)
            await page.wait_for_timeout(wait)
            # scroll down to trigger lazy-load
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
            await page.wait_for_timeout(1500)
            html = await page.content()
        finally:
            await browser.close()
    return html


async def fetch_onefootball() -> list[dict]:
    log.info("Fetching onefootball.com home page…")
    try:
        html = await _playwright_get(OF_HOME, wait=4000)
    except Exception as e:
        log.error("Could not load OneFootball home: %s", e)
        return []

    soup      = BeautifulSoup(html, "html.parser")
    seen_urls: set[str] = set()
    results:   list[dict] = []

    # Collect all /en/news/ links from the page
    for a in soup.find_all("a", href=re.compile(r"/en/news/")):
        href     = a.get("href", "")
        full_url = urljoin("https://onefootball.com", href)

        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        # Quick title check from link text to pre-filter obvious video/review
        quick_title = a.get_text(" ", strip=True)[:150]
        if is_skip(quick_title):
            continue

        results.append({"url": full_url})
        if len(results) >= 15:  # grab extra, some will be filtered later
            break

    log.info("  onefootball home: %d candidate links", len(results))
    return results


async def fetch_onefootball_article(url: str) -> dict | None:
    """
    Load a single OneFootball article page.
    Returns dict with title, image_url, body, pub_dt — or None if should be skipped.
    """
    try:
        html = await _playwright_get(url, wait=2500)
    except Exception as e:
        log.warning("Could not load OF article %s: %s", url, e)
        return None

    soup = BeautifulSoup(html, "html.parser")

    # ── Title ──────────────────────────────────────────────────────────────
    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = str(og_title["content"]).strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""
    if not title or is_skip(title):
        return None

    # ── Image — prefer og:image (full res) ─────────────────────────────────
    image_url: str | None = None
    og_img = soup.find("meta", property="og:image")
    if og_img and og_img.get("content"):
        image_url = str(og_img["content"])
    if not image_url:
        # fallback: first large <img> in article
        for sel in ["article", "main", "[class*='article']"]:
            block = soup.select_one(sel)
            if block:
                for img in block.find_all("img"):
                    src = img.get("src") or img.get("data-src") or ""
                    if src and "logo" not in src.lower():
                        image_url = ("https:" + src) if src.startswith("//") else src
                        break
                if image_url:
                    break

    # ── Publish date ───────────────────────────────────────────────────────
    pub_dt: datetime.datetime | None = None
    for time_tag in soup.find_all("time"):
        dt_str = time_tag.get("datetime", "")
        if dt_str:
            try:
                pub_dt = datetime.datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                break
            except ValueError:
                pass
    if not pub_dt:
        meta_date = soup.find("meta", property="article:published_time")
        if meta_date and meta_date.get("content"):
            try:
                pub_dt = datetime.datetime.fromisoformat(
                    str(meta_date["content"]).replace("Z", "+00:00")
                )
            except ValueError:
                pass

    # ── Body text ──────────────────────────────────────────────────────────
    body = ""
    for sel in ["article", ".article-body", "main"]:
        block = soup.select_one(sel)
        if block:
            body = block.get_text(" ", strip=True)
            break

    return {
        "title":     title,
        "url":       url,
        "image_url": image_url,
        "pub_dt":    pub_dt,
        "body":      body,
        "source":    "onefootball",
    }

# ─── TELEGRAM POSTING ────────────────────────────────────────────────────────
async def post_to_telegram(bot: Bot, article: dict, channel: str) -> None:
    title   = article["title"]
    summary = article.get("summary") or ""
    img_url = article.get("image_url")

    # Caption: bold title + summary only, no source link
    caption = f"<b>{title}</b>\n\n{summary}"

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
                disable_web_page_preview=True,
            )
        log.info("  ✅ Posted: %s", title[:70])
    except Exception as e:
        log.error("  ❌ Telegram error: %s", e)

# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────
async def run_pipeline() -> None:
    check_env()
    conn    = init_db()
    bot     = Bot(token=_env("TELEGRAM_BOT_TOKEN"))
    channel = _env("TELEGRAM_CHANNEL")

    # ── 1. Football.ua ───────────────────────────────────────────────────────
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

    # ── 2. OneFootball ───────────────────────────────────────────────────────
    of_candidates = await fetch_onefootball()
    cutoff_dt     = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=MINUTES_WINDOW)
    posted_of     = 0

    for candidate in of_candidates:
        url = candidate["url"]

        if is_published(conn, url):
            log.debug("  SKIP (already published): %s", url)
            continue

        article = await fetch_onefootball_article(url)
        if not article:
            continue

        # Date filter
        pub_dt = article.get("pub_dt")
        if pub_dt:
            pub_aware = pub_dt if pub_dt.tzinfo else pub_dt.replace(tzinfo=datetime.timezone.utc)
            if pub_aware < cutoff_dt:
                log.debug("  SKIP (too old %s): %s", pub_dt, article["title"][:60])
                mark_published(conn, url, article["title"])  # remember so we don't re-check
                continue

        article["summary"] = ai_summarise(
            article["title"],
            article.get("body", ""),
            translate_to_uk=True,
        )
        await post_to_telegram(bot, article, channel)
        mark_published(conn, url, article["title"])
        posted_of += 1
        await asyncio.sleep(3)

        if posted_of >= 10:
            break

    log.info("onefootball: %d articles posted", posted_of)
    conn.close()
    log.info("Pipeline done.")


if __name__ == "__main__":
    asyncio.run(run_pipeline())
