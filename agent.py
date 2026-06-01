"""
Football News Telegram Agent
=============================
Sources:
  1. football.ua  — scrapes «ГОЛОВНЕ ЗА ДОБУ» block from homepage,
                    posts only articles published in last MINUTES_WINDOW minutes
  2. onefootball.com — last 10 news via Playwright, translate to Ukrainian
AI:      Groq API (llama-3.3-70b-versatile)
Hosting: Railway — all secrets come from environment variables
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
MINUTES_WINDOW = 20  # post only articles newer than this

SKIP_KEYWORDS = [
    "відео", "огляд", "video", "review", "highlights",
    "highlight", "дивіться", "трансляці", "обзор", "watch",
    "анонс", "прогноз",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

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
    if not url:
        return None
    try:
        r = requests.get(url, timeout=15, headers=HEADERS)
        r.raise_for_status()
        return r.content
    except Exception as e:
        log.warning("Image download failed %s: %s", url, e)
        return None

def cutoff_dt() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=MINUTES_WINDOW)

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
        "Зроби стислий дайджест новини (2-3 речення, максимум 200 символів).\n"
        "Не використовуй лапки. Не починай з «Коротко:». "
        "Будь живим і динамічним.\n\n"
        f"Заголовок: {title}\n\nТекст: {body[:3000]}"
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
# We scrape the «ГОЛОВНЕ ЗА ДОБУ» list from the homepage,
# then open each article to check its publish time and get full image + body.

FOOTBALL_UA_HOME = "https://football.ua/"

def fetch_football_ua_links() -> list[str]:
    """Return URLs from the «ГОЛОВНЕ ЗА ДОБУ» block on football.ua homepage."""
    try:
        r    = requests.get(FOOTBALL_UA_HOME, timeout=10, headers=HEADERS)
        soup = BeautifulSoup(r.text, "html.parser")

        # Find the heading «головне за добу» then grab the next <ul>
        heading = soup.find(string=re.compile(r"головне за добу", re.IGNORECASE))
        if not heading:
            log.warning("Could not find «головне за добу» block")
            return []

        # Walk up to find the parent section, then find <ul> or <li> inside
        parent = heading.find_parent()
        for _ in range(5):
            ul = parent.find("ul") if parent else None
            if ul:
                break
            parent = parent.find_parent() if parent else None

        if not ul:
            log.warning("Could not find <ul> in «головне за добу» block")
            return []

        links = []
        for li in ul.find_all("li"):
            a = li.find("a", href=True)
            if a:
                url = urljoin(FOOTBALL_UA_HOME, a["href"])
                links.append(url)

        log.info("  football.ua: %d links in «головне за добу»", len(links))
        return links
    except Exception as e:
        log.error("fetch_football_ua_links error: %s", e)
        return []


def fetch_football_ua_article(url: str) -> dict | None:
    """
    Open a football.ua article page.
    Returns dict with title, image_url, body, pub_dt — or None to skip.
    """
    try:
        r    = requests.get(url, timeout=10, headers=HEADERS)
        soup = BeautifulSoup(r.text, "html.parser")

        # Title
        og_title = soup.find("meta", property="og:title")
        title = str(og_title["content"]).strip() if og_title and og_title.get("content") else ""
        if not title:
            h1 = soup.find("h1")
            title = h1.get_text(strip=True) if h1 else ""
        if not title or is_skip(title):
            return None

        # Publish date — football.ua puts it in <time> or meta
        pub_dt: datetime.datetime | None = None

        # Try <time datetime="...">
        time_tag = soup.find("time")
        if time_tag and time_tag.get("datetime"):
            try:
                pub_dt = datetime.datetime.fromisoformat(
                    str(time_tag["datetime"]).replace("Z", "+00:00")
                )
            except ValueError:
                pass

        # Try meta article:published_time
        if not pub_dt:
            meta = soup.find("meta", property="article:published_time")
            if meta and meta.get("content"):
                try:
                    pub_dt = datetime.datetime.fromisoformat(
                        str(meta["content"]).replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

        # Try visible date text like "27 травня 2026, 10:30"
        if not pub_dt:
            date_el = soup.select_one(".news-date, .article-date, .date, [class*='date']")
            if date_el:
                log.debug("  date text: %s", date_el.get_text(strip=True))

        # If we found a date — check it's within window
        if pub_dt:
            pub_aware = pub_dt if pub_dt.tzinfo else pub_dt.replace(tzinfo=datetime.timezone.utc)
            if pub_aware < cutoff_dt():
                log.debug("  SKIP (too old %s): %s", pub_dt.strftime("%H:%M"), title[:60])
                return None
        else:
            # No date found — skip to be safe (avoids reposting old news)
            log.debug("  SKIP (no date): %s", title[:60])
            return None

        # Full-res image: prefer og:image
        image_url: str | None = None
        og_img = soup.find("meta", property="og:image")
        if og_img and og_img.get("content"):
            image_url = str(og_img["content"])

        if not image_url:
            for sel in [".news-text img", ".article-content img", "article img"]:
                img = soup.select_one(sel)
                if img:
                    src = img.get("src") or img.get("data-src") or ""
                    if src and not src.endswith(".gif"):
                        image_url = ("https://football.ua" + src) if src.startswith("/") else src
                        break

        # Body text
        body = ""
        for sel in [".news-text", ".article-content", "article"]:
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
            "source":    "football.ua",
        }
    except Exception as e:
        log.warning("fetch_football_ua_article error %s: %s", url, e)
        return None


# ─── SOURCE 2: ONEFOOTBALL ───────────────────────────────────────────────────
OF_HOME = "https://onefootball.com/en/home"

async def _playwright_get(url: str, wait: int = 3000) -> str:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        page = await browser.new_page(user_agent=HEADERS["User-Agent"])
        try:
            await page.goto(url, wait_until="networkidle", timeout=35_000)
            await page.wait_for_timeout(wait)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
            await page.wait_for_timeout(1500)
            html = await page.content()
        finally:
            await browser.close()
    return html


async def fetch_onefootball_links() -> list[str]:
    """Return up to 15 unique article URLs from OneFootball home page."""
    log.info("Fetching onefootball.com home…")
    try:
        html = await _playwright_get(OF_HOME, wait=4000)
    except Exception as e:
        log.error("Could not load OneFootball home: %s", e)
        return []

    soup      = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    links: list[str] = []

    for a in soup.find_all("a", href=re.compile(r"/en/news/")):
        href = a.get("href", "")
        url  = urljoin("https://onefootball.com", href)
        if url not in seen:
            seen.add(url)
            quick = a.get_text(" ", strip=True)[:150]
            if not is_skip(quick):
                links.append(url)
        if len(links) >= 15:
            break

    log.info("  onefootball: %d candidate links", len(links))
    return links


async def fetch_onefootball_article(url: str) -> dict | None:
    """Load a OneFootball article page. Returns dict or None to skip."""
    try:
        html = await _playwright_get(url, wait=2500)
    except Exception as e:
        log.warning("Could not load OF article %s: %s", url, e)
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Title
    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = str(og_title["content"]).strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""
    if not title or is_skip(title):
        return None

    # Publish date
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

    # Date filter
    if pub_dt:
        pub_aware = pub_dt if pub_dt.tzinfo else pub_dt.replace(tzinfo=datetime.timezone.utc)
        if pub_aware < cutoff_dt():
            log.debug("  SKIP (too old %s): %s", pub_dt.strftime("%H:%M"), title[:60])
            return None
    # If no date found — allow through (OneFootball doesn't always expose date)

    # Full-res image
    image_url: str | None = None
    og_img = soup.find("meta", property="og:image")
    if og_img and og_img.get("content"):
        image_url = str(og_img["content"])
    if not image_url:
        for sel in ["article img", "main img", "[class*='article'] img"]:
            img = soup.select_one(sel)
            if img:
                src = img.get("src") or img.get("data-src") or ""
                if src and "logo" not in src.lower():
                    image_url = ("https:" + src) if src.startswith("//") else src
                    break

    # Body
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

    log.info("Pipeline started. Window: last %d min.", MINUTES_WINDOW)

    # ── 1. Football.ua — «ГОЛОВНЕ ЗА ДОБУ» ──────────────────────────────────
    fu_links = fetch_football_ua_links()
    new_fu   = [u for u in fu_links if not is_published(conn, u)]
    log.info("football.ua: %d new links to check", len(new_fu))

    for url in new_fu:
        article = fetch_football_ua_article(url)
        if not article:
            # article returned None — either too old or no date found
            # DON'T mark as published — re-check next cycle in case date appears
            continue
        article["summary"] = ai_summarise(
            article["title"], article.get("body", ""), translate_to_uk=False
        )
        await post_to_telegram(bot, article, channel)
        mark_published(conn, url, article["title"])
        await asyncio.sleep(2)

    # ── 2. OneFootball ───────────────────────────────────────────────────────
    of_links  = await fetch_onefootball_links()
    new_of    = [u for u in of_links if not is_published(conn, u)]
    log.info("onefootball: %d new links to check", len(new_of))

    posted_of = 0
    for url in new_of:
        article = await fetch_onefootball_article(url)
        if not article:
            # too old or skip keyword — don't save, re-check next cycle
            continue
        article["summary"] = ai_summarise(
            article["title"], article.get("body", ""), translate_to_uk=True
        )
        await post_to_telegram(bot, article, channel)
        mark_published(conn, url, article["title"])
        posted_of += 1
        await asyncio.sleep(3)
        if posted_of >= 10:
            break

    log.info("onefootball: %d posted. Pipeline done.", posted_of)
    conn.close()


if __name__ == "__main__":
    asyncio.run(run_pipeline())
