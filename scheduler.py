"""
Scheduler для Railway
======================
Railway тримає процес живим — просто запустіть цей файл як start command:
    python scheduler.py

Інтервал задається змінною середовища INTERVAL_MINUTES (за замовчуванням 60).
"""

import asyncio
import logging
import os
from bot import run_pipeline

INTERVAL_MINUTES = int(os.environ.get("INTERVAL_MINUTES", "60"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scheduler")


async def main() -> None:
    log.info("Scheduler started. Interval: %d min.", INTERVAL_MINUTES)
    while True:
        log.info("▶ Running pipeline…")
        try:
            await run_pipeline()
        except Exception as e:
            log.error("Pipeline crashed: %s", e, exc_info=True)
        log.info("⏳ Next run in %d min.", INTERVAL_MINUTES)
        await asyncio.sleep(INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    asyncio.run(main())
