"""Print remaining OpenRouter credit (reads OPENROUTER_API_KEY from .env). Costs nothing."""
import os

import httpx
from dotenv import load_dotenv

load_dotenv(".env")
d = httpx.get("https://openrouter.ai/api/v1/credits",
              headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=30).json()["data"]
print(f"{d['total_credits'] - d['total_usage']:.2f}")
