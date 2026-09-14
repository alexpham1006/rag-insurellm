from pathlib import Path

from dotenv import load_dotenv
from tenacity import wait_exponential

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)

KNOWLEDGE_BASE_PATH = ROOT / "data" / "knowledge-base"
CHROMA_PATH = ROOT / "data" / "chroma"
TEST_FILE = ROOT / "data" / "tests.jsonl"
EVAL_RESULTS_PATH = ROOT / "data" / "eval_results.json"

COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "text-embedding-3-large"
MODEL = "openai/gpt-4.1-nano"
JUDGE_MODEL = "gpt-4.1-nano"

RETRIEVAL_K = 20
FINAL_K = 10
WORKERS = 3
AVERAGE_CHUNK_SIZE = 100

WAIT = wait_exponential(multiplier=1, min=10, max=240)
