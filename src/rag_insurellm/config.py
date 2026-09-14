import os
from pathlib import Path

from dotenv import load_dotenv
from tenacity import wait_exponential


def _find_root() -> Path:
    if env := os.environ.get("RAG_ROOT"):
        return Path(env).resolve()
    markers = ("pyproject.toml", "data/tests.jsonl")
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents, Path.cwd()]:
        if any((candidate / m).exists() for m in markers):
            return candidate
    return Path.cwd()


ROOT = _find_root()
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
