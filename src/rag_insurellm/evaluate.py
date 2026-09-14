import json
import math
from collections import defaultdict

from litellm import completion
from pydantic import BaseModel, Field

from rag_insurellm.config import EVAL_RESULTS_PATH, JUDGE_MODEL, TEST_FILE
from rag_insurellm.generate import answer_question
from rag_insurellm.models import TestQuestion
from rag_insurellm.retrieve import fetch_context


class RetrievalEval(BaseModel):
    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )


def load_tests() -> list[TestQuestion]:
    tests = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            tests.append(TestQuestion(**data))
    return tests


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    keyword_lower = keyword.lower()
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]
    ]
    dcg = calculate_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    retrieved_docs = fetch_context(test.question)
    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    ndcg_scores = [calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0
    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    generated_answer, retrieved_docs = answer_question(test.question)
    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
        },
    ]
    judge_response = completion(
        model=JUDGE_MODEL, messages=judge_messages, response_format=AnswerEval
    )
    answer_eval = AnswerEval.model_validate_json(judge_response.choices[0].message.content)
    return answer_eval, generated_answer, retrieved_docs


def evaluate_all_retrieval():
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test)
        progress = (index + 1) / total_tests
        yield test, result, progress


def evaluate_all_answers():
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_answer(test)[0]
        progress = (index + 1) / total_tests
        yield test, result, progress


def load_eval_results() -> dict:
    if EVAL_RESULTS_PATH.exists():
        return json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    return {}


def save_eval_results(patch: dict) -> None:
    data = load_eval_results()
    data.update(patch)
    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Saved eval results -> {EVAL_RESULTS_PATH}")
    print(json.dumps(patch, indent=2, ensure_ascii=False))


def run_retrieval_benchmark(on_progress=None) -> dict:
    total_mrr = total_ndcg = total_coverage = 0.0
    category_mrr = defaultdict(list)
    count = 0
    for test, result, prog_value in evaluate_all_retrieval():
        count += 1
        total_mrr += result.mrr
        total_ndcg += result.ndcg
        total_coverage += result.keyword_coverage
        category_mrr[test.category].append(result.mrr)
        if on_progress:
            on_progress(prog_value, count)

    return {
        "mrr": total_mrr / count,
        "ndcg": total_ndcg / count,
        "keyword_coverage": total_coverage / count,
        "by_category": [
            {"Category": cat, "Average MRR": sum(scores) / len(scores)}
            for cat, scores in sorted(category_mrr.items())
        ],
    }


def run_answer_benchmark(on_progress=None) -> dict:
    total_accuracy = total_completeness = total_relevance = 0.0
    by_acc = defaultdict(list)
    count = 0
    for test, result, prog_value in evaluate_all_answers():
        count += 1
        total_accuracy += result.accuracy
        total_completeness += result.completeness
        total_relevance += result.relevance
        by_acc[test.category].append(result.accuracy)
        if on_progress:
            on_progress(prog_value, count)

    return {
        "accuracy": total_accuracy / count,
        "completeness": total_completeness / count,
        "relevance": total_relevance / count,
        "by_category": [
            {"Category": cat, "Average Accuracy": sum(scores) / len(scores)}
            for cat, scores in sorted(by_acc.items())
        ],
    }
