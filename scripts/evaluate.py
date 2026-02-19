"""
Evaluation: run test queries and compute precision/recall.
Run from project root: python -m scripts.evaluate
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import process_query


# Example test cases: query -> expected paper stems (e.g. PMC id) that should appear in top results
TEST_CASES = [
    {
        "query": "What papers discuss theta wave activity in sleep?",
        "expected_papers": [],  # Fill with e.g. ["PMC123", "PMC456"] after you have known-good data
    },
    {
        "query": "EEG seizure detection",
        "expected_papers": [],
    },
]


def evaluate_retrieval(retrieved_sources: list[str], expected: list[str]) -> tuple[float, float]:
    """Precision and recall. expected can be empty (skip that metric)."""
    retrieved_set = set(retrieved_sources)
    expected_set = set(expected)
    if not expected_set:
        return 0.0, 0.0
    if not retrieved_set:
        return 0.0, 0.0
    tp = len(retrieved_set & expected_set)
    precision = tp / len(retrieved_set)
    recall = tp / len(expected_set)
    return precision, recall


def main():
    for test in TEST_CASES:
        query = test["query"]
        expected = test.get("expected_papers", [])
        results = process_query(query=query)
        text_results = results.get("text_results", [])
        retrieved = [r.get("metadata", {}).get("source", r.get("id", "")) for r in text_results]
        precision, recall = evaluate_retrieval(retrieved, expected)
        print(f"Query: {query[:60]}...")
        print(f"  Retrieved: {retrieved[:5]}")
        if expected:
            print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}")
        else:
            print("  (No expected set; add expected_papers to TEST_CASES for metrics)")
        print()


if __name__ == "__main__":
    main()
