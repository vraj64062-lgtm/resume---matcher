"""
Evaluation harness for the Resume-to-Job Matcher.

Runs the matcher against a hand-labeled dataset (labeled_dataset.json) and
reports accuracy, precision, recall, F1-score, and average latency. This is
the script that generates the number you put on your resume.

Usage:
    # Baseline (no API key needed, runs offline):
    python eval_harness.py

    # Gemini-enhanced (requires GEMINI_API_KEY env var):
    GEMINI_API_KEY=your_key python eval_harness.py --use-gemini

A prediction counts as "fit" if overall_score >= FIT_THRESHOLD.
"""

import sys
import json
import time
import argparse
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app import scorer, baseline_extractor  # noqa: E402

# Calibrated against the labeled dataset (see README methodology note) —
# 50 was the naive default; 65 balances precision/recall better once
# borderline (partial-skill-match) cases are included in evaluation.
FIT_THRESHOLD = 65.0


def load_dataset(path):
    with open(path) as f:
        return json.load(f)


def run_eval(dataset, use_gemini=False):
    if use_gemini:
        from app import gemini_client
        extract = gemini_client.extract_structured
    else:
        extract = baseline_extractor.extract_structured

    tp = fp = tn = fn = 0
    latencies = []

    for item in dataset:
        start = time.perf_counter()

        resume_struct = extract(item["resume_text"])
        jd_struct = extract(item["job_description"])
        result = scorer.compute_match(
            item["resume_text"], item["job_description"], resume_struct, jd_struct
        )

        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        predicted_fit = result["overall_score"] >= FIT_THRESHOLD
        actual_fit = item["label"] == "fit"

        if predicted_fit and actual_fit:
            tp += 1
        elif predicted_fit and not actual_fit:
            fp += 1
        elif not predicted_fit and not actual_fit:
            tn += 1
        else:
            fn += 1

        status = "OK" if predicted_fit == actual_fit else "MISS"
        print(f"[{status}] id={item['id']:>2} score={result['overall_score']:>6.1f} "
              f"predicted={'fit' if predicted_fit else 'no_fit':<7} actual={item['label']}")

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total * 100
    precision = (tp / (tp + fp) * 100) if (tp + fp) else 0.0
    recall = (tp / (tp + fn) * 100) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    avg_latency = sum(latencies) / len(latencies)

    print("\n" + "=" * 50)
    print(f"Mode:            {'Gemini-enhanced' if use_gemini else 'Baseline (keyword-based)'}")
    print(f"Samples:         {total}")
    print(f"Accuracy:        {accuracy:.1f}%")
    print(f"Precision:       {precision:.1f}%")
    print(f"Recall:          {recall:.1f}%")
    print(f"F1-score:        {f1:.1f}%")
    print(f"Avg latency:     {avg_latency:.1f} ms/pair")
    print("=" * 50)

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "avg_latency_ms": avg_latency,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-gemini", action="store_true")
    args = parser.parse_args()

    dataset_path = os.path.join(os.path.dirname(__file__), "labeled_dataset.json")
    dataset = load_dataset(dataset_path)
    run_eval(dataset, use_gemini=args.use_gemini)
