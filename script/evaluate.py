"""Evaluate prediction files with EM and token-level F1."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Iterable, List, Tuple

from src.utils import load_config, load_json, save_json


_TOKEN_RE = re.compile(r"\w+")


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall((text or "").lower()))


def _tokens(text: str) -> List[str]:
    return _normalize(text).split()


def exact_match(prediction: str, gold: str) -> float:
    return float(_normalize(prediction) == _normalize(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _iter_examples(dataset) -> Iterable[dict]:
    if isinstance(dataset, list):
        return dataset
    if isinstance(dataset, dict):
        for key in ("data", "examples", "questions"):
            if isinstance(dataset.get(key), list):
                return dataset[key]
    return []


def _pair_predictions_with_gold(predictions, dataset) -> List[Tuple[str, str]]:
    prediction_map = {str(item.get("id")): item.get("prediction", "") for item in predictions}
    pairs = []
    for index, example in enumerate(_iter_examples(dataset)):
        example_id = str(example.get("id", index))
        gold = example.get("answer", "")
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        pairs.append((prediction_map.get(example_id, ""), str(gold)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    predictions = load_json(config["qa"]["pred_path"])
    dataset = load_json(config["data"]["dataset_path"])
    pairs = _pair_predictions_with_gold(predictions, dataset)

    em = sum(exact_match(pred, gold) for pred, gold in pairs) / max(len(pairs), 1)
    f1 = sum(token_f1(pred, gold) for pred, gold in pairs) / max(len(pairs), 1)
    summary = {"exact_match": em, "token_f1": f1, "count": len(pairs)}

    print("metric         value")
    print(f"exact_match    {em:.4f}")
    print(f"token_f1       {f1:.4f}")
    save_json(config["evaluation"]["eval_path"], summary)


if __name__ == "__main__":
    main()
