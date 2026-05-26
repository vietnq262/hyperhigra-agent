"""Preprocess datasets with coreference resolution and entity extraction."""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Iterable, List

from src.ner.ner import NERModel
from src.utils import load_config, load_json, save_json


def _iter_examples(dataset) -> Iterable[dict]:
    if isinstance(dataset, list):
        return dataset
    if isinstance(dataset, dict):
        for key in ("data", "examples", "questions"):
            if isinstance(dataset.get(key), list):
                return dataset[key]
    return []


def _extract_documents(dataset) -> List[dict]:
    documents: List[dict] = []
    for example_index, example in enumerate(_iter_examples(dataset)):
        for doc_index, doc in enumerate(example.get("documents", example.get("context", []))):
            if isinstance(doc, dict):
                text = str(doc.get("text") or doc.get("passage") or doc.get("content") or "")
                title = doc.get("title")
            elif isinstance(doc, list) and len(doc) >= 2:
                title = doc[0]
                text = " ".join(doc[1]) if isinstance(doc[1], list) else str(doc[1])
            else:
                title = None
                text = str(doc)
            if not text.strip():
                continue
            documents.append(
                {
                    "passage_id": f"doc-{example_index}-{doc_index}",
                    "title": title,
                    "text": text,
                }
            )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = load_json(config["data"]["dataset_path"])
    ner = NERModel()

    documents = _extract_documents(dataset)
    coref_documents = []
    er_data = []
    usage_counter: Counter[str] = Counter()

    for doc in documents:
        resolved = ner.resolve_coreferences(doc["text"])
        entities = ner.extract_from_doc(resolved)
        usage_counter.update(entities)
        coref_doc = {**doc, "text": resolved, "entities": entities}
        coref_documents.append(coref_doc)
        er_data.append({"passage_id": doc["passage_id"], "entities": entities})

    save_json(config["data"]["documents_path"], documents)
    save_json(config["data"]["coref_document_path"], coref_documents)
    save_json(config["data"]["er_path"], er_data)
    save_json(config["data"]["er_usage_path"], dict(usage_counter))
    print(f"processed_documents={len(documents)}")


if __name__ == "__main__":
    main()
