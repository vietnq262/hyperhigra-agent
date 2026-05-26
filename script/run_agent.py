"""Run the HyperHiGraAgent QA pipeline."""

from __future__ import annotations

import argparse
from typing import Iterable, List

from src.agents.librarian import Librarian
from src.agents.question_clarifier import QuestionClarifier
from src.agents.question_decomposer import QuestionDecomposer
from src.agents.seeker import Seeker
from src.hgraph.hgraph import Hypergraph
from src.llm_client.openai_client import OpenAIClient
from src.ner.ner import NERModel
from src.prompts.prompts import QA_ADAPTIVE_PROMPT, QA_SINGLE_STEP_PROMPT
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.hyperhigra_retriever import HyperHiGraRetriever
from src.utils import load_config, load_json, save_json


def _iter_examples(dataset) -> Iterable[dict]:
    if isinstance(dataset, list):
        return dataset
    if isinstance(dataset, dict):
        for key in ("data", "examples", "questions"):
            if isinstance(dataset.get(key), list):
                return dataset[key]
    return []


def _answer_question(llm_client: OpenAIClient, question: str, context: List[dict], adaptive: bool) -> str:
    context_text = "\n\n".join(item.get("text", "") for item in context)
    prompt = (QA_ADAPTIVE_PROMPT if adaptive else QA_SINGLE_STEP_PROMPT).format(
        question=question,
        context=context_text,
    )
    answer = llm_client.complete_sync(prompt).strip()
    if answer and answer != "LLM client is not configured.":
        return answer
    return context[0].get("text", "") if context else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = load_json(config["data"]["dataset_path"])

    llm_cfg = config.get("llm_client", {})
    llm_client = OpenAIClient(
        api_key=llm_cfg.get("api_key", ""),
        model_name=llm_cfg.get("model_name", "gpt-4o-mini"),
        batch_size=int(llm_cfg.get("batch_size", 200)),
    )
    ner = NERModel()
    dense_retriever = DenseRetriever()
    dense_retriever.load_index(config["data"]["embedding_path"])

    retrieval_mode = config.get("qa", {}).get("retrieval_mode", "hyperhigra_retriever")
    if retrieval_mode == "hybrid_retriever":
        retriever = HybridRetriever(dense_retriever, dense_retriever.documents)
        retrieve = lambda q, entities, top_k: retriever.retrieve(q, top_k)
    else:
        hgraph = Hypergraph.load(config["data"]["hypergraph_path"])
        retriever = HyperHiGraRetriever(config, hgraph, dense_retriever)
        retrieve = retriever.retrieve

    clarifier = QuestionClarifier(llm_client)
    decomposer = QuestionDecomposer(llm_client)
    seeker = Seeker(config, type("RetrieverAdapter", (), {"retrieve": staticmethod(retrieve)})(), ner, llm_client)
    librarian = Librarian(config, llm_client)

    adaptive = config.get("qa", {}).get("reasoning_mode", "adaptive_reasoning") == "adaptive_reasoning"
    predictions = []

    for index, example in enumerate(_iter_examples(dataset)):
        question = str(example.get("question") or example.get("query") or "")
        clarified = clarifier.clarify(question)
        sub_questions = decomposer.decompose(clarified) if adaptive else [clarified]

        librarian.reset()
        for sub_question in sub_questions:
            new_passages = seeker.seek(sub_question, librarian.get_context())
            librarian.update(new_passages)
            if adaptive and librarian.is_sufficient(question):
                break

        context = librarian.get_context()
        answer = _answer_question(llm_client, clarified, context, adaptive=adaptive)
        predictions.append(
            {
                "id": example.get("id", index),
                "question": question,
                "prediction": answer,
                "context": context,
            }
        )

    save_json(config["qa"]["pred_path"], predictions)
    print(f"predictions={len(predictions)}")


if __name__ == "__main__":
    main()
