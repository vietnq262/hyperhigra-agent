"""Construct HyperHiGra retrieval assets."""

from __future__ import annotations

import argparse

from src.hgraph.build_hgraph import HypergraphBuilder
from src.hgraph.ent_extraction import EntityExtractor
from src.retriever.dense_retriever import DenseRetriever
from src.utils import load_config, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    documents = load_json(config["data"]["coref_document_path"])

    extractor = EntityExtractor()
    builder = HypergraphBuilder()
    for doc in documents:
        doc["entities"] = doc.get("entities") or extractor.extract_from_doc(doc.get("text", ""))

    hgraph = builder.build(documents, extractor)
    dense_retriever = DenseRetriever()
    dense_retriever.build_index(documents, config["data"]["embedding_path"])
    hgraph.save(config["data"]["hypergraph_path"])

    print(f"num_nodes={len(hgraph.nodes)}")
    print(f"num_hyperedges={len(hgraph.hyperedges)}")


if __name__ == "__main__":
    main()
