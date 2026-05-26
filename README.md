# HyperHiGraAgent

HyperHiGraAgent is a merged multi-hop QA scaffold that combines **HGRAG**'s cross-granularity hypergraph retrieval with **HiGraAgent**'s dual-agent adaptive reasoning loop.

## Why merge HGRAG and HiGraAgent?

- **From HGRAG (MF-AIR/HGRAG):** hypergraph construction, diffusion-based retrieval, dense retrieval oriented around multi-hop evidence propagation.
- **From HiGraAgent (headinthecloud6453/higra_agent):** Seeker/Librarian agent loop, question clarification and decomposition, coreference-aware preprocessing, async OpenAI client.
- **HyperHiGraAgent goal:** replace HiGraAgent's standard KG backend with HGRAG's hypergraph backend while preserving the agentic reasoning flow.

## Architecture

```text
raw corpus
   |
   v
coref resolution + NER
   |
   v
hypergraph construction
   |
   +--> dense retrieval ------------------+
   |                                      |
   +--> hypergraph diffusion              |
                    |                     |
                    +------ RRF fusion <--+
                               |
                               v
                    seeker / librarian loop
                               |
                               v
                             answer
```

## Project layout

```text
├── README.md
├── requirements.txt
├── config.yaml
├── script/
│   ├── __init__.py
│   ├── process_dataset.py
│   ├── construct_hyperhigra.py
│   ├── run_agent.py
│   └── evaluate.py
└── src/
    ├── __init__.py
    ├── hgraph/
    ├── agents/
    ├── retriever/
    ├── llm_client/
    ├── ner/
    ├── prompts/
    └── utils.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Configuration modes

| Mode | `qa.reasoning_mode` | `qa.retrieval_mode` | Purpose |
|---|---|---|---|
| Full | `adaptive_reasoning` | `hyperhigra_retriever` | Hypergraph retrieval + agent loop |
| Reasoning ablation | `single_step_reasoning` | `hyperhigra_retriever` | Retrieval only, no iterative loop |
| Retrieval ablation | `adaptive_reasoning` | `hybrid_retriever` | Agent loop with BM25+dense hybrid |

## 4-step usage

1. **Preprocess the dataset**
   ```bash
   python script/process_dataset.py --config config.yaml
   ```
2. **Construct HyperHiGra retrieval assets**
   ```bash
   python script/construct_hyperhigra.py --config config.yaml
   ```
3. **Run the adaptive agent**
   ```bash
   python script/run_agent.py --config config.yaml
   ```
4. **Evaluate predictions**
   ```bash
   python script/evaluate.py --config config.yaml
   ```

## Notes

- The hypergraph stores entities as nodes and sentence/passage groupings as hyperedges.
- The retriever fuses dense retrieval with hypergraph diffusion using Reciprocal Rank Fusion (RRF).
- The LLM client is async-first and provides a sync wrapper for script entrypoints.
- If no OpenAI API key is provided, the scaffold still runs with safe placeholder responses.

## Citations

### HGRAG

```bibtex
@inproceedings{wang2026hgrag,
  title={Cross-Granularity Hypergraph Retrieval-Augmented Generation for Multi-hop Question Answering},
  author={Wang, et al.},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

### HiGraAgent

```bibtex
@misc{higraagent2025,
  title={HiGraAgent: Hierarchical Graph Retrieval with Dual-Agent Adaptive Reasoning},
  author={headinthecloud6453 et al.},
  year={2025},
  note={Project scaffold and paper release}
}
```

## Credits

This project is inspired by and credits:

- **MF-AIR/HGRAG** for the hypergraph retrieval design.
- **headinthecloud6453/higra_agent** for the adaptive multi-agent reasoning architecture.
