# dynamic-rag-oilspill
Code and dataset for our paper: "Scientific Discovery via Dynamic RAG Architecture: Gemma Based Local Framework for up-to-date Trend Synthesis Identification on Oil Spill in Naval Research"
# Scientific Discovery via Dynamic RAG Architecture

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-orange.svg)
![GPU](https://img.shields.io/badge/GPU-NVIDIA_RTX_A4000-green.svg)

> **Official repository for the paper:** *"Scientific Discovery via Dynamic RAG Architecture: Gemma Based Local Framework for up-to-date Trend Synthesis Identification on Oil Spill in Naval Research"*

## About The Project

Traditional literature reviews are often hindered by the static nature of data and information explosion. This project introduces a **Retrieval-Augmented Generation (RAG) framework** designed specifically for scientific mining in the maritime domain, focusing on marine oil spill detection, monitoring, and transportation safety.

To completely eliminate data leakage risks and ensure maximum privacy, the entire pipeline operates on a **100% local environment (on-premise)**. We utilize **gemma3:27B** via the Ollama engine, combined with FAISS vector databases and Hugging Face embeddings, to perform dynamic trend synthesis, identify research gaps, and conduct zero-hallucination academic analysis on a curated dataset of 154 full-text PDF documents.

### Key Features
* **Fully Local Execution:** No third-party API calls; complete data privacy.
* **Zero-Hallucination Mechanism:** Evaluated using the Ragas framework (Faithfulness and Answer Relevancy metrics).
* **Semantic Retrieval:** Uses `all-MiniLM-L6-v2` embeddings and FAISS for highly accurate context filtering.
* **Automated Chunking:** Intelligent document processing via LangChain's `RecursiveCharacterTextSplitter` (1000 chunk size, 200 overlap).

## Hardware Requirements

To replicate the experiments and run the Gemma 3 (27B) model efficiently, the following (or equivalent) hardware is recommended:
* **GPU:** NVIDIA RTX A4000 (16GB VRAM) or higher.
* **RAM:** Minimum 32 GB system memory.
* **Storage:** SSD with at least 50 GB of free space for vector databases, PDF datasets, and LLM weights.

## Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/](https://github.com/)KMDR82/dynamic-rag-oilspill.git
cd dynamic-rag-oilspill
