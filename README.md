
# 🩺 Med-KGMA: A Novel AI-Driven Medical Support System Leveraging Knowledge Graphs and Medical Advisors

This repository contains the implementation of **Med-KGMA**, a comprehensive medical AI system that integrates the SequentialRotatE knowledge graph embedding model and a Mixture-of-Medical-Advisors (MoMA) framework. This system is designed for intelligent medical diagnosis, treatment recommendations, and healthcare provider suggestions.

---

## 📘 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage Instructions](#-usage-instructions)
- [Model Components](#-model-components)
- [Example Queries](#-example-queries)
- [Disclaimer](#️-disclaimer)
- [Acknowledgements](#-acknowledgements)

---

## 🧠 Overview

**Med-KGMA** builds an intelligent medical assistant by combining symbolic reasoning with neural embeddings and expert models. It aims to provide:

- Accurate medical diagnoses based on user-reported symptoms  
- Context-aware treatment recommendations  
- Personalized doctor recommendations

---

## 🏗️ System Architecture

The system is composed of the following modules:

### 1. Medical Knowledge Graph Construction

- **Data Source:** Public medical datasets from Hugging Face  
- **Entity & Relation Extraction:** Performed using LangChain's `LLMGraphTransformer` and large language models  
- **Graph Structure:** Implemented using the `NetworkX` library to construct a directed heterogeneous graph with entities such as `Disease`, `Symptom`, `Treatment`, and `Doctor`

### 2. Knowledge Graph Embedding: Sequential RotatE

- The knowledge graph is embedded using the **Sequential RotatE** model, a temporal extension of the RotatE model that captures sequential and relational patterns  
- Trained embeddings provide high-quality vector representations of medical concepts and enable semantic similarity matching during inference

### 3. Enhanced Medical Diagnosis

The `diagnose_enhanced()` function uses both:

- Graph-based reasoning (neighborhood and path search)  
- Embedding-based similarity (from Sequential RotatE)  

This hybrid approach increases diagnostic accuracy, especially in cases with incomplete or vague symptom inputs

### 4. Mixture-of-Medical-Advisors (MoMA)

A modular ensemble system comprising three expert models:

- Diagnosis Expert  
- Treatment Recommendation Expert  
- Doctor Recommendation Expert  

A **gating network** (using Sentence Transformers) routes the user query to the most appropriate expert by computing cosine similarity between the query and expert descriptions

---

## ⚙️ Installation

Ensure you are using **Google Colab** or a compatible Python environment.

```bash
pip install langchain
pip install networkx
pip install transformers
pip install sentence-transformers
pip install spacy
pip install google-generativeai
python -m spacy download en_core_web_sm
```

Also install and load the **Clinical NER model**:

```bash
pip install https://huggingface.co/Clinical-AI-Apollo/Medical-NER/resolve/main/clinical_ai_apollo_medical_ner.whl
```

---

## 🚀 Usage Instructions

### 1. Set API Key

Set your API key for Google Generative AI:

```python
import os
os.environ["GOOGLE_API_KEY"] = "your-api-key"
```

### 2. Run the Notebook

Follow the Colab notebook sequentially to:

- Load data  
- Construct the knowledge graph  
- Train the Sequential RotatE model  
- Initialize the MoMA framework

### 3. Ask Medical Questions

Use the main query function:

```python
medical_moe_system("I am feeling fatigue, weakness, and pale skin. What could be the issue?")
```

This function will:

- Identify relevant symptoms  
- Query the knowledge graph and embeddings  
- Route the query through the appropriate expert in MoMA  
- Return diagnosis, treatment, or doctor recommendations

---

## 🧩 Model Components

Med-KGMA integrates multiple AI components:

- **LLMGraphTransformer** for medical entity and relation extraction  
- **Sequential RotatE** for embedding the medical knowledge graph  
- **MoMA** (Mixture-of-Medical-Advisors) for modular expert reasoning  
- **Gating Network** using Sentence Transformers for expert selection  
- **Clinical NER** model for high-quality entity extraction from raw medical input

---

## 🔍 Example Queries

```python
medical_moe_system("What are some common treatments for asthma?")
medical_moe_system("Recommend a doctor for severe skin allergy.")
medical_moe_system("I feel feverish with joint pain. What might I have?")
```

---

## ⚠️ Disclaimer

This tool is intended for **research and educational purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. For any health concerns, always consult a qualified healthcare provider. In case of emergency, contact emergency services immediately.

---

## 🙏 Acknowledgements

- [Hugging Face](https://huggingface.co) for open-access medical datasets  
- [LangChain](https://www.langchain.com) for enabling LLM-driven entity and relation extraction  
- [NetworkX](https://networkx.org) for graph representation  
- [Clinical-AI-Apollo/Medical-NER](https://huggingface.co/Clinical-AI-Apollo/Medical-NER) for named entity recognition in medical text  
- [Google Generative AI](https://ai.google) for large language model inference  
- [Sentence Transformers](https://www.sbert.net) for implementing the gating network in MoMA
