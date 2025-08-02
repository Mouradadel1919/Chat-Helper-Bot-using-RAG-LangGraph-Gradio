# Code Helper Bot using RAG and LangGraph

This project implements a code assistant bot capable of generating and explaining code using a Retrieval-Augmented Generation (RAG) architecture integrated with LangGraph. The system combines embeddings, vector search, and a powerful language model to deliver accurate and context-aware code assistance.

## ChatBot



https://github.com/user-attachments/assets/83875370-608e-4db1-ab7f-bb5b8716f716



## Overview

The application supports the following capabilities:

- **Code Generation and Explanation**: Users can input code or questions to receive generated code snippets or detailed explanations.
- **Retrieval-Augmented Generation**: Relevant context is retrieved from a vector store to guide the model’s output.
- **LangGraph-Based Architecture**: Workflow orchestration and decision-making are handled through LangGraph.
- **Embeddings**: Utilizes the `BAAI/bge-large-en-v1.5` model to embed user queries and documents.
- **Vector Store**: Employs ChromaDB to store and retrieve semantically similar queries and responses.
- **Language Model**: Uses `Qwen/Qwen2.5-Coder-7B-Instruct` for high-quality code generation and explanation.
- **Evaluation**: Model performance was assessed using a subset of 10 examples from the MBPP dataset, achieving 80% accuracy while maintaining efficient GPU usage and fast inference time.
- **Deployment**: The application is deployed using Gradio's ChatBotInterface for interactive usage.

## System Architecture

A visual representation of the LangGraph application flow is provided below:

<img width="480" height="580" alt="Capture2" src="https://github.com/user-attachments/assets/30236479-cf7e-4b76-b2ed-147daf38f7ab" />


## Technology Stack

| Component       | Description                              |
|----------------|------------------------------------------|
| Embedding Model | `BAAI/bge-large-en-v1.5`                 |
| Vector Store    | ChromaDB                                 |
| Language Model  | `Qwen/Qwen2.5-Coder-7B-Instruct`         |
| Framework       | LangGraph                                |
| Deployment      | Gradio ChatBotInterface                  |
| Evaluation Data | MBPP (Mostly Basic Python Problems)      |

## Evaluation

- **Dataset**: MBPP Dataset test cases for evaluation.
- **Correctness Metric**: 90% on a selected sample of MBPP tasks.  
- **Performance Considerations**: Designed for efficient GPU utilization and optimized inference speed.


