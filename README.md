# General-RAG

A comprehensive RAG (Retrieval-Augmented Generation) application framework with OCR capabilities.

## Features

- **OCR**: Convert documents (PDF, images, etc.) to Markdown format using Marker
- **RAG Strategies**: 11 advanced RAG strategies for document retrieval and question answering
- **Interactive CLI**: User-friendly command-line interfaces with cursor navigation
- **Flexible Backends**: Support for multiple embedding models and LLM providers (Ollama)

## Project Structure

```
General-RAG/
├── OCR/                    # OCR application
│   ├── ocr_cli.py         # Interactive OCR CLI
│   └── requirements.txt   # OCR dependencies
├── RAG/                    # RAG application
│   ├── rag_cli.py         # Main RAG CLI
│   ├── embedding_loader.py # Embedding models loader
│   ├── agent_llm.py       # Ollama LLM interface
│   ├── vector_store.py    # FAISS vector database
│   ├── strategies/        # RAG strategy implementations
│   └── requirements.txt   # RAG dependencies
├── documents/              # Document storage (created on first run)
└── README.md              # This file
```

## Quick Start

### OCR Application

Convert documents to Markdown:

```bash
cd OCR
pip install -r requirements.txt
python ocr_cli.py
```

The application will:
- List all documents in the `documents/` folder
- Process documents that don't have a corresponding `.md` file
- Show which files were processed and which were skipped

### RAG Application

Run the RAG application:

```bash
cd RAG
pip install -r requirements.txt
python rag_cli.py
```

The interactive CLI will guide you through:
1. Selecting an embedding model
2. Choosing an Ollama LLM model
3. Selecting a RAG strategy
4. Choosing a documents folder
5. Asking questions and getting answers

Type `/bye` or `exit` to quit.

## Requirements

- Python 3.10+
- Ollama running locally on `http://localhost:11434`
- Marker installed for OCR functionality

## Installation

### OCR Dependencies

```bash
cd OCR
pip install -r requirements.txt
```

### RAG Dependencies

```bash
cd RAG
pip install -r requirements.txt
```

## RAG Strategies

This project implements 11 advanced RAG strategies:

1. **Re-ranking** - Two-stage retrieval with cross-encoder
2. **Agentic RAG** - Agent dynamically chooses retrieval tools
3. **Knowledge Graphs** - Graph-based retrieval (requires Neo4j)
4. **Contextual Retrieval** - Document context added to chunks
5. **Query Expansion** - Generate multiple query variations
6. **Multi-Query RAG** - Parallel searches with multiple queries
7. **Context-Aware Chunking** - Semantic boundary detection
8. **Late Chunking** - Embed full document before chunking
9. **Hierarchical RAG** - Parent-child chunk relationships
10. **Self-Reflective RAG** - Iterative refinement with self-assessment
11. **Fine-tuned Embeddings** - Domain-specific embedding models

## Documentation

Detailed documentation for each strategy is available in `RAG/docs/`.

## Contributing

This is a research project. Feel free to fork and adapt for your use case.

## License

See individual strategy files for licensing information.

