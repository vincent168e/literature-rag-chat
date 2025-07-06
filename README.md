# README - Literature RAG Chat

## Overview

Literature RAG Chat is a retrieval-augmented generation (RAG) application designed for interacting with literary content. This Python project leverages OpenAI's language models and ChromaDB for vector storage to provide intelligent responses based on ingested literature. The system is built using the LangChain framework, which facilitates the integration of language models with retrieval components.

## Key Features

- **Retrieval-Augmented Generation**: Combines document retrieval with LLM capabilities for context-aware responses
- **ChromaDB Integration**: Uses Chroma vector database for efficient document storage and retrieval
- **OpenAI Support**: Works with OpenAI's language models for text generation

## Installation

1. Ensure you have Python 3.12 or later installed
2. Install [uv](https://github.com/astral-sh/uv), a fast Python package installer:
   ```bash
   pip install uv
   ```
3. Create and activate a virtual environment:
   ```bash
   uv venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   uv pip install -e .
   ```

## Configuration

1. Create a `.env` file in the project root
2. Add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

After installation and configuration, you can run the application to:
- Ingest literary documents into the vector database
- Query the system with literature-related questions
- Receive contextually relevant responses based on the stored content

The project demonstrates how to build a specialized chatbot that can answer questions about literary works by combining document retrieval with modern language models.

## Dependencies

The project relies on:
- LangChain and its components for the RAG pipeline
- ChromaDB for vector storage
- OpenAI for language model access
- Pydantic for data validation

For a complete list of dependencies, see `pyproject.toml`.
