# e1_drlocal

[日本語](README-ja.md)

*This project is being developed with the assistance of Google Antigravity.*

## Project Overview
**e1_drlocal** is a multi-agent, autonomous research system built on CrewAI Flows.
Given a topic, multiple AI agents collaborate to automatically execute: "formulating a research plan," "parallel information gathering," "data quality evaluation and proposing additional methods," "drafting report outlines," and "writing and refining the final Markdown report."

Previously built on LangGraph (StateGraph), it has now been refactored to an architecture using CrewAI flow control (`@start`, `@listen`, `@router`).

## Key Features

- **Multi-Agent Collaboration**:
  - **Scout / Planner**: Formulates research plans and search queries.
  - **Worker / Researcher**: Gathers information from the Web in parallel using multiple queries simultaneously.
  - **Commander / Reviewer**: Strictly evaluates the quality of gathered data, and if insufficient, suggests additional research dimensions and recommended queries back to the Planner.
  - **Writer & Editor**: Drafts each chapter in parallel based on the outline, then integrates them. In advanced modes, the Editor executes a refinement and rewrite loop for significant quality improvements.

- **Hybrid Model Support**:
  - Run with local lightweight models (e.g., Gemma3 / Qwen2.5 via Ollama) using the `--light` flag.
  - Run with cloud models (via OpenRouter) using the `--online` flag, with automatic adjustments based on API limits and context length.

- **Advanced Mode**:
  - By enabling the `--advanced` flag (or running in online mode), Pydantic-based structured extraction, existing data synthesis/compression (Synthesizer), and a final report quality audit with automatic correction (Editor refinement loop) are activated, achieving overwhelming quality improvements.

## Setup & Configuration

### 1. Prerequisites
- **Python**: 3.10 to < 3.14
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or pip

### 2. Installation
Clone the repository and install dependencies.

```bash
# Using uv
cd e1_drlocal
uv sync
```

### 3. Environment Variables
Create a `.env` file in the project root or `e1_drlocal/` directory and set the necessary API keys.

```ini
# Required for cloud models (--online)
OPENROUTER_API_KEY="your_openrouter_api_key"

# (Optional) If using Google GenAI, etc.
# GOOGLE_API_KEY="your_google_api_key"
```

### 4. Local LLM Setup
If running locally with the `--light` flag, install [Ollama](https://ollama.ai/) and pull the following models.

```bash
ollama pull gemma3n:e2b
ollama pull qwen2.5:3b
```

## Directory Structure

```text
e1_drlocal/ (Repository Root)
├── e1_drlocal/ (Project Root)
│   ├── src/e1_drlocal/
│   │   ├── main.py        # Main execution flow using CrewAI Flows
│   │   ├── crew.py        # Agent and task instance management
│   │   ├── state.py       # Pydantic-based State schema definitions
│   │   ├── config/        # YAML definition files for agents and tasks
│   │   └── constants.py   # Constants and prompt configurations
│   ├── pyproject.toml     # Package and script dependencies/build settings (uv/hatchling)
├── research_reports/      # Destination for generated final Markdown reports
├── plan-logs/             # Archive directory for AI execution plans and logs
└── README.md              # This document
```

## Change History

This project is continuously expanded by an AI and human pair. Key recent changes include:

1. **Initial Build**: Committed the initial structure to receive topics from clients and conduct research.
2. **Migration from LangGraph to CrewAI**: Migrated state management to Pydantic, implemented YAML-based agent definitions and CrewAI flow architecture (`DeepResearchFlow`).
3. **Advanced Hybrid Deep Research**: Introduced Pydantic structured extraction via the `--advanced` flag, Synthesizer data compression to prevent local token overflow, and logic to leverage high-quality cloud models via OpenRouter.
4. **Git Management Optimization**: Added ignore settings for environment files like `.venv`.
5. **Project Rename**: Unified the project and repository names to `e1_drlocal`.

## Usage

Set up and run the project using a package manager like `uv`.

```bash
# Basic start (interactive topic input)
uv run kickoff

# Example with options
# Research a specific topic using advanced features via the cloud (OpenRouter)
uv run kickoff --online --topic "Latest trends in Large Language Models (2026)"

# Run with local lightweight models
uv run kickoff --light --topic "About local LLM architectures"
```