# e1_drlocal
 
[日本語版の説明はこちら](README-ja.md)

*This project is being developed in collaboration with AI, utilizing Google Antigravity.*

## Project Overview
**e1_drlocal** is a research system built on CrewAI Flows.
Given a topic, multiple AI agents collaborate to automatically execute: "formulating a research plan," "parallel information gathering," "data quality evaluation and proposing additional methods," "drafting report outlines," and "writing and refining the final Markdown report."

Previously built on LangGraph (StateGraph), it has now been refactored to an architecture using CrewAI flow control (`@start`, `@listen`, `@router`).

## Key Features

- **Multi-Agent Collaboration**:
  - **Scout / Planner**: Formulates research plans and search queries.
  - **Worker / Researcher**: Gathers information from the Web in parallel.
  - **Commander / Reviewer**: Evaluates the quality of gathered data, and if insufficient, suggests additional research dimensions and recommended queries back to the Planner.
  - **Writer & Editor**: Drafts each chapter in parallel based on the outline, then integrates them. In advanced modes, it also executes refinement and rewrite processing.

- **Hybrid Model Support**:
  - Run with local lightweight models (e.g., Gemma3 / Qwen2.5 via Ollama) using the `--light` flag.
  - Run with cloud models (via OpenRouter) using the `--online` flag.

- **Advanced Mode**:
  - By enabling the `--advanced` flag (or running in online mode), Pydantic-based structured output, existing data compression, and a final report quality audit with automatic correction are activated, aiming for high-quality research output.

## Usage

*Setup instructions can be found in the [section at the bottom of the page](#setup--configuration).*

The project is set up and run using a package manager like `uv`.

```bash
# Basic start (interactive topic input)
uv run kickoff

# Example with options
# Research a specific topic using advanced features via the cloud (OpenRouter)
uv run kickoff --online --topic "Latest trends in Large Language Models (2026)"

# Run with local lightweight models
uv run kickoff --light --topic "About local LLM architectures"
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

1. **Initial Build**: Initial structure committed to receive topics and conduct research.
2. **Migration to CrewAI**: Migrated state management to Pydantic and implemented CrewAI flow architecture.
3. **Advanced Hybrid Deep Research**: Introduced Advanced Mode (structured extraction, data compression, and cloud model integration).
4. **Project Naming**: Unified the project and repository names to `e1_drlocal` and released.

## Setup & Configuration

### 1. Prerequisites (For Windows 10/11)
To build the environment from scratch, you need the following:
- **Git**: [Install](https://git-scm.com/downloads)
- **Microsoft C++ Build Tools**: Required for some AI-related Python packages. [Install](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- **Docker Desktop**: Necessary for running the local search engine (SearxNG). [Install](https://www.docker.com/products/docker-desktop/)
- **Package Manager [uv]**: Open PowerShell and run:
  ```powershell
  Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1 -UseBasicParsing | Invoke-Expression
  ```

### 2. SearxNG Setup (Local Search Engine)
Information gathering agents rely on a local SearxNG instance. Start it using Docker on port 8081:
```bash
docker run -d -p 8081:8080 -e "BASE_URL=http://localhost:8081/" -e "INSTANCE_NAME=e1_searxng" searxng/searxng
```

### 3. Installation
Clone the repository and install dependencies:

```bash
cd e1_drlocal
uv sync
```
*Note for Linux/macOS users: If you encounter a resolution error during `uv sync`, please update `tool.uv.required-environments` in `pyproject.toml` or remove that section.*

### 4. Environment Variables
Create a `.env` file in the project root or `e1_drlocal/` directory:

```ini
# Required for cloud models (--online)
OPENROUTER_API_KEY="your_openrouter_api_key"

# (Optional) If using Google GenAI, etc.
# GOOGLE_API_KEY="your_google_api_key"
```

### 5. Local LLM Setup
To run models locally, install [Ollama](https://ollama.ai/) and pull the necessary models:

# Default Profile
```bash
ollama pull llama4-scout-q2:latest
ollama pull gpt-oss:20b
ollama pull nemotron-3-nano:4b
```

# Lightweight Profile (Recommended)
```bash
ollama pull gemma3n:e2b
ollama pull qwen2.5:3b
```

You can also customize the models used in `src/e1_drlocal/constants.py` to suit your environment.

---

**Setup is complete!** You can start your research as described in the [Usage](#usage) section.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
