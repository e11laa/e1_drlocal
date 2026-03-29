# e1_drlocal

[English](README.md)

*本プロジェクトは、 Google Antigravity を利用しつつ、AIと協働して開発されています。*

## プロジェクト概要 (Project Overview)
**e1_drlocal** は、CrewAI Flows をベースに構築されたマルチエージェント型・自律調査システムです。
与えられたトピックに対して、複数のAIエージェントが協調しながら「調査計画の策定」「並列情報収集」「データ品質評価および追加手法の提案」「レポート構成案作成」、そして「最終的なMarkdownレポートの執筆と推敲」を自動で実行します。

以前は LangGraph（StateGraph）ベースで構築されていましたが、現在は CrewAI のフロー制御（`@start`, `@listen`, `@router`）を用いたアーキテクチャにリファクタリングされています。

## 主な特徴 (Key Features)

- **マルチエージェント協調 (Multi-Agent Collaboration)**:
  - **Scout / Planner**: リサーチプランと検索クエリを立案。
  - **Worker / Researcher**: 複数クエリを同時に用いて並列にWebから情報を収集。
  - **Commander / Reviewer**: 収集されたデータの品質を厳格に評価し、不足があれば追加の調査軸や推奨クエリを提示して Planner に差し戻し。
  - **Writer & Editor**: アウトラインに基づき各章を並列で執筆、統合。さらに高度なモードでは編集長(Editor)による推敲とリライト処理を実行。

- **ハイブリッド・モデル対応 (Hybrid Model Support)**:
  - `--light` フラグによるローカルの軽量モデル（Ollama 経由の Gemma3 / Qwen2.5 等）での実行。
  - `--online` フラグによるクラウドモデル（OpenRouter経由）での実行に対応しており、API制限や文脈長に応じた自動調整が可能。

- **アドバンスド・モード (Advanced Mode)**:
  - `--advanced` フラグ（またはオンラインモード）を有効にすることで、Pydanticベースの構造化出力、既存データの圧縮・統合（Synthesizer）、最終レポートの品質監査と自動修正（Editor 推敲ループ）が作動し、圧倒的な品質向上を実現します。

## 使い方 (Usage)

*セットアップの分かりやすい手順は[ページ下部のセクション](#セットアップと環境設定-setup--configuration)にあります。*

プロジェクトは `uv` パッケージマネージャーなどを用いてセットアップし実行します。

```bash
# 基本的な起動（トピックは対話的に入力）
uv run kickoff

# オプションを指定した起動例
# クラウド(OpenRouter)経由でアドバンスド機能を利用して特定トピックを調査
uv run kickoff --online --topic "大規模言語モデルの最新動向(2026年)"

# ローカルの軽量モデルで実行
uv run kickoff --light --topic "ローカルLLMアーキテクチャについて"
```

## ディレクトリ構成 (Directory Structure)

```text
e1_drlocal/ (Repository Root)
├── e1_drlocal/ (Project Root)
│   ├── src/e1_drlocal/
│   │   ├── main.py        # CrewAI Flows によるメイン実行フロー
│   │   ├── crew.py        # エージェント・タスクのインスタンス管理
│   │   ├── state.py       # Pydanticベースの状態(State)スキーマ定義
│   │   ├── config/        # 各エージェント・タスクの YAML 定義ファイル
│   │   └── constants.py   # 定数・プロンプト設定
│   ├── pyproject.toml     # パッケージおよびスクリプトの依存関係・ビルド設定 (uv/hatchling)
├── research_reports/      # 生成された最終マークダウンレポートの保存先
├── plan-logs/             # AIの実行計画・ログのアーカイブ用ディレクトリ
└── README.md              # 当ドキュメント
```

## 開発と変更の履歴 (Change History)

本プロジェクトは継続的にAIと人間のペアで拡張されています。直近の主要な変更履歴は以下の通りです：

1. **初期構築**: クライアントからトピックを受け取り調査を行う初期構造のコミット。
2. **LangGraphからCrewAIへの移行**: 状態管理をPydanticに移行し、YAMLベースでのエージェント定義および CrewAI フローアーキテクチャ（`DeepResearchFlow`）を実装。
3. **ハイブリッド実行と高度な推敲の導入 (Advanced Hybrid Deep Research)**: `--advanced` フラグによるPydantic構造化抽出の導入、ローカルでのトークン溢れを防ぐSynthesizerデータ圧縮、OpenRouter経由での高品質クラウドモデルの活用ロジックを追加。
4. **Git管理の最適化**: `.venv` などの環境ファイルの無視設定を追加。
5. **プロジェクト改名**: プロジェクトおよびリポジトリ名を `e1_drlocal` に統一。

## セットアップと環境設定 (Setup & Configuration)

### 1. 動作環境と必須ツール (Windows 10/11 初期セットアップ用)
まっさらな Windows 環境から構築する場合、以下の準備が必要です。
- **Git**: [公式サイト](https://git-scm.com/downloads) からインストールしてください。
- **Microsoft C++ Build Tools**: 一部の AI 関連 Python パッケージは C++ コンパイルを要求します。[事前に入手](https://visualstudio.microsoft.com/visual-cpp-build-tools/) し、「C++ によるデスクトップ開発」ワークロードをインストールしてエラーを防ぎましょう。
- **Docker Desktop**: ローカルの検索エンジン (SearxNG) を稼働させるために必須です。[こちらからインストール](https://www.docker.com/products/docker-desktop/) してください。
- **パッケージマネージャー [uv]**: 以下のコマンドを PowerShell で実行してインストールします。
  ```powershell
  Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1 -UseBasicParsing | Invoke-Expression
  ```

### 2. SearxNG の起動 (ローカル検索エンジン)
情報収集エージェントはローカルの SearxNG サーバーにアクセスして Web 検索を行います。Docker Desktop を起動後、ターミナルで以下のコマンドを実行し、ポート 8081 で立ち上げてください。
```bash
docker run -d -p 8081:8080 -e "BASE_URL=http://localhost:8081/" -e "INSTANCE_NAME=e1_searxng" searxng/searxng
```

### 3. インストール (Installation)
Git でリポジトリをクローンし、依存関係をインストールします。（`uv` が Python 実行環境の構築まで自動で完了させてくれます）

```bash
# uv を使用する場合
cd e1_drlocal
uv sync
```

### 4. 環境変数の設定 (Environment Variables)
プロジェクトのルートまたは `e1_drlocal/` ディレクトリに `.env` ファイルを作成し、必要なAPIキーを設定してください。

```ini
# クラウドモデル利用時 (--online) に必要
OPENROUTER_API_KEY="your_openrouter_api_key"

# (任意) Google GenAI などを利用する場合
# GOOGLE_API_KEY="your_google_api_key"
```

### 5. ローカルLLMの準備 (Local LLM Setup)
ローカルで実行する場合は、[Ollama](https://ollama.ai/) をインストールしてください。**基本的には `--light` フラグを使用した軽量プロファイルでの実行を推奨しています**。以下のモデルを事前にプルしておいてください：

```bash
ollama pull gemma3n:e2b
ollama pull qwen2.5:3b
```

*※ ゲーミングPCなどの高性能なGPU環境がある場合のみ、フラグなしのデフォルト起動をお試しください。その場合は以下の重いモデルが必要になります:*

```bash
ollama pull llama4-scout-q2:latest
ollama pull gpt-oss:20b
ollama pull nemotron-3-nano:4b
```
