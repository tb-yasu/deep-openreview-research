# Deep OpenReview Research

AI-Powered Deep Paper Review and Analysis Agent

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## 📋 Overview

**Deep OpenReview Research** is an AI agent that automatically searches and evaluates accepted papers from conferences, ranking them based on your research interests. By combining the OpenReview API with LLMs, it extracts deep information from papers including meta reviews, review details, and acceptance reasons to support efficient paper research.

> **⚠️ Important Notice**: This tool is in the development stage, and quantitative validation of paper evaluation accuracy is a future challenge. Scoring methods and other aspects are subject to debate. AI-generated evaluation results should be used as reference information, and human final confirmation is recommended for important research decisions.

## ✨ Key Features

- 🔍 **Automatic Paper Search**: Automatically search papers from specified conferences and years
- 🤖 **Unified LLM Evaluation**: Comprehensive evaluation of relevance, novelty, impact, and practicality in a single call
- ⚡ **Parallel Processing**: Execute up to 10 concurrent LLM evaluations for ~10x faster processing
- 📊 **Scoring System**: Combined scoring using OpenReview evaluations and AI assessments
- 🔑 **Synonym Expansion**: Automatically generate synonyms for keywords to expand search scope
- 💬 **Natural Language Input**: Describe research interests in natural language
- 📝 **Detailed Reports**: Auto-generate detailed reports in Markdown format
- 🎤 **Presentation Format Display**: Automatically extract Oral/Spotlight/Poster distinctions
- 📋 **Deep Review Analysis**: Display review summaries, score averages, acceptance reasons, and author comments
- 🔄 **Dynamic Field Detection**: Automatically adapt to evaluation criteria from ICLR/NeurIPS/ICML and other conferences

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/tb-yasu/deep-openreview-research.git
cd deep-openreview-research

# 2. Create virtual environment and install packages
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Set up OpenAI API key (.env file)
cp .env.example .env
# Edit the .env file to set your actual API key

# 4. Fetch paper data (first time only, 60-90 minutes)
python fetch_all_papers.py --venue NeurIPS --year 2025

# 5. Run paper review
python run_deep_research.py \
  --venue NeurIPS \
  --year 2025 \
  --research-description "I am interested in graph generation and its applications to drug discovery"
```

## 📦 Installation

### Prerequisites

- Python 3.12 or higher
- OpenAI API key
- Internet connection (only for fetching paper data)

### Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/tb-yasu/deep-openreview-research.git
cd deep-openreview-research
```

#### 2. Create Virtual Environment

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:
- `langchain` / `langgraph` - LLM application framework
- `langchain-openai` - OpenAI integration
- `openreview-py` - OpenReview API client
- `pydantic` - Data validation
- `loguru` - Logging

#### 4. Set Up Environment Variables

**Recommended: Use .env file**

```bash
# Copy .env.example to create .env file
cp .env.example .env

# Edit the .env file to set your actual API key
# Open .env in an editor and edit as follows:
# OPENAI_API_KEY=sk-your-actual-api-key-here
```

The `.env` file is included in `.gitignore`, so it won't be accidentally committed to Git.

**Alternative: Set environment variable directly**

For temporary testing, you can set the environment variable directly:

```bash
# macOS/Linux
export OPENAI_API_KEY="your-api-key-here"

# Windows (Command Prompt)
set OPENAI_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-api-key-here"
```

**Note**: This method loses the setting when you close the terminal.

#### 5. Fetch Paper Data

```bash
python fetch_all_papers.py --venue NeurIPS --year 2025
```

**Note**: The first run takes 60-90 minutes, but subsequent runs use local cache.

## 💻 Basic Usage

### Pattern 1: Specify Research Interests in Natural Language (Recommended)

```bash
python run_deep_research.py \
  --venue NeurIPS \
  --year 2025 \
  --research-description "I have a strong interest in graph generation and its applications to drug discovery"
```

### Pattern 2: Specify with Keyword List

```bash
python run_deep_research.py \
  --venue NeurIPS \
  --year 2025 \
  --research-interests "graph generation,drug discovery,molecular design"
```

### Pattern 3: Quick Start Script

```bash
./quickstart.sh
```

## 🎛️ Command Line Options

### Required Options

| Option | Description | Example |
|--------|-------------|---------|
| `--venue` | Conference name | `NeurIPS`, `ICML`, `ICLR` |
| `--year` | Year | `2025` |
| `--research-description` or `--research-interests` | Research interests | See below |

### Evaluation Criteria Options

| Option | Default | Description |
|--------|---------|-------------|
| `--min-relevance-score` | 0.2 | Minimum relevance score (0.0-1.0) |
| `--top-k` | 100 | Number of top papers for LLM evaluation |
| `--max-papers` | 9999 | Maximum number of papers to search |
| `--focus-on-novelty` | True | Prioritize novelty |
| `--focus-on-impact` | True | Prioritize impact |

### LLM Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | gpt-5-nano | LLM model to use |
| `--temperature` | 0.0 | LLM temperature parameter (0.0-1.0) |
| `--max-tokens` | 1000 | Maximum LLM tokens |

### Output Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | storage/outputs | Output directory |
| `--output-file` | Auto-generated | Output filename |
| `--top-n-display` | 10 | Number of papers to display in console |

### Other Options

| Option | Description |
|--------|-------------|
| `--verbose`, `-v` | Show detailed logs |
| `--no-llm-eval` | Skip LLM evaluation |
| `--help`, `-h` | Show help |

## 📚 Usage Examples

### Example 1: Basic Usage

```bash
python run_deep_research.py \
  --venue NeurIPS \
  --year 2025 \
  --research-description "I'm interested in efficient methods for large language models"
```

### Example 2: With Detailed Settings

```bash
python run_deep_research.py \
  --venue NeurIPS \
  --year 2025 \
  --research-description "I'm interested in reinforcement learning and its applications" \
  --top-k 50 \
  --min-relevance-score 0.3 \
  --model gpt-4o
```

### Example 3: Keyword-Based Only (Fast)

```bash
python run_deep_research.py \
  --venue NeurIPS \
  --year 2025 \
  --research-interests "reinforcement learning,robotics" \
  --no-llm-eval \
  --top-n-display 20
```

### Example 4: With Verbose Logging

```bash
python run_deep_research.py \
  --venue NeurIPS \
  --year 2025 \
  --research-description "graph neural networks" \
  --verbose
```

### Example 5: Custom Output Settings

```bash
python run_deep_research.py \
  --venue NeurIPS \
  --year 2025 \
  --research-description "transformer architecture" \
  --output-dir ./my_results \
  --output-file transformers_review.md
```

## 🏗️ Architecture

```
deep-openreview-research/
├── fetch_all_papers.py      # Paper data fetching script
├── run_deep_research.py     # Main execution script
├── quickstart.sh            # Quick start script
├── app/
│   ├── paper_review_workflow/  # Paper review workflow
│   │   ├── agent.py            # Main agent
│   │   ├── config.py           # Configuration
│   │   ├── constants.py        # Constants
│   │   ├── models/             # Data models
│   │   ├── nodes/              # Workflow nodes
│   │   │   ├── unified_llm_evaluate_papers_node.py  # Unified LLM evaluation
│   │   │   ├── generate_paper_report_node.py        # Report generation
│   │   │   └── ...
│   │   ├── tools/              # Tool functions
│   │   └── utils/              # Utilities
│   ├── core/                   # Core features
│   ├── domain/                 # Domain layer
│   └── infrastructure/         # Infrastructure layer
└── storage/
    ├── cache/                  # Cache data
    ├── outputs/                # Output reports
    └── papers_data/            # Paper data
```

## 🔄 Workflow

1. **Keyword Collection**: Extract research keywords from natural language (or specify directly)
2. **Synonym Generation**: Generate synonyms for each keyword using LLM
3. **Paper Search**: Search papers from specified conference and year
4. **Initial Evaluation**: Calculate relevance scores with keyword matching
5. **Ranking**: Select top k papers based on scores
6. **Unified LLM Evaluation** (⚡Parallel Processing): Execute up to 10 concurrent evaluations, comprehensively evaluating in a single call:
   - Relevance score
   - Novelty score
   - Impact score
   - Practicality score
   - Review summary
   - Field insights
   - AI rationale
   - **Processing Time**: Evaluate 100 papers in ~30 seconds (10x faster than sequential)
7. **Re-ranking**: Final ranking based on LLM evaluation scores
8. **Report Generation**: Generate detailed report (including review score averages)

## 📝 Output

After execution completes, the following file is generated:

```
storage/outputs/paper_review_report_NeurIPS_2025.md
```

The report includes:
- Search criteria and evaluation criteria
- Keywords and synonym lists
- Statistics (average scores, etc.)
- Top paper details:
  - Title, authors, keywords
  - Abstract
  - Score details (relevance, novelty, impact, practicality)
  - Acceptance decision and presentation format (Oral/Spotlight/Poster)
  - **🤖 AI Evaluation** - Detailed analysis from unified LLM evaluation
  - **📊 Review Summary** - Integrated summary of all reviewer evaluations
  - **🔍 Evaluation Data Sources** - Description of review fields used
  - **📝 Acceptance Reason (Decision Comment)** - Program Chairs' acceptance decision comments
  - **📊 Review Score Averages** - Average values for each evaluation item (varies by conference)
  - **💬 Author Comments (Author Remarks)**
  - Links (OpenReview, PDF)

## 🔧 Troubleshooting

### ModuleNotFoundError

Ensure the virtual environment is activated:

```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### OPENAI_API_KEY not found error

Verify the environment variable is correctly set:

```bash
echo $OPENAI_API_KEY  # macOS/Linux
echo %OPENAI_API_KEY%  # Windows
```

### Paper data not found

First fetch the paper data:

```bash
python fetch_all_papers.py --venue NeurIPS --year 2025
```

### Paper data fetch interrupted

Progress is auto-saved, so re-running will resume from where it stopped:

```bash
python fetch_all_papers.py --venue NeurIPS --year 2025
```

### Memory insufficient error

Reduce the `--top-k` value:

```bash
python run_deep_research.py ... --top-k 30
```

### AI evaluation or review summary not displayed

If using old cache, the dynamic field detection feature may not be included. Re-fetch paper data:

```bash
python fetch_all_papers.py --venue NeurIPS --year 2025 --force
```

**Note**: Re-fetching takes 60-90 minutes.

### ICML shows only one review score

This is normal behavior. ICML 2025's review system has only `overall_recommendation` as a numerical score, while other evaluation items (experimental design, methods, etc.) are in text description format. These text evaluations are summarized by the unified LLM in the "📊 Review Summary" section.

### Slow execution

You can speed up with the following options:

```bash
# Skip LLM evaluation
python run_deep_research.py ... --no-llm-eval

# Reduce number of papers to evaluate
python run_deep_research.py ... --top-k 50

# Use more capable model if needed
python run_deep_research.py ... --model gpt-4o-mini
```

## ⚡ Performance and Optimization

### Speed Improvement with Parallel Processing

Parallel execution of LLM evaluations achieves significant processing time reduction:

| Papers | Sequential | Parallel (10 concurrent) | Speedup |
|--------|-----------|-------------------------|---------|
| 10 papers | 30s | 3s | **10x** |
| 50 papers | 150s (2.5 min) | 15s | **10x** |
| 100 papers | 300s (5 min) | 30s | **10x** |

*Assumes 3 seconds per paper (actual time depends on LLM response speed)

### Optimization Mechanisms

- **asyncio + Semaphore**: Efficiently executes up to 10 concurrent requests
- **Rate Limiting**: Concurrent limit configured considering OpenAI API rate limits
- **Error Handling**: Continues other evaluations even if some fail
- **Caching Mechanism**: Avoids re-evaluation of same papers to reduce API calls

### Tips for Reducing API Costs

1. **Use smaller models**: Default `gpt-5-nano` is fastest & cheapest (use `gpt-4o-mini` for higher quality)
2. **Limit top-k**: Appropriately restrict the number of papers to evaluate (default: 30)
3. **Leverage cache**: Re-runs with same conference/year/keywords use cached results

## 🛠️ Tech Stack

- **Python**: 3.12+
- **Framework**: LangGraph, LangChain
- **LLM Provider**: OpenAI
- **API**: OpenReview API
- **Data Validation**: Pydantic
- **Code Quality**: Ruff, MyPy

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under **CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License)**.

### ✅ You are free to:
- Use for **academic and research purposes**
- Share and adapt the source code
- Use for personal learning and research

### ❌ You may NOT:
- Use for **commercial purposes**
- Provide paid services using this software
- Incorporate into commercial products

### 📋 Under the following conditions:
- **Attribution** - Give appropriate credit and provide a link to the license
- **ShareAlike** - Distribute derivatives under the same license
- **NonCommercial** - No commercial use allowed

For commercial use inquiries, please contact us for a separate license agreement. See the [LICENSE](LICENSE) file for full details.

## 🙏 Acknowledgments

This project uses the following excellent tools and services:

- [OpenReview](https://openreview.net/) - Academic paper peer review platform
- [LangGraph](https://github.com/langchain-ai/langgraph) - LLM application framework
- [LangChain](https://github.com/langchain-ai/langchain) - LLM integration framework
- [OpenAI](https://openai.com/) - LLM API

## 📞 Support

If you encounter issues or have questions, please contact us via [GitHub Issues](https://github.com/tb-yasu/deep-openreview-research/issues).

---

**Happy Paper Reviewing! 📚✨**
