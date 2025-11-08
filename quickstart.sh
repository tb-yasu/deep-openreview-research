#!/bin/bash
# Paper Review Agent - Quick Start Script
#
# This script runs the most basic usage example.

echo "=========================================="
echo "Paper Review Agent - Quick Start"
echo "=========================================="
echo ""

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    # Check if .env file exists
    if [ ! -f ".env" ]; then
    echo "❌ Error: OPENAI_API_KEY is not set"
    echo ""
        echo "Method 1: Create .env file (recommended)"
        echo "  cp .env.example .env"
        echo "  # Edit .env file and set your API key"
        echo ""
        echo "Method 2: Set as environment variable"
    echo "  export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    exit 1
    else
        echo "ℹ️  Found .env file"
        echo "   Python script will automatically load it"
        echo ""
    fi
fi

# Check if paper data exists
if [ ! -f "storage/papers_data/NeurIPS_2025/all_papers.json" ]; then
    echo "⚠️  Warning: Paper data not found"
    echo "Please fetch paper data first:"
    echo "  python fetch_all_papers.py"
    echo ""
    read -p "Fetch now? (y/N): " response
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        echo "Fetching paper data..."
        python fetch_all_papers.py
        if [ $? -ne 0 ]; then
            echo "❌ Failed to fetch paper data"
            exit 1
        fi
    else
        echo "Please fetch paper data and try again"
        exit 1
    fi
fi

echo "✅ Environment check passed"
echo ""
echo "📚 Execution conditions:"
echo "  - Conference: NeurIPS 2025"
echo "  - Research interests: Graph generation and drug discovery"
echo "  - Model: GPT-4o-mini"
echo "  - Evaluation target: Top 100 papers"
echo ""
echo "Starting execution..."
echo ""

# Run the public release script
python run_deep_research.py \
  --venue NeurIPS \
  --year 2025 \
  --research-description "I am interested in graph generation and its applications to drug discovery. Specifically, I am looking for research related to molecular graph generation and drug design." \
  --top-k 100 \
  --model gpt-4o-mini

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Processing completed successfully!"
    echo "=========================================="
    echo ""
    echo "📝 Report saved at:"
    echo "  storage/outputs/paper_review_report_NeurIPS_2025.md"
    echo ""
    echo "Next steps:"
    echo "  1. Review the report file"
    echo "  2. Try with different research interests"
    echo "  3. Use custom options (see README.md)"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ An error occurred"
    echo "=========================================="
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check if OpenAI API key is correctly set"
    echo "  2. Check if paper data exists"
    echo "  3. Run with --verbose option for detailed logs"
    echo ""
    echo "See README.md for details"
    echo ""
    exit 1
fi
