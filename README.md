# AI Evals CLI 

## 🎯 What it does:
**AI Eval CLI** is a command-line tool that leverages AI evals to A/B test multiple AI models against goldens and generated edge cases. By scoring responses on accuracy and robustness, it helps product managers and developers compare models under real-world conditions. The tool outputs clear reports that highlight strengths, weaknesses, and trade-offs between models—making it easier to guide model selection and deployment decisions.

## 🔧 How it works:

### 1. **Input**: 
- Takes a JSON dataset with prompts + "golden" (perfect) responses
- Example: `"Who is the CEO of Tesla?"` → `"Elon Musk."`

### 2. **Edge Case Generation**:
For each prompt, automatically generates 4 types of edge cases:
- **Paraphrase**: Slight rewording of the original
- **Constraint**: Adds requirements (e.g., "answer in one sentence")
- **Noisy**: Adds typos, emojis, shorthand
- **Ambiguity**: Makes the prompt less specific

### 3. **Model Testing**:
- Calls **Model A** and **Model B** (e.g., GPT-4 vs Llama2) on each prompt
- Tests both the original prompt + all 4 edge cases
- Scores each response 1-5 based on:
  - Semantic similarity to golden response
  - Clarity/readability
  - Intent alignment

### 4. **Output**:
Generates 3 files:
- **CSV**: Raw data with all scores
- **Markdown**: Human-readable report
- **PDF**: Printable report with recommendations

## 📊 Example Results:
```
Normal Case: Model A wins 70%  
Edge Cases: Model B wins 55%  
Recommendation: Model A for accuracy; Model B for robustness
```

## 🚀 Use Cases:
- **Model Selection**: "Which model should we deploy?"
- **Robustness Testing**: "How well does our model handle real user variations?"
- **Quality Assurance**: "Does our model break under edge cases?"

## 💡 Key Insight:
Most tools test models on perfect inputs. This tool tests how models perform when users ask questions in messy, real-world ways - which is what actually happens in production.

**Bottom line**: It's a PM-friendly way to evaluate AI models beyond just accuracy, focusing on real-world robustness.

## 📦 Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd ai-eval-cli
```

2. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install in editable mode:**
```bash
pip install -e .
```

4. **Set up OpenAI API key:**
```bash
export OPENAI_API_KEY=your_api_key_here
```

## 🎯 Quick Start

### Basic Usage
```bash
# Compare two OpenAI models
aieval evaluate --dataset datasets/starter.json --model-a gpt-4o-mini --model-b gpt-4o-mini --outdir .
```

### Available Models
- `gpt-4o-mini` (recommended for testing)
- `gpt-4o`
- `gpt-4`
- `gpt-3.5-turbo`

## 📊 Understanding the Output

The tool generates three files:
- `results_edge.csv` - Detailed results in CSV format
- `results_edge.md` - Human-readable markdown report
- `results_edge.pdf` - Professional PDF report

### Evaluation Metrics
- **Semantic Similarity**: Token overlap with golden response
- **Clarity**: Flesch reading ease score
- **Tone Alignment**: VADER sentiment analysis
- **Overall Score**: Weighted average mapped to 1-5 scale

## 📁 Dataset Format

Create your own dataset in JSON format:

```json
[
  {
    "task": "summarization",
    "prompt": "Summarize: The cat sat on the mat.",
    "golden": "A cat is sitting on a mat.",
    "tone": "neutral"
  }
]
```

### Required Fields
- `task`: Type of task (summarization, qa, instruction, reasoning, creative)
- `prompt`: The input prompt for the model
- `golden`: The expected/perfect response
- `tone`: Expected tone (neutral, polite, concise, creative, friendly, punchy)

## 🔧 Advanced Usage

### Custom Output Directory
```bash
aieval evaluate --dataset datasets/starter.json --model-a gpt-4o-mini --model-b gpt-4o-mini --outdir ./results
```

### Using Different Models
```bash
# Compare different model families
aieval evaluate --dataset datasets/starter.json --model-a gpt-4o-mini --model-b gpt-4o --outdir .
```

## 📈 Understanding Results

### Win Rates
- **Normal-case wins**: Performance on original prompts
- **Edge-case wins**: Performance on generated edge cases
- **Overall recommendation**: Based on combined performance

### Scoring (1-5 Scale)
- **5**: Excellent match with golden response
- **4**: Good match with minor differences
- **3**: Acceptable match with some differences
- **2**: Poor match with significant differences
- **1**: Very poor match or error

## 🛠️ Development

### Project Structure
```
ai-eval-cli/
├── aieval/
│   └── cli.py          # Main CLI implementation
├── datasets/
│   └── starter.json    # Example dataset
├── pyproject.toml      # Project configuration
└── README.md          # This file
```

### Adding New Features
1. Modify `aieval/cli.py`
2. Update `pyproject.toml` if adding dependencies
3. Test with `pip install -e .`
4. Update documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

**Error 429 - Insufficient Quota:**
- Check your OpenAI account billing
- Add payment method even for free tier
- Try a different model or wait for rate limits

**Import Errors:**
- Ensure virtual environment is activated
- Reinstall with `pip install -e .`

**PDF Generation Fails:**
- Check if reportlab is installed
- Ensure write permissions in output directory

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub

---

**Built for Product Managers and AI practitioners who need simple, effective AI evals.** 🎯
