# Finance Agent

A deep agent that investigates personal and small-business expense data. Give it a transactions CSV and ask a question; it plans, picks analysis tools, drills into suspicious spending, and answers with evidence.

> **Design principle:** the LLM reasons, the code calculates. Every number in an answer comes from a deterministic tool, never from the model.

## Highlights

- **Deep agent, not a fixed script:** built with [deepagents](https://github.com/langchain-ai/deepagents) (LangGraph). It can plan with `write_todos`, choose tools per question, delegate to a subagent, and save findings to files.
- **Open-ended questions:** ask "Why did my spending increase last month?" and it drills down from category trends to individual transactions.
- **Robust data layer:** flexible column mapping, strict validation, and a data quality report before any analysis runs.
- **Privacy by design:** sensitive columns are dropped and emails/phone numbers are masked before data reaches the model.
- **Two anomaly views:** statistical (Z-score, per category) and ML (Isolation Forest with rolling time-window features), compared and never double-counted.
- **Guardrails:** no invented figures, no fraud claims from a single anomaly, every recommendation tied to a specific finding.
- **Measured, not assumed:** offline evaluation of each detector against labeled data, plus per-run tool and token reporting.

## Architecture

```
┌──────────── Deterministic code (no LLM) ────────────┐
│ CSV ─▶ data_cleaner ─▶ quality report ─▶ privacy ─▶ DataFrame │
└──────────────────────────────┬───────────────────────┘
                               ▼
┌──────────────────── Agent layer (LLM) ──────────────────────┐
│ User question                                                │
│      ▼                                                       │
│ Main agent ── plans (write_todos) ── chooses tools           │
│      ├─▶ financial_investigator subagent (+ financial-       │
│      │    investigation skill)                               │
│      ├─▶ analysis & drill-down tools (below)                 │
│      └─▶ files (/findings.md) ─▶ final Markdown answer       │
└──────────────────────────────────────────────────────────────┘
```

### Tools

| Tool | Purpose |
|---|---|
| `get_financial_summary` | Total income, expenses, net savings |
| `investigate_spending_increases` | Month-to-month increases per category |
| `investigate_unusual_transactions` | Z-score outliers per category (top 10) |
| `investigate_ml_anomalies` | Isolation Forest anomalies with rolling features |
| `get_category_trend` | Monthly totals and change % for one category |
| `list_transactions` | Individual transactions, filterable by category and month |

The agent also gets the built-in deepagents tools (planning, file read/write, subagent delegation).

## Project structure

```
finance-agent/
├── main.py                  # Entry point: builds the agent and asks a question
├── agent.py                 # Agent, subagent, and system prompt
├── tools.py                 # LangChain tools exposed to the agent
├── analysis_tools.py        # Analysis logic (summary, trends, Z-score, IQR, Isolation Forest)
├── data_cleaner.py          # CSV loading, column mapping, validation, quality report
├── privacy.py               # Anonymization before data reaches the agent
├── skills/
│   └── financial-investigation/SKILL.md
├── experiments/
│   ├── evaluate_models.py   # Precision / recall / F1 per detector
│   ├── compare_features.py  # Isolation Forest with vs. without rolling features
│   └── list_models.py       # Lists available Groq models
├── tests/
├── data/                    # Synthetic sample data
├── requirements.txt
└── .env.example
```

## Getting started

### 1. Install

```bash
git clone https://github.com/wafaashour499/finance-agent.git
cd finance-agent
uv venv
uv pip install -r requirements.txt
```

Python 3.12+ is recommended.

### 2. Configure

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Set `GROQ_API_KEY` in `.env`. LangSmith variables are optional (tracing).

### 3. Run

```bash
uv run python main.py
```

By default it reads `data/transactions.csv`. To change the data or the question, edit the bottom of `main.py`:

```python
run_financial_investigation(
    file_path,
    question="Why did my spending increase last month?",
    date_format=None,
)
```

Questions to try:

| Question | What to expect |
|---|---|
| `Give me a quick financial summary` | One tool call, short answer |
| `Why did my spending increase last month?` | Drill-down: trend, then individual transactions |
| `Investigate my finances and tell me what needs attention.` | Broad investigation with findings and recommendations |

After each run, the console prints the **tools used** and **token usage per LLM call**.

## Example

On the included synthetic data, the question *"Why did my spending increase last month?"* produced:

- Shopping rose from 2521.25 (Apr 2026) to 33009.93 (May 2026), **+1209.27%**, flagged by `investigate_spending_increases`.
- The agent then called `get_category_trend` and `list_transactions` to find the individual May purchases behind the jump.
- One recommendation: verify the specific large Shopping transactions. No invented budgets, caps, or percentages.

## Input data

Required columns: **date**, **amount**, **type** (`income` or `expense`).
Optional columns: **category**, **description** (filled with `Unknown` if missing), **label** (offline evaluation only, never sent to the agent).

```csv
date,description,amount,type,category,label
2026-01-03,Grocery store,54.20,expense,Food,0
2026-01-05,Salary,3000,income,Salary,0
```

Common column names (`transaction_date`, `merchant`, `income_expense`, ...) are mapped automatically.

### Data quality

`data_cleaner.py` validates every row and returns a quality report (rows accepted/rejected, reasons, counts per error type).

- Rejected: missing or invalid dates, future dates, missing or non-numeric amounts, zero or negative amounts, invalid transaction types.
- Analysis **stops** if more than 40% of rows are rejected.
- Files with multiple currencies are rejected.
- Identical rows are **reported** as possible duplicates, not removed, since repeated purchases are legitimate.

### Dates

Unambiguous formats (`2026-01-15`) work automatically. For `DD/MM/YYYY` vs `MM/DD/YYYY`, the format is inferred from the whole column when at least one date is clear (for example `25/03/2026`). If every date is ambiguous (for example `04/09/2026`), the program stops and asks you to choose:

```python
run_financial_investigation(file_path, date_format="%d/%m/%Y")  # day first
run_financial_investigation(file_path, date_format="%m/%d/%Y")  # month first
```

### Privacy

Before the agent sees any data, `privacy.py` drops identifier columns (name, email, phone, account number, IBAN, address), masks emails and phone numbers inside descriptions, and removes the `label` column.

## Evaluation

```bash
uv run python -m experiments.evaluate_models
```

Results on the labeled synthetic dataset (485 expense transactions, 7 labeled anomalies):

| Method           | Precision | Recall  | F1      |
|------------------|-----------|---------|---------|
| Z-Score          | 100.00%   | 100.00% | 100.00% |
| IQR              | 87.50%    | 100.00% | 93.33%  |
| Isolation Forest | 100.00%   | 85.71%  | 92.31%  |

These numbers illustrate how the detectors behave. They are **not** a claim about real-world performance: the data is synthetic and small, and the thresholds were tuned on it. Each method catches a different kind of anomaly, which is why the agent uses both and compares them.

## Known limitations

- **Agent autonomy depends on the model and the question.** With `openai/gpt-oss-120b`, broad questions often lead to a near-fixed tool sequence, and the ML tool is not always called. Behavior improves with open-ended questions and prompt tuning, but it is not guaranteed.
- **Token limits.** A multi-step investigation sends the system prompt, tool definitions, and prior tool results on every LLM call. On Groq's free tier (about 8,000 tokens per minute) this can trigger rate limits and retries. Mitigations: compact tool outputs, a shorter prompt, and (planned) summarization and fallback models.
- **Refunds:** negative amounts are rejected as "possible refund". Proper support is planned.
- **Single currency, no currency inference:** the agent is instructed not to assume a currency that is not in the data.
- **Fixed detector parameters:** Z-score threshold 2.0, spike threshold 30%, rolling window 5 days, contamination 0.02. They are constants at the top of `analysis_tools.py`, not auto-tuned.
- **Anomaly is not fraud.** Flagged transactions are items to verify, never proof of wrongdoing.
- **LLM-written text.** The final answer is generated by an LLM from tool results. Verify important figures against the tool output.

## Tech stack

Python, pandas, scikit-learn, LangChain, LangGraph, deepagents, Groq (`openai/gpt-oss-120b`).

## Roadmap

- Smaller tool outputs and a shorter system prompt to cut token usage
- Retry with backoff and a fallback model
- Summarization middleware for long conversations
- Refund and negative-amount support
- Parameters that adapt to dataset size
- More unit tests for the cleaner, privacy layer, and detectors

## License

Add a `LICENSE` file (for example MIT) and update this section.
