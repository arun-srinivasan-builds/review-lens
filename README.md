# ReviewLens — Customer Feedback Analyzer

Upload a weekly CSV of customer reviews (Google, Zomato, Swiggy, Amazon). ReviewLens classifies every
review with a structured LangChain pipeline (English, Tamil and Tanglish), highlights urgent complaints,
and produces a weekly insights report a manager can read in about five minutes.

- **UI:** Streamlit (single page: Upload → Analyse → Dashboard / Report → Export)
- **LLM:** OpenAI via `langchain-openai` (`ChatOpenAI`) with Pydantic structured output
- **Metrics:** exact counts and percentages are computed in pandas, never by the LLM
- **Storage:** none. Uploads live only in the Streamlit session

> **Privacy:** review text is sent to OpenAI for processing. Do not upload data you are not allowed to share
> with that provider. Suggested replies are drafts for human review and are never sent automatically.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` and `OPENAI_MODEL` (a model that supports structured output).
`.env` is git-ignored; never commit it.

**Model choice:** use a chat model that supports structured output. ReviewLens sends no `temperature` by
default, because some models reject anything but their default; set `OPENAI_TEMPERATURE` only if your model
accepts it. Reasoning models can be slower, so raise `LLM_TIMEOUT_SECONDS` if requests time out. If the
provider rejects the key, model or quota, ReviewLens stops after the first batch instead of sending every
review.

**Tracing:** LangChain/LangSmith tracing is forced off at startup so review text goes only to OpenAI.

## Run

```bash
streamlit run app.py
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `openai` | Only `openai` is supported initially |
| `OPENAI_API_KEY` | — | Your OpenAI API key (required) |
| `OPENAI_MODEL` | — | Chat model name (required) |
| `OPENAI_TEMPERATURE` | unset | Optional; unset sends no temperature (some models reject non-default values) |
| `MAX_UPLOAD_MB` | `10` | Maximum CSV size |
| `MAX_ROWS` | `5000` | Maximum rows per upload |
| `MAX_REVIEW_CHARS` | `2000` | Maximum characters per review |
| `MAX_CONCURRENCY` | `5` | Parallel LLM calls |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-request timeout |
| `MAX_RETRIES` | `2` | Retries for invalid or transient failures |

## CSV format

Required columns, exactly:

```csv
review_id,date,platform,rating,review_text
R-1041,2026-10-03,Zomato,5,"Biryani was amazing, packing also neat. Will order again!"
R-1042,2026-10-03,Swiggy,1,"Ordered at 8pm, came at 9:10. Food fully cold. Very bad."
R-1043,2026-10-04,Google,2,"Saapadu nalla irundhuchu but bill la extra charge pottanga"
```

Rows with empty review text are dropped and counted. Rows with invalid dates, IDs or ratings (1–5) are
rejected and listed with a reason. Sample files are in `sample_data/`.

## Output

Each review gets `sentiment`, `category`, `urgency`, `language`, `summary` (always English) and
`suggested_reply`, plus `processing_status` (`completed` or `needs_review`). Reviews the model could not classify after the
retry limit are flagged `needs_review` with their original text preserved and are not counted as analysed.

Suggested replies follow the review's language: English → English, Tamil script → Tamil, Tanglish →
Tanglish, uncertain or mixed → English.

The weekly report shows the period, analysed total, sentiment percentages, top 3 complaints, top praise,
the HIGH-urgency count and exactly three action items. Export is an Excel-compatible UTF-8 (BOM) CSV with
formula-injection protection on text cells.

## Safeguards

- Upload checks: `.csv` only, size and row caps, strict header and encoding validation
- Reviews are treated as untrusted data; instructions inside a review are never followed
- Enum-validated structured output, bounded retries, bounded concurrency, request timeouts
- No review text or API keys in logs or UI error messages
- CSV formula-injection protection on export
- Prompt delimiters are neutralised in untrusted text; narrative numbers in the weekly report are checked
  against the computed metrics, and counts shown on screen always come from pandas
- Upload size is limited in the uploader (`.streamlit/config.toml` and `MAX_UPLOAD_MB`); extra CSV columns
  are discarded after the header check

This is a local, single-user tool without authentication. Add access control before exposing it publicly
with sensitive customer data.

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -v
pip-audit -r requirements.txt
```

## Project layout

```
app.py                  Streamlit entrypoint
reviewlens/             validation, classification, aggregation, report, export, llm, config, security
ui/dashboard.py         charts, filters, tables
sample_data/            sample and 200+ row test CSVs
tests/                  pytest suite (LLM mocked)
```
