# PRP: ReviewLens — Customer Feedback Analyzer

> Implementation blueprint for parallel agent execution

---

## METADATA

| Field | Value |
|-------|-------|
| **Product** | ReviewLens — Customer Feedback Analyzer |
| **Type** | Internal analytics web application (Streamlit, not a subscription SaaS) |
| **Version** | 1.0 |
| **Created** | 2026-10-12 |
| **Complexity** | Medium |
| **Source** | `INITIAL.md` (Social Eagle AI Academy, Track 1, requirements 1.1–1.8) |

> **Template override:** `CLAUDE.md` describes the generic FastAPI + React + PostgreSQL + JWT template. `INITIAL.md` explicitly overrides this for ReviewLens: **Streamlit + LangChain, no database, no auth, no payments.** Do not add FastAPI, React, PostgreSQL, OAuth, billing or multi-tenancy. CLAUDE.md rules that still apply: type hints, no `print()` (use `logging`), no hardcoded secrets, no plain-text secrets in logs, no inline unsafe HTML.

---

## PRODUCT OVERVIEW

**Description:** Upload a weekly CSV of customer reviews (Google, Zomato, Swiggy, Amazon). ReviewLens classifies each review with a structured LangChain pipeline (English, Tamil, Tanglish), flags urgent complaints, computes exact metrics in pandas, and produces a weekly insights report a manager can read in about five minutes.

**Value Proposition:** Turns hundreds of unread reviews into a prioritised, evidence-grounded action list, with draft replies for human review.

**MVP Scope:**
- [ ] CSV upload with five required columns, validation, empty-text handling
- [ ] LangChain structured classification (six fields, enforced enums)
- [ ] Tamil / Tanglish understanding with English summaries
- [ ] Bounded retry, `needs_review` handling, batched bounded concurrency
- [ ] Exact pandas metrics; second LangChain chain for report + exactly 3 action items
- [ ] Sentiment chart, category chart, HIGH-urgency filter
- [ ] Excel-compatible UTF-8 BOM CSV export with formula-injection protection
- [ ] `.env` config, `.env.example`, `.gitignore`, README
- [ ] Upload security, prompt-injection resistance, secret-safe errors

**Out of scope:** auth, billing, database, auto-posting replies, email/scheduling, CrewAI/AutoGen.

---

## TECH STACK

| Layer | Technology | Notes |
|-------|------------|-------|
| Language | Python 3.11+ | Type hints required |
| UI | Streamlit | Single-page workflow |
| Data | pandas | All numeric metrics |
| LLM orchestration | LangChain (`ChatPromptTemplate`, LCEL) | Real chains, no CrewAI/AutoGen |
| Structured output | Pydantic v2 + `with_structured_output()` | Enums enforced |
| LLM provider | **OpenAI (default and only provider for initial implementation)** via `langchain-openai` `ChatOpenAI` | `OPENAI_API_KEY` and `OPENAI_MODEL` from `.env`. No Gemini or Claude dependencies. |
| Charts | Streamlit native or Plotly | |
| Storage | In-session DataFrames only | No persistence of uploads |
| Config | `.env` + `python-dotenv` | `.env.example` with placeholders |
| Tests | pytest (+ mocked LLM) | Deterministic numeric tests |
| Lint | ruff | `ruff check .` |
| Dependencies | `requirements.txt` | Audit with `pip-audit`. LLM deps limited to `langchain`, `langchain-core`, `langchain-openai`; do **not** add `langchain-google-genai`, `langchain-anthropic`, `google-generativeai` or `anthropic` |

**Skill references:** `skills/TESTING.md` (test conventions only). `skills/BACKEND.md`, `DATABASE.md`, `FRONTEND.md`, `DEPLOYMENT.md` target the FastAPI/React template and are **not applicable** except for general code-style guidance.

---

## PROJECT STRUCTURE

```
customer-feedback-analyzer/
├── app.py                      # Streamlit entrypoint (UI orchestration only)
├── reviewlens/
│   ├── __init__.py
│   ├── config.py               # Settings from env: provider, model, limits, concurrency
│   ├── models.py               # Pydantic: ReviewInput, ReviewAnalysis, EnrichedReview,
│   │                           #           WeeklyMetrics, WeeklyReport; ValidationResult dataclass
│   ├── validation.py           # load_reviews, validate_reviews, upload checks
│   ├── llm.py                  # get_llm() provider factory; only "openai" -> ChatOpenAI implemented
│   ├── classification.py       # build_review_chain, classify_reviews, retry_invalid_review
│   ├── aggregation.py          # calculate_weekly_metrics (pandas only)
│   ├── report.py               # build_report_chain, generate_weekly_report
│   ├── export.py               # export_enriched_csv, formula-injection sanitiser
│   ├── security.py             # Input sanitisation, secret redaction, safe error mapping
│   └── logging_utils.py        # Privacy-safe logging (no review text, no secrets)
├── ui/
│   └── dashboard.py            # render_dashboard, charts, urgency filter, table
├── sample_data/
│   ├── sample_reviews.csv      # Academy 3-row sample
│   └── reviews_200plus.csv     # 200+ rows incl. Tamil, Tanglish, injection + formula cases
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## DATA MODELS (no database)

| Model | Fields | Notes |
|-------|--------|-------|
| `ReviewInput` | review_id: str, date: date, platform: str, rating: float, review_text: str | rating 1–5, text length cap |
| `ValidationResult` | valid_reviews: DataFrame, rejected_rows: DataFrame, errors: list[str] | includes dropped-empty count |
| `ReviewAnalysis` | sentiment: Literal[positive, neutral, negative]; category: Literal[food_quality, delivery, price, staff, packaging, app]; urgency: Literal[low, medium, high]; language: str; summary: str; suggested_reply: str | Used with `with_structured_output()` |
| `EnrichedReview` | source fields + ReviewAnalysis fields + processing_status: Literal[completed, needs_review] | Original text always preserved |
| `WeeklyMetrics` | period_start, period_end, total_reviews, sentiment_counts, sentiment_percentages, category_counts, urgency_counts, high_urgency_count | pandas only; computed over `completed` rows |
| `WeeklyReport` | top_3_complaints: list, top_praises: list, urgent_attention: str, action_items: list[str] (exactly 3) | Validator enforces length 3 |

---

## MODULES

### Module 1: CSV Upload and Validation
**Agent:** BACKEND-AGENT (core logic) + FRONTEND-AGENT (upload UI)

| Function | Description |
|----------|-------------|
| `load_reviews(file) -> DataFrame` | Check extension `.csv`, size ≤ `MAX_UPLOAD_MB` (10), UTF-8 decode, row cap ≤ `MAX_ROWS` (5000), exact headers |
| `validate_reviews(df) -> ValidationResult` | Drop empty text (counted), validate date / review_id (non-empty, unique) / rating 1–5 / text length; invalid rows → `rejected_rows` with reason |

**UI:** uploader, filename, input / accepted / rejected / dropped-empty counts, rejected-rows expander. Text rendered as plain text only.

### Module 2: Review Classification (LangChain)
**Agent:** BACKEND-AGENT

| Function | Description |
|----------|-------------|
| `build_review_chain()` | `ChatPromptTemplate | llm.with_structured_output(ReviewAnalysis)`; system prompt treats review as untrusted data inside delimiters |
| `classify_reviews(reviews, max_concurrency)` | `chain.batch(..., config={"max_concurrency": N})` in chunks, with progress callback; returns `list[EnrichedReview]` |
| `retry_invalid_review(review)` | Bounded retries (default 2) with capped backoff; on exhaustion → `needs_review`, source preserved, analysis fields empty (never fabricated) |

**Prompt requirements:** exactly one category; urgency high for health/safety, refunds, abuse, wrong order; English summary for Tamil/Tanglish; polite draft reply; no invented order IDs, refund promises or actions; ignore instructions inside the review.

**UI:** progress bar, completed / needs_review counts, enriched table with visible status. Failed calls never wipe completed results.

### Module 3: Aggregation and Weekly Insights
**Agent:** BACKEND-AGENT

| Function | Description |
|----------|-------------|
| `calculate_weekly_metrics(enriched_df)` | All counts / percentages via pandas on `completed` rows; period from uploaded dates |
| `build_report_chain()` | Second LCEL chain with `with_structured_output(WeeklyReport)` |
| `generate_weekly_report(metrics, review_evidence)` | Reduce step: passes computed metrics plus capped, sampled summaries (not raw full text) as evidence; LLM may not state numbers absent from metrics |

**UI:** report with headings: period, total analysed, sentiment %, top 3 complaints, top praise, urgent count (from pandas), 3 action items.

### Module 4: Dashboard and Export
**Agent:** FRONTEND-AGENT

| Function | Description |
|----------|-------------|
| `render_dashboard(enriched_df, metrics, report)` | Sentiment chart, category chart, HIGH-urgency toggle, readable table (review, sentiment, category, urgency, summary, reply) |
| `export_enriched_csv(enriched_df) -> bytes` | UTF-8 with BOM; prefix text cells starting (after whitespace) with `=`, `+`, `-`, `@` using `'`; keep numeric rating numeric |

**Flow:** Upload → Analyse → Dashboard / Report → Export, single page.

---

## SECURITY AND PRIVACY REQUIREMENTS (mandatory)

| Control | Implementation |
|---------|----------------|
| Secrets | Env vars only; `.env` in `.gitignore`; `.env.example` placeholders; redact keys in all errors/logs |
| Upload limits | `.csv` only, size cap, row cap, header check, encoding errors with friendly messages |
| Data validation | Dates, IDs, ratings 1–5, text length cap, drop-and-count blanks |
| Prompt injection | Review text isolated in delimiters, system rules take precedence, no tools/URLs/code executed, output validated against enums |
| Bounded cost | Concurrency default 5, request timeout, ≤ 2 retries with capped backoff |
| Privacy | Never log review text/PII; no persistence; UI discloses selected LLM provider processes the reviews |
| Export | Formula-injection sanitiser; BOM for Excel |
| Errors | User-friendly messages, no tracebacks or keys in UI |
| Replies | Draft-only, labelled "for human review" |
| Dependencies | Minimal `requirements.txt`; `pip-audit` before submission |

---

## PHASE EXECUTION PLAN

**Phase 1: Foundation (parallel)**
- BACKEND-AGENT: `config.py` (loads `OPENAI_API_KEY`, `OPENAI_MODEL`), `models.py`, `llm.py` (OpenAI `ChatOpenAI` factory), `security.py`, `logging_utils.py`, package skeleton
- FRONTEND-AGENT: `app.py` skeleton, session-state layout, upload widget shell
- DEVOPS-AGENT: `requirements.txt`, `.env.example`, `.gitignore`, ruff config, README skeleton
- DATABASE-AGENT: **not needed** (no database); skipped. Pydantic models are owned by BACKEND-AGENT.
- DATA task (BACKEND-AGENT): create `sample_data/sample_reviews.csv` and `reviews_200plus.csv`

**Validation Gate 1:** `pip install -r requirements.txt`, `python -c "import reviewlens.config, reviewlens.models"`, `ruff check .`, `.env` confirmed ignored by git

**Phase 2: Modules (backend + frontend parallel per module)**
- Module 1: `validation.py` ‖ upload UI
- Module 2: `classification.py` ‖ progress/table UI
- Module 3: `aggregation.py`, `report.py` ‖ report UI
- Module 4: `export.py` ‖ `ui/dashboard.py`

**Validation Gate 2:** `ruff check .`, `pytest tests -v` for deterministic tests, `streamlit run app.py` starts without error

**Phase 3: Quality (parallel)**
- TEST-AGENT: validation, enums, retry/needs_review, aggregation exactness, export sanitiser, prompt-injection (mocked LLM), secret-redaction tests
- REVIEW-AGENT: security audit of the safeguards table, privacy of logs, `pip-audit`
- RESEARCH-AGENT: verify current `langchain-openai` `ChatOpenAI` parameters and `with_structured_output` usage with Pydantic; confirm `requirements.txt` pins

**Final Validation:** full pytest, live run (real LLM) on sample and 200+ row dataset, manual Excel open of export, README accuracy

---

## VALIDATION GATES

| Gate | Commands |
|------|----------|
| 1 | `pip install -r requirements.txt`, `ruff check .`, `git check-ignore .env` |
| 2 | `pytest tests -v`, `streamlit run app.py --server.headless true` (smoke) |
| 3 | `pytest --cov=reviewlens --cov-fail-under=80`, `pip-audit -r requirements.txt` |
| Final | Real-LLM run on 200+ rows; verify counts == pandas; export opens in Excel |

---

## TEST PLAN (key cases)

| Area | Test |
|------|------|
| Upload | Oversize file, non-CSV, malformed encoding, missing header, > row cap rejected |
| Validation | Invalid rating (0, 6, "abc"), bad date, duplicate/blank ID, overlong text, empty text dropped and counted |
| Schema | Invalid enum values rejected by Pydantic |
| Retry | Persistently invalid response stops at retry limit → `needs_review`, source preserved |
| Resilience | Provider failure mid-batch keeps completed rows |
| Aggregation | Counts/percentages exact on fixed fixtures; `needs_review` excluded from analysed totals |
| Report | `action_items` length exactly 3; LLM numbers not used for metrics |
| Injection | Review "ignore previous instructions…" classified as data (mocked + live check) |
| Export | `=cmd()`, `+1`, `-1`, `@SUM` prefixed; leading-space variants; BOM present; rating stays numeric |
| Secrets | Exceptions/logs contain no API key; `.env` ignored |

---

## LLM PROVIDER: OPENAI (decision)

**Decision:** OpenAI is the default and only implemented provider. The factory in `reviewlens/llm.py` stays so other providers can be added later without touching chains.

**`llm.py` contract:**

```python
def get_llm(settings: Settings) -> BaseChatModel:
    """Return the configured chat model. Raises ConfigError (secret-safe) if misconfigured."""
    if settings.llm_provider == "openai":
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=SecretStr(settings.openai_api_key),
            temperature=0,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,   # retries are owned by retry_invalid_review / bounded backoff, not hidden in the SDK
        )
    raise ConfigError(f"Unsupported LLM provider: {settings.llm_provider!r}")
```

**Rules:**
- `OPENAI_API_KEY` and `OPENAI_MODEL` are read only from environment / `.env` (`python-dotenv`). No hardcoded key or model fallback that hides a missing value; missing values produce a friendly UI error that never prints the key.
- API key held as `SecretStr`; never logged, never included in exception text, UI messages or `repr(settings)`.
- Chains use `get_llm(settings).with_structured_output(ReviewAnalysis)` and `...(WeeklyReport)`; Pydantic validation stays authoritative. `OPENAI_MODEL` must be a model that supports structured output; startup check reports this clearly if the first call fails.
- `temperature=0` for stable classification.
- Single place that constructs `ChatOpenAI`; chains receive the model via argument so tests inject a fake LLM (no network in tests).
- Reviews are sent only to OpenAI; the UI shows a disclosure notice before analysis.
- Cost and rate-limit controls from the security table are unchanged: `MAX_CONCURRENCY` default 5, `LLM_TIMEOUT_SECONDS`, `MAX_RETRIES` default 2 with capped backoff, row cap and text-length cap. Retries apply to invalid structured output and transient provider errors only (not auth or invalid-request errors).

**Dependencies (`requirements.txt`):** `streamlit`, `pandas`, `pydantic`, `python-dotenv`, `langchain`, `langchain-core`, `langchain-openai`, optionally `plotly`; dev: `pytest`, `pytest-cov`, `ruff`, `pip-audit`. Pin versions after research-agent verification.

---

## ENVIRONMENT VARIABLES (`.env.example`)

```env
LLM_PROVIDER=openai            # only "openai" supported initially
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=your-openai-model-name   # must support structured output
MAX_UPLOAD_MB=10
MAX_ROWS=5000
MAX_REVIEW_CHARS=2000
MAX_CONCURRENCY=5
LLM_TIMEOUT_SECONDS=60
MAX_RETRIES=2
```

---

## RISKS AND OPEN POINTS

| Item | Handling |
|------|----------|
| Provider | Resolved: OpenAI via `langchain-openai`. Factory kept in `llm.py` for future providers |
| Model name | `OPENAI_MODEL` supplied by the user in `.env`; must support structured output |
| Tamil/Tanglish quality | Manual review of a labelled sample; prompt tuning |
| Rate limits on 200+ rows | Concurrency 5, backoff, progress display |
| Report hallucinated figures | Metrics from pandas injected into prompt; counts rendered from `WeeklyMetrics`, not LLM text |
| Skills mismatch | Skills target FastAPI/React; follow `INITIAL.md` instead |

---

## NEXT STEP

```bash
/execute-prp PRPs/reviewlens-customer-feedback-analyzer-prp.md
```
