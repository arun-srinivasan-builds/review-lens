# INITIAL.md - Define Your Product

> ReviewLens product definition for Claude Code agentic coding. First generate the PRP, review it, then execute it.

---

## PRODUCT

**Name:** ReviewLens — Customer Feedback Analyzer

**Description:** A Streamlit application for restaurant chains, online stores, operations managers, support teams and business owners. Upload a weekly CSV of customer reviews from platforms such as Google, Zomato, Swiggy and Amazon. ReviewLens classifies each review using a structured LangChain pipeline, understands English, Tamil and Tanglish, highlights urgent complaints, and produces an actionable weekly insights report. A manager should be able to review the results in about five minutes.

**Type:** Internal analytics web application (not a subscription SaaS)

**Assignment:** Social Eagle AI Academy — Track 1, Gen AI / LangChain; requirements 1.1–1.8.

**Build method:** Agentic coding using Claude Code. Claude Code is the development tool; LangChain is the application's runtime framework. Do not substitute CrewAI or AutoGen for the required LangChain chains.

---

## TECH STACK

| Layer | Choice |
|-------|--------|
| Application language | Python |
| UI | Streamlit |
| Data processing | pandas |
| LLM orchestration | LangChain: ChatPromptTemplate and LCEL chains |
| Structured output | Pydantic and `with_structured_output()` |
| LLM provider | One configurable provider: OpenAI, Gemini or Claude |
| Charts | Streamlit charts or Plotly |
| Storage | Uploaded CSV and in-session processed DataFrames; no database required |
| Authentication | None for assignment MVP |
| Payments | None |
| Configuration | `.env` for API key and model configuration; `.env.example` without secrets |
| Security | CSV upload limits, input sanitisation, prompt-injection resistance, secret-safe logging, dependency checks |
| Tests | pytest, including deterministic data-processing tests and mocked LLM tests |
| Dependency management | `requirements.txt` |

**Security boundary:** This is a local/trusted-user learning lab without accounts or public multi-user access. Authentication (JWT/Google OAuth) from the sample template is deliberately not implemented because the academy does not require it. If deployed publicly with sensitive customer reviews, add appropriate access control before allowing uploads.

**Implementation constraints:** Keep the assignment's Streamlit + LangChain architecture. Do not add FastAPI, React, PostgreSQL, OAuth, billing or multi-tenancy unless explicitly requested later.

---

## MODULES

### Module 1: CSV Upload and Validation

**Description:** Upload a week's reviews, validate required columns and prepare valid rows for analysis.

**Input columns:**

```text
review_id,date,platform,rating,review_text
```

**Data models:**

```text
ReviewInput:
  review_id: string
  date: date
  platform: string
  rating: numeric
  review_text: string

ValidationResult:
  valid_reviews: DataFrame
  rejected_rows: DataFrame
  errors: list[string]
```

**Functions (internal Python, not HTTP endpoints):**

```text
load_reviews(file) -> DataFrame
validate_reviews(df) -> ValidationResult
```

**Behavior:**
- Require all five columns and show actionable errors for missing columns or unreadable CSVs.
- Drop rows with empty review text, as required by the brief; clearly report the dropped-row count.
- Validate dates, review IDs and ratings; surface invalid records rather than silently treating them as valid.
- Preserve source review IDs, dates, platforms, ratings and original review text.
- Accept at least 200 reviews in one upload.

**UI:** Streamlit upload area with filename, input count, accepted count and rejected count.

---

### Module 2: Review Classification (LangChain)

**Description:** Analyse each accepted review using a reusable LangChain chain with strictly typed output. Interpret Tamil and Tanglish, but write summaries in English.

**Models:**

```text
ReviewAnalysis (Pydantic):
  sentiment: Literal["positive", "neutral", "negative"]
  category: Literal["food_quality", "delivery", "price", "staff", "packaging", "app"]
  urgency: Literal["low", "medium", "high"]
  language: string
  summary: string
  suggested_reply: string

EnrichedReview:
  review_id, date, platform, rating, review_text
  sentiment, category, urgency, language, summary, suggested_reply
  processing_status: Literal["completed", "needs_review"]
```

**Required six analysis fields:** `sentiment`, `category`, `urgency`, `language`, `summary`, `suggested_reply`.

**Functions:**

```text
build_review_chain() -> LangChain runnable
classify_reviews(reviews, max_concurrency) -> list[EnrichedReview]
retry_invalid_review(review) -> EnrichedReview
```

**Behavior:**
- Use `ChatPromptTemplate`, LCEL and `with_structured_output()` with Pydantic.
- Classify sentiment and select exactly one category from the six allowed values.
- Set urgency to low, medium or high. Treat health/safety concerns, refund issues, abusive treatment and wrong orders as potential high-urgency cases, based on the review's context.
- Return a concise, faithful one-sentence English summary, including for Tamil/Tanglish reviews.
- Generate a polite suggested customer-service reply. Replies are drafts for human review, not automatically posted.
- Process reviews in batches using `chain.batch()` and a configurable `max_concurrency` to respect provider rate limits.
- Automatically retry invalid structured responses a bounded number of times; then mark the affected review `needs_review` without losing the original review.
- Do not fabricate complaint details, order IDs, refund promises or business actions.
- Make unsuccessful rows clearly visible; do not count them as successfully classified.

**UI:** Processing progress, completion/error counts, enriched review table and a visible needs-review status.

---

### Module 3: Aggregation and Weekly Insights

**Description:** Compute exact metrics in pandas, then use a second LangChain chain to write a short management report grounded in those computed results.

**Models:**

```text
WeeklyMetrics:
  period_start: date
  period_end: date
  total_reviews: integer
  sentiment_counts: dict
  sentiment_percentages: dict
  category_counts: dict
  urgency_counts: dict
  high_urgency_count: integer

WeeklyReport:
  top_3_complaints: list
  top_praises: list
  urgent_attention: string
  action_items: list[string]  # exactly three
```

**Functions:**

```text
calculate_weekly_metrics(enriched_df) -> WeeklyMetrics
build_report_chain() -> LangChain runnable
generate_weekly_report(metrics, review_evidence) -> WeeklyReport
```

**Behavior:**
- Calculate all review counts, sentiment percentages, category counts and urgency counts using pandas, never the LLM.
- Use the uploaded dates for the reporting period; make clear which rows were successfully analysed.
- Use a map-then-reduce design: first analyse individual reviews, then synthesise the batch results into a report.
- Show top three complaints, top praises, number of urgent reviews and exactly three actionable recommendations.
- Ground complaint and praise statements in actual analysed reviews. Do not invent exact issue-specific counts, time windows or platform correlations that have not been calculated or evidenced.
- Keep the report readable by a manager in about five minutes.

**UI:** Weekly feedback report with clear headings and verified metrics.

---

### Module 4: Dashboard and Export

**Description:** Provide a premium, readable Streamlit dashboard for reviewing the weekly results and exporting them.

**Functions:**

```text
render_dashboard(enriched_df, metrics, report) -> None
export_enriched_csv(enriched_df) -> bytes
```

**Features:**
- Sentiment split chart.
- Category breakdown chart.
- HIGH urgency-only filter and review table.
- Readable columns for source review, sentiment, category, urgency, English summary and suggested reply.
- Download the enriched table as an Excel-compatible UTF-8 CSV.
- Display the weekly insights report on the same application workflow.
- Keep UI responsive and understandable without unnecessary navigation.

**Pages:** A single Streamlit app is sufficient: Upload → Analyse → Dashboard / Weekly Report → Export.

---

## SECURITY AND PRIVACY SAFEGUARDS

**Mandatory for this assignment (relevant controls from the sample and safe upload/LLM practices):**
- Load provider keys only from `.env` using environment variables; provide `.env.example` with placeholder values. Add `.env`, local secrets, caches and uploaded datasets to `.gitignore`. Never include secrets in commits, screenshots, logs, exception messages or UI.
- Validate CSV uploads before parsing: accept only `.csv`, enforce a configurable maximum upload size (default 10 MB), reject malformed encoding/CSV with helpful errors, enforce a configurable row cap (default 5,000), and verify exact required headers. These limits must still support the academy's 200+ row test.
- Validate dates, review IDs, ratings (1–5), missing fields and text length. Drop blank review text as specified; clearly count rejected rows. Escape or render uploaded text as plain text, not unsafe HTML.
- Treat every review as **untrusted input**. Prompt instructions inside a review must never override classification rules, reveal system prompts/secrets or control tools. Do not execute uploaded content, generated code or URLs from reviews.
- Use Pydantic enums and structured output to validate classifications; retry invalid responses only a bounded number of times (e.g. 2 retries). Flag unresolved rows `needs_review` and preserve their source text. Never invent a valid-looking result to hide failures.
- Set explicit concurrency limits (default 5), request timeouts, and capped retry/backoff for provider errors. Do not automatically retry forever or make unnecessary duplicate model calls.
- Protect privacy: avoid logging full review texts or personally identifying information; display only needed fields, do not persist uploads beyond the session unless explicitly configured, and never transmit review data to services other than the selected LLM provider. Clearly disclose that selected provider processing is involved.
- Prevent CSV formula injection in exported text cells: prefix dangerous spreadsheet-leading characters (`=`, `+`, `-`, `@`, including after leading whitespace) with a safe apostrophe before export. Preserve legitimate numeric rating fields. Export as UTF-8 with BOM for Excel compatibility.
- Keep dependencies declared in `requirements.txt`, review vulnerabilities with a dependency audit before submission, and avoid unnecessary packages.
- Show user-friendly failures without tracebacks or API secrets in the UI. Keep developer diagnostics minimal and secret-free.
- Suggested replies are **human-reviewed drafts**; never send or post automatically.

**Not part of MVP:** JWT, Google OAuth, User/RefreshToken models, login pages, payments, and database permissions from the generic sample. If authentication is later added, use established authentication libraries, secure password hashing and server-side access checks rather than a cosmetic login screen.

**Security validation tests:**
- Reject oversize files, malformed CSVs, missing headers, invalid ratings and overly long review text.
- A review containing a prompt-injection instruction is treated as data, not obeyed.
- Spreadsheet formula-like review text exports safely and opens in Excel.
- Invalid model responses stop after the retry limit and appear as `needs_review`.
- Error messages/logs contain no API key; `.env` is excluded from Git.

---

## USER WORKFLOW

1. Upload a CSV containing the week's reviews.
2. Validate the required columns and drop empty review-text rows.
3. Run the structured per-review LangChain classification in bounded batches.
4. Retry invalid results; flag unresolved items as `needs_review`.
5. Calculate exact sentiment, category and urgency metrics in pandas.
6. Run the second LangChain chain to generate weekly insights and three action items.
7. Display charts, HIGH urgency filter, report and enriched table.
8. Export the enriched CSV for the support team.

---

## SAMPLE INPUT AND EXPECTED OUTPUT

**Sample CSV:**

```csv
review_id,date,platform,rating,review_text
R-1041,2026-10-03,Zomato,5,"Biryani was amazing, packing also neat. Will order again!"
R-1042,2026-10-03,Swiggy,1,"Ordered at 8pm, came at 9:10. Food fully cold. Very bad."
R-1043,2026-10-04,Google,2,"Saapadu nalla irundhuchu but bill la extra charge pottanga"
```

**Example enriched record:**

```json
{
  "review_id": "R-1042",
  "sentiment": "negative",
  "category": "delivery",
  "urgency": "high",
  "language": "English",
  "summary": "Order arrived 70 minutes late and the food was cold.",
  "suggested_reply": "We are sorry your order arrived late and cold. Please share your order ID privately so our team can investigate."
}
```

**Weekly report structure:**

```text
WEEKLY FEEDBACK REPORT | <start date> to <end date>
Total analysed reviews: <calculated count>
Positive: <calculated %> | Neutral: <calculated %> | Negative: <calculated %>

TOP 3 COMPLAINTS
1. <evidence-based issue>
2. <evidence-based issue>
3. <evidence-based issue>

TOP PRAISE
<evidence-based positive themes>

URGENT
<calculated HIGH urgency count> reviews require attention

ACTION ITEMS
1. <grounded action>
2. <grounded action>
3. <grounded action>
```

The example is illustrative; actual output must be generated from the uploaded CSV and calculated metrics.

---

## MVP SCOPE

**Must Have:**
- [ ] CSV upload with all five required input columns.
- [ ] Input validation and empty review-text handling.
- [ ] LangChain structured classification of every valid review.
- [ ] All six analysis fields on each successfully processed review.
- [ ] English, Tamil and Tanglish review understanding.
- [ ] Allowed sentiment, category and urgency values enforced by Pydantic.
- [ ] Bounded retry and `needs_review` handling for invalid outputs.
- [ ] Batched processing with configurable concurrency.
- [ ] Exact pandas-calculated counts and percentages.
- [ ] Second LangChain chain for weekly insights and three action items.
- [ ] Sentiment chart, category chart and HIGH urgency filter.
- [ ] Excel-compatible enriched CSV export.
- [ ] `.env` configuration, `.env.example`, `.gitignore` and clear README.
- [ ] Upload validation, prompt-injection resistance, privacy-safe error handling and CSV formula-injection protection.

**Out of Scope for MVP:**
- User registration, OAuth, billing and subscriptions.
- PostgreSQL or another persistent database.
- Automatic posting of suggested replies.
- Automatic email delivery or scheduled runs.
- Multi-agent CrewAI / AutoGen workflows.

**Stretch Goals (only after all acceptance criteria pass):**
- Compare current and previous weeks with trend arrows.
- Generate suggested replies in the review's original language.
- Automatically email the weekly report every Monday.

---

## ACCEPTANCE CRITERIA

**Assignment Done Checklist (section 1.8):**
- [ ] Processes a CSV with 200+ reviews without crashing or timing out under the documented test configuration.
- [ ] Every successfully analysed review contains all six required analysis fields; sentiment, category and urgency use only allowed values. Unresolved failures are explicitly flagged `needs_review`.
- [ ] Tamil and Tanglish reviews produce correct, faithful English summaries.
- [ ] Report counts and percentages match pandas calculations exactly; the LLM does not invent metrics.
- [ ] HIGH urgency filtering works and the exported enriched CSV opens correctly in Excel.
- [ ] API keys are loaded from `.env` and are never hard-coded or committed.

**Implementation quality checks (additional to the academy brief):**
- [ ] Tests cover validation, enums, retry/failure handling, aggregation and CSV export.
- [ ] Run the workflow on the academy sample and a separate 200+ row test dataset, including Tamil and Tanglish.
- [ ] A failed LLM call does not erase previously completed results or misrepresent unprocessed reviews as completed.
- [ ] README explains setup, required CSV format, environment variables, run command and a sample input/output.
- [ ] Repository has a clear structure, `requirements.txt`, `.env.example` and `.gitignore`.
- [ ] Demonstrate a complete upload-to-report-to-export workflow.
- [ ] Security tests pass for upload limits, malicious CSV cells, injected review instructions, secret-safe errors and bounded retries.
- [ ] Dependency audit performed; findings fixed or documented.

---

## DEVELOPMENT GUIDANCE FOR CLAUDE CODE

- Build this through **agentic coding with Claude Code**; generate a plan/PRP first and implement against it.
- Prioritise the exact academy requirements and working end-to-end behavior over extra features.
- Keep UI code separate from data validation, LangChain pipelines, aggregation and export functions.
- Use real LangChain calls for actual analysis; mocked LLM outputs are acceptable only in automated tests.
- Use bounded concurrency and retries to avoid unnecessary token usage or runaway API costs.
- Show meaningful processing status and failures without exposing API keys or private reasoning.
- Include deterministic tests for all numeric calculations.
- Implement and test the SECURITY AND PRIVACY SAFEGUARDS section; do not add JWT/OAuth to this single-user assignment MVP.
- Do not claim a feature is complete until it has been tested.

---

## RUN

**Claude Code workflow (if the PRP commands are available in the installed template):**

```bash
/generate-prp INITIAL.md
/execute-prp PRPs/[generated-name]-prp.md
```

**Application run command (after implementation):**

```bash
streamlit run app.py
```

**Final delivery:** Working Streamlit app, source code, dependencies, example CSV, test results, README, enriched CSV output and a weekly insights report suitable for a short community demo.
