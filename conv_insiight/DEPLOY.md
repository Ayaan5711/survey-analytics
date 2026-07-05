# ConvInsight — Deployment Guide (SVN → Jenkins → QA)

This documents how the changed files map to the **TextAnalyticsWebUtility** iON ML
Framework project and how to promote them. Everything ships through the **same**
Maven/SVN/Jenkins pipeline as the existing ConvInsight backend (build #126) — no
new infrastructure, no new files.

## 1. Architecture (as deployed)

```
Browser
  │  relative URL  /TextAnalyticsWebUtility-WS/ConvInsightServlet
  ▼
Java WAR  (TextAnalyticsWebUtility)  — Tomcat/WebSphere
  - ConvInsightServlet.java  (thin proxy)
  - WebContent/  surveyinsight.html / css / javascript
  │  server-to-server HTTP → 127.0.0.1:8001
  ▼
Python (FastAPI) under GUNICORN (uvicorn workers) on localhost:8001
  - app.py / agent.py / session_store.py / code_executor.py
  - started/stopped by TAWeb_findGunicornPid.sh / TAWeb_killGunicornPid.sh
```
- The Python is an **internal** service (localhost); only the servlet calls it.
- Gunicorn typically runs **multiple workers** → sessions are persisted to local
  disk (see session_store.DATA_DIR) so any worker can serve any request and they
  survive a restart. (This is why the in-memory version was replaced.)

## 2. Files changed → SVN / workspace locations

Branch: `/SolutionEngg/CI/iON_ML_Framework/Branches/Major/TextAnalyticsWebUtility/`

| File (in this repo) | Commit to (workspace path) |
|---|---|
| `text_analytics-utility/app.py`            | `src/main/resources/app.py` |
| `text_analytics-utility/agent.py`          | `src/main/resources/agent.py` |
| `text_analytics-utility/session_store.py`  | `src/main/resources/session_store.py` |
| `text_analytics-utility/code_executor.py`  | `src/main/resources/code_executor.py` |
| `text_analytics-utility/requirements.txt`  | `src/main/resources/requirements.txt` |
| `text_analytics-ws/ConvInsightServlet.java`| `src/main/java/com/tcsion/textanalyticsweb/ws/service/ConvInsightServlet.java` |
| `text_analytics-ws/surveyinsight.html`     | `WebContent/surveyinsight.html` |
| `text_analytics-ws/surveyinsight.js`       | `WebContent/javascript/surveyinsight.js` |
| `text_analytics-ws/surveryinsight.css`     | `WebContent/css/surveryinsight.css` |

All are **edits to existing files** — no new files were added.

## 3. TAWebResourceFileList.java — NO change needed

`com.tcs.ion.textanalyticsweb.utils.TAWebResourceFileList` already lists all our
resource files (`agent.py`, `app.py`, `code_executor.py`, `requirements.txt`,
`session_store.py`). Because we only edited existing files and added none, this
registry does **not** need modifying. (If a new .py file were ever added, it would
have to be registered here.)

## 4. Deploy steps

1. Commit the files above to SVN on the `Major` branch (single revision,
   e.g. "ConvInsight: DuckDB multi-file + deterministic tools + dashboard/PDF").
2. Jenkins → **TextAnalyticsWebUtility** → **Build with Parameters** (#131).
3. Promotion: **1. Approve to Dev Env** → **2. Promote to QA Env**.
4. On promote, the framework extracts the resource-listed files and the
   `.sh` scripts restart gunicorn → new code is live on :8001.

## 5. Two things to verify before/after promote

1. **Python dependencies** — we added 3 libraries to `requirements.txt`:
   `duckdb`, `pyarrow`, `fpdf2`. Confirm the hosting runs
   `pip install -r requirements.txt` and that it can reach them:
   - public PyPI → they download automatically.
   - internal mirror/Artifactory → **request these 3 be added to the mirror first.**
   (Do not remove any existing requirement lines — additions only.)
2. **Gunicorn worker `--timeout`** — the LLM chat already runs here and works, so
   the timeout is already generous; the new deterministic tools are faster than
   the old code-gen path, so no change is expected. Just confirm the start command
   (autodeploy config / start script) hasn't a short default.

## 6. Credentials (QA)

Azure OpenAI settings are kept **inline** in `agent.py` (QA env cannot store
secrets externally). Set the real key in `agent.py` before promote:
```python
api_key = "REPLACE_WITH_AZURE_OPENAI_KEY"   # -> paste the QA key
```
TODO for higher environments: switch to `os.getenv(...)`.

## 7. Smoke test after QA promote

1. Open the app → upload **one** CSV/Excel → confirm dashboard + auto-insights.
2. Upload **two same-schema files** → confirm combined row count = sum of both.
3. Ask "pie chart of <a categorical column>" → chart + exact table returned.
4. Click **Dashboard** → stat tiles + quality + charts render.
5. Click **Export Report (PDF)** → a PDF downloads.
6. Refresh the page → "Resumed your previous session" (localStorage) and chat
   still answers (server reloaded the session from disk).
