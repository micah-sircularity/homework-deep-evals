<img src="assets/icon.svg" width="64" height="64" alt="">

# homework-deep-evals

Created in [T3 Code](https://t3.codes).

## Setup

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
```

The starter files include the recipe collection, search tools, recipe agent,
scenario runner, and judge. Run commands from the repository root so the scripts
can find their data files.

Copy `.env.example` to `.env` and set `FIREWORKS_API_KEY`, or export it in
your shell. Both the recipe agent and judge use Pydantic AI with Fireworks AI
and `accounts/fireworks/models/glm-5p3-flash`.

The runner writes full answers, elapsed time, aggregate input, output, total and
cached input tokens, model request count, and estimated USD cost to `results.json`
and a standalone `results.html` review report. The terminal shows brief progress
and final totals.
Usage includes all model responses in each agent run, including tool follow-ups.

GLM 5.3 Flash serverless estimates use $0.15 / $0.03 / $0.50 per million
uncached input / cached input / output tokens, verified 2026-10-31 against the
[Fireworks model page](https://fireworks.ai/models/fireworks/glm-5p3-flash).
These estimates are not invoice charges; rates and their source are saved with
results and should be reviewed if pricing changes.

Try the agent:

```bash
uv run python -c 'from recipe_agent import agent; print(agent.run_sync("Find an Italian recipe").output)'
```

The included `scenarios.csv` contains 21 scenarios with `question`, `category`,
and `type` columns. To run evaluations:

```bash
uv run python run_scenarios.py
```

The runner validates all scenario rows before making calls, then replaces
`results.json` for the new run and saves it atomically after each successful
scenario. It stops with a nonzero exit status on an API failure and preserves
completed results. Failed request usage and cost are unknown and excluded from
the completed-scenario totals. Each scenario starts with a fresh conversation.

After labeling the answers in HTML, export reviewed JSON as
`results_reviewed.json`, then run `uv run python judge.py` to compare the AI
judge against your review. Model calls incur API usage charges.

Run offline runner tests with `uv run python -m unittest discover -s tests -v`.

## Review and label results

After running `uv run python run_scenarios.py`, open `results.html` in your browser.
Each result includes the question, answer, usage, timing, and estimated cost.
Expand the recipe collection to validate claims, select **Good** or **Bad**, and
write validation notes. Search and label filters help you work through the results.
Labels save in browser storage when available; this does not modify the source
JSON or HTML. Export reviewed JSON or spreadsheet CSV to keep a portable copy.
Storage is specific to the browser and run, so export before moving the report.

To rebuild an HTML report from existing results without API calls:

```bash
uv run python review_results.py
```

You can also review automated judge results:

```bash
uv run python review_results.py results_judged.json --output results_judged.html
```

The JSON export retains the original result fields and adds `human_label` and
`human_notes`. CSV exports include these annotations and flattened usage columns.

## Compare the AI judge with your review

`results_reviewed.json` contains the original 21 human-labeled answers. The judge
uses a shared rubric refined from your comments and the full recipe collection;
it never receives an individual answer's human label or notes. It evaluates the
saved answers without calling the recipe agent again.

```bash
uv run python judge.py
# Or select another reviewed dataset:
uv run python judge.py results_reviewed.json --output results_judged.json
```

Open `results_judged.html` for side-by-side labels, comments, AI reasoning,
disagreement filters, agreement counts and chart, and separate judge usage and
estimated cost. The initial filter shows disagreements. Original human labels
remain fixed; editable follow-up decisions and notes are stored separately in
browser storage and included in JSON/CSV exports. Export to retain a portable
copy. Comparing on the same dataset used to refine the rubric measures calibration
agreement, not held-out accuracy.

Each completed judgment is saved to JSON and HTML. An API failure stops the run
with a nonzero exit status and preserves completed judgments; its unknown usage
and cost are excluded from totals. A new invocation starts a fresh paid run.

## Track judge rubric experiments

`judge_runs/human-notes-v1/` preserves the strict first run, and
`judge_runs/human-notes-v2/` preserves the rerun with the overall decision rule
and rule 7 softened. Rule 2 is unchanged. Each folder retains its prompt, input
human reviews, recipe collection, judgments, HTML review, and metrics. The source
answers and original human labels stay fixed across runs.

Recalculate alignment and the experiment comparison without model calls:

```bash
uv run python calculate_alignment.py
```

Open `judge_runs/alignment_metrics.html` for metrics and label changes, or use
`judge_runs/alignment_metrics.json` for complete metrics, costs, token usage,
and input/prompt hashes. "Bad" is the positive class. Model sampling varies,
so a single before/after run cannot isolate the effect of the prompt change.

To resume a failed judge run without repeating saved evaluations, use
`uv run python judge.py --output <saved-results.json> --resume` with the same
input and rubric version. The checkpoint must match all original input fields.
The default response allowance is 8192 tokens; `--max-tokens` can override it.
The v2 run resumed after 17 successful evaluations when one call exceeded the
original 4096-token allowance. Its unknown failed-call cost is excluded from
reported totals and documented in the saved execution notes.
