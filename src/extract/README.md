# Phase 2 — Haddon/ICECI event extraction

Phase 1 (`src/classify/`) answers *was this a real child injury event?* Phase 2 answers *how did it unfold?* For each confirmed event it extracts 24 structured fields across the Haddon pre-event / event / post-event phases plus four interpretive fields. Every field carries a verbatim evidence span and a `stated | inferred | not_stated` status.

> **ICECI provenance warning.** The controlled vocabularies in `schema.py` are **adapted from** the ICECI framework and extended with child-specific values ICECI's core does not carry (button batteries, sleep surfaces, developmental triggers, supervision granularity). They have **not** been transcribed from the canonical ICECI data dictionary. Before any manuscript claims ICECI coding, check each enumeration against the published ICECI documentation and state the adaptation explicitly in the Methods. WHO no longer maintains ICECI and has carried its elements into ICD-11 as extension codes ([WHO](https://who.int/standards/classifications/other-classifications/international-classification-of-external-causes-of-injury)), so the check should also cover the ICD-11 external-cause extension codes.

## Files

| File | Purpose |
|---|---|
| `schema.py` | Fields, vocabularies, guardrails, prompt-spec renderer. Single source of truth; no OpenAI dependency. |
| `common.py` | Shared I/O helpers (Phase 1 helpers copied verbatim, plus label reading and the `selftext` guard). |
| `filter_confirmed.py` | Writes the confirmed-only analytic dataset; asserts the count against Phase 1's timing JSON. |
| `sample_pilot.py` | Purposive stratified sampler (stratified on Phase 1 `mechanism_of_injury`) for the pilot and the human-coding samples; `--exclude` drops posts already used. |
| `extract_events_openai.py` | Extraction runner. |
| `validate_extractions.py` | Automated QA tier. |
| `compare_extractions.py` | Side-by-side comparison of runs on the same posts (model choice) and the hand-review sheet. |
| `make_review_md.py` | Builds the single-reviewer hand-review file: definitions, then each post with the model's answers (Markdown, local only). |
| `export_coding_app.py` | Exports the codebook, the tutorial and a human-coding sample (title, body, permalink only; no model output or Phase 1 labels) as JSON documents for the coding app. Written under `reports/coding_app/`. |
| `tutorial.json` | Eight constructed practice posts with an answer key and a reason per field, for the coding app's Tutorial tab. Invented text, so it is committed. Tests check every key against `schema.py`; a schema bump fails them until the key is re-checked and its `schema_version` updated. |
| `make_review_workbook.py` | Builds the multi-reviewer hand-review Excel workbook with per-cell verdicts; protocol in `HAND_REVIEW.md`. Needs `openpyxl`. |

Tests: `python -m unittest tests/test_extract_schema.py` from the repo root.

## Order of operations

```bash
# 0. confirmed-only dataset, count asserted against Phase 1 timing JSON
python src/extract/filter_confirmed.py \
  --in data/interim/reddit_child_injury_20251112_labels.jsonl

# 1. purposive stratified pilot (v1.1: 60 posts, falls boosted so the fall-only fields can be assessed;
#    a strict superset of the v1.0 pilot50, so results stay comparable across schema versions)
python src/extract/sample_pilot.py \
  --in data/interim/confirmed_events_n1237.jsonl --n 60 --per-stratum 4 --boost fall=10

# 2. head-to-head on the same 50 posts
python src/extract/extract_events_openai.py \
  --in data/interim/pilot50_stratified.jsonl --model gpt-5-nano --tag nano
python src/extract/extract_events_openai.py \
  --in data/interim/pilot50_stratified.jsonl --model gpt-4.1-mini --tag mini

# 3. QA each run
python src/extract/validate_extractions.py \
  --in data/interim/pilot50_stratified_extractions_nano.jsonl --outdir reports/extraction_qa/nano
python src/extract/validate_extractions.py \
  --in data/interim/pilot50_stratified_extractions_mini.jsonl --outdir reports/extraction_qa/mini
```

# 4. side by side + hand-review sheet
python src/extract/compare_extractions.py \
  --run nano=data/interim/pilot50_stratified_extractions_nano.jsonl \
  --run mini=data/interim/pilot50_stratified_extractions_mini.jsonl
```

Extractor options added in v1.1: `--workers N` runs concurrent API calls while keeping output order; `--reasoning-effort minimal|low|medium|high` applies to GPT-5-family models only. Changing `--reasoning-effort` changes the model's behaviour, so record it and keep it fixed across the pilot comparison and the full run.

Then **read every pilot output by hand**. The QA numbers only tell you where to look. Revise `schema.py`, bump `SCHEMA_VERSION` (with a dated `SCHEMA_HISTORY` entry), and only then run the full set.

The human-coding samples reuse the sampler with `--exclude`, which drops posts used elsewhere before stratifying. It takes a sample JSONL, a `*_manifest.json` or a text file with one post id per line, and can be repeated. The deep tier must exclude the 60 pilot posts, because the study lead has seen the model's answers for them, and the coder's practice set:

```bash
python src/extract/sample_pilot.py --name deeptier --n <N> \
  --exclude data/interim/pilot60_stratified.jsonl \
  --exclude data/interim/practice<K>_stratified.jsonl
```

The manifest records the excluded files and how many posts each removed. Stratum sizes are reported after exclusion.

Compare the two models on unsupported-span rate, per-field `not_stated` rate and enum-violation rate. `gpt-5-nano` is the default only for consistency with Phase 1. Whichever model wins, the timing JSON records the requested model, the API-resolved model version (`model_resolved`), temperature, schema version and run date.

## Inputs

Read the Phase 1 **JSONL**, never the CSV. The JSONL keeps the full crawler record including `selftext`, while the CSV has no post body. Every script asserts `selftext` is present on the first record and exits if it is not. Labels are read from the literal key `"**labels"` (falling back to `"labels"` or a flat record).

Phase 1's `rationale_short` is **not** sent to the Phase 2 model, so extraction is not anchored to Phase 1's answer. It is carried into the extraction CSV as `phase1_model_rationale`, as a reference for human coders only. It is the Phase 1 model's note, not a human judgement.

## Pilot design

The pilot is **purposive, not random**. Every stratum gets up to `--per-stratum` posts (which over-samples rare mechanisms), then the sample is topped up to `--n` proportionally from the remaining posts. If the floor alone exceeds `--n` (13 strata × 4 = 52 > 50 on the 2025-11-12 data), the floor is trimmed from the largest strata first. A `*_manifest.json` records the seed, stratum sizes, composition and IDs for the Methods.

## Outputs

Extractor (named `<input_stem>_extractions[_<tag>]`):
- `.csv`: one row per event. Columns `<field>`, `<field>__status`, `<field>__span`, `n_violations`, `violation_codes`. Multi-valued `help_seeking` is `|`-joined.
- `.jsonl`: nested cells, full violation list, post text and Phase 1 labels.
- `_timing.json`: run metadata and violation tally.

QA (`reports/extraction_qa/`): `field_coverage.csv`, `violations_by_field.csv`, `qa_summary.json`.

## Violation codes

Violations are logged and never silently repaired. Their rates are the QA output.

| Code | Meaning |
|---|---|
| `malformed_cell` | Cell missing or not an object → blanked |
| `bad_status` | Status not `stated`/`inferred`/`not_stated` → treated as `not_stated` |
| `enum_snapped` | Unambiguous near-miss snapped to the allowed value |
| `enum_ambiguous` | Near-miss equidistant from ≥2 values (e.g. `ed_presentation`, bare `fall`) → refused, `unspecified` |
| `enum_violation` | No match → `unspecified` (single-valued) or member dropped (multi-valued) |
| `stated_without_span` | `stated` with no span → downgraded to `inferred` |
| `unsupported_span` | `stated` span not found in the post → downgraded to `inferred` |
| `inferred_span_unsupported` | `inferred` span not found in the post (logged; status unchanged) |
| `conditional_violation` | Fall field populated for a non-fall mechanism → cleared (the cleared value is kept in `detail`) |
| `category_without_object` | `object_category` set but `object_involved` empty |
| `extra_field` | Model returned a key not in the schema |
| `parser_error` | Model output was not parseable JSON → all fields blank |
| `key_recovered` | Key was phase-prefixed (`pre_event.x`), nested under a phase, whitespace-padded, or a lone phase-named cell when exactly one field of that phase was missing → mapped back to the field |
| `elided_span` | Span joins passages with "..." or line breaks, and every passage (≥ 3 words) is verbatim → accepted, status unchanged |

Span matching is substring containment after normalising whitespace and case. It also folds curly quotes and decodes HTML entities, because about 60% of confirmed posts contain curly quotes that models straighten when copying.

## QA cut rule and scope

The QA script flags any field above 80% `not_stated` for redefinition or cutting **before** the full run. Two exceptions apply:

- `fall_height` and `landing_surface` are judged over fall events only, since they are blank by design for every other event.
- `care_pathway_outcome`, `countermeasure_adopted` and `presenting_signs` are reported as `scope_limited` rather than flagged, because comment threads were never collected (see below). Judge them on whether the captured values are correct.
- `developmental_trigger` and `stated_counterfactual` are reported as `rare_by_design`. Reading the v1.0 pilot by hand showed that real values are rare (4–5 of 50 posts carry a developmental trigger), so a high `not_stated` rate is the correct answer.
- A field that applies to fewer than 10 events is reported as `insufficient_n` rather than recommended for cutting.

Watch `supervision_status` above all else. The prevention argument leans on it hardest, and parenting narratives most often leave it implicit.

### Scope: originating post only

Extraction uses the post title and body. Comment threads were never collected (the crawler stores `num_comments` only). This was deliberate: the originating post is the parent's own account, which is also cleaner on consent than including third-party commenters. Suggested Limitations wording:

> Extraction was restricted to the originating post; comment threads were not collected, so post-event detail — particularly care pathways and adopted countermeasures — is likely under-captured relative to what the full thread would contain.

## Phase 1 audit (`injury_event_check`, from v1.2)

Phase 2 trusts Phase 1's `is_injury_event`, so Phase 1's errors flow straight into the extraction. A 25th field, `injury_event_check`, sits outside the 24 extraction fields. It records the extraction model's own judgement (`actual_injury_event` / `suspected_injury_event` / `near_miss` / `not_an_injury_event` / `unclear`), using Phase 1's definition of an injury event, with an evidence span like every other field. The prompt never says that Phase 1 accepted the post, so the check is independent. On these posts, `not_an_injury_event` marks a candidate Phase 1 false positive.

`near_miss` (from v1.4) is a real hazardous incident with no injury and none suspected: caught before falling, a pebble taken out of the mouth, pulled out of the pool. Phase 1's prompt defines an injury event as a *harmful* incident but never mentions near misses, and it kept some. Before v1.4 they were lumped with off-topic posts under `not_an_injury_event`. They are reported separately because whether they belong in the analysis is a study decision, not a Phase 1 error.

From v1.5 the line between the two is the hazard reaching the child, whatever the outcome. A stated ingestion or excess dose with no symptoms is `actual_injury_event`, and so is submersion with coughing, spluttering, blue or limp (non-fatal drowning under the WHO 2002 definition). A face under water, lifted out at once with no distress, is `near_miss`. Worry alone is not suspicion, and a side effect of a correctly given dose is `not_an_injury_event`. Phase 1's definition never decided these cases, so they are settled here and apply to the Phase 1 audit as well.

`validate_extractions.py` prints the audit and writes `audit_disagreements.csv` (candidate false positives and near misses, one row each). These are **candidates**: model disagreement is not a ground-truth error, so a human must read each one before any rate is cited. The audit judgement does not filter or blank the extraction. Excluding non-events from the analysis is a decision to make after the human check.

The pilot's accepted posts are purposively stratified and over-sample Phase 1's `other`/`unknown` mechanisms, where false positives concentrate. The pilot false-positive rate therefore overstates the rate across all 1,237 posts. Estimate that rate from the full run, or from a random sample.

## Re-scoring without re-calling the API

From v1.1, each JSONL record keeps `raw_response` (the model's raw output) and `usage` (token counts). After a guardrail change, re-apply `normalize_extraction(json.loads(raw_response), title + "\n" + selftext)` instead of paying for a re-run. A change to the prompt or vocabulary still needs a re-run. v1.0 outputs have no `raw_response`.

## Schema versioning

`SCHEMA_VERSION` is written into every output record as `extraction_schema_version`. It is a separate key from the crawler's own `schema_version`. Bump it on any change to field names or enumerations and add a dated `SCHEMA_HISTORY` entry. Undocumented drift between coding rounds invalidates the human–model agreement statistics.
