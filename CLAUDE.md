# Reddit child-injury study

Parenting posts on Reddit, analysed for child injury events.

- **Phase 1** screened posts with an LLM classifier (`src/classify/`). It produced **1,237 confirmed injury events** (`data/interim/confirmed_events_n1237.jsonl`), each labelled with a mechanism.
- **Phase 2** (`src/extract/`) extracts a structured record from each post with an LLM. The schema has 25 fields: Haddon-matrix fields with ICECI-adapted values, interpretive fields, and a Phase 1 audit check. The extraction is then checked against blind human coding.

## Hard rules

- **This repo is public.** Never commit `data/`, `reports/`, post text, model outputs, or personal emails. The only post text allowed in git is the invented tutorial posts in `src/extract/tutorial.json`.
- **Code is the source of truth.** If `src/extract/schema.py` and any document disagree, the schema wins.
- **The schema is frozen at 1.6.0.** Change field names, definitions or value lists only by a dated amendment: bump `SCHEMA_VERSION`, add a `SCHEMA_HISTORY` entry, then re-export the codebook. Undocumented drift invalidates the human–model comparison.
- Commit or push only when the study lead asks.

## Where things are documented

| What | Where |
|---|---|
| Schema, definitions, value lists, dated change log | `src/extract/schema.py` (`SCHEMA_HISTORY`) |
| Phase 2 methods and settled coding rules | `src/extract/README.md` |
| Human-coding design, step order, coding app usage | `src/extract/HAND_REVIEW.md` |
| Coding app: roles, loading, backups, tests | `coding_app/README.md` |
| How each sample was drawn (seed, exclusions, swaps) | `data/interim/*_manifest.json` (not in git) |
| Readable codebook | `reports/codebook_<version>.md` (generated, not in git) |
| Study design for readers | The study guide, a Claude Doc kept outside the repo. Ask the study lead for the link. |

## Key decisions

| Date | Decision | Recorded in |
|---|---|---|
| 2026-10-28 | `near_miss` added to `injury_event_check` (schema 1.4.0): a real hazardous incident stopped before the hazard reached the child. | schema.py, README |
| 2026-10-29 | Human coding is **one blind coder plus a reviewer**, coding to consensus (the in-app review was dropped on 2026-11-10, see below). The study reports agreement between the agreed human coding and the model (kappa or PABAK, span precision and recall), not inter-rater reliability. Don't propose double coding. | HAND_REVIEW, study guide |
| 2026-10-30 | Schema 1.5.0: the incident is the hazard reaching the child, whatever the outcome. A stated ingestion or excess dose is an actual event even with no symptoms. Coming out of water coughing, blue or limp is an actual event (WHO non-fatal drowning). A face under water, lifted out at once with no distress, is a near miss; the test is "couldn't get out unaided", not age. Worry alone is not suspicion. | schema.py, README, HAND_REVIEW |
| 2026-10-30 | Schema **frozen at 1.5.0** after the study lead's review of the 60 pilot posts. | schema.py |
| 2026-10-30 | The human coding app moved from a claude.ai artifact to **Firebase** (https://human-coding.web.app, project `human-coding`, Firestore in Sydney). The database rules enforce allowlisted roles: coder, reviewer, admin. | coding_app/README.md |
| 2026-10-30 | Practice set: 13 posts, one per Phase 1 mechanism group, seed 2026, pilot posts excluded. The drowning post `1ehlo1h` was swapped for a second fall post `1k2zp6e`, which keeps both remaining drowning posts for the deep tier. | practice13 manifest (`swap` key) |
| 2026-11-02 | Sign-in emails are capped at 5 a day on the free Spark plan, so coders sign in with **Google**. Sign-in links always return to the official app address. | coding_app/public/backend.js |
| 2026-11-04 | **Near misses are included in the analysis.** | README, study guide |
| 2026-11-11 | Near misses are **reported as their own category**, not pooled with injury events. | README, study guide |
| 2026-11-10 | Schema **1.6.0** after the practice round. Scope: **unintentional injury to children under 5**. Intentional harm (assault, maltreatment, self-harm), a child clearly aged 5 or older, and illness with no external cause are `not_an_injury_event`; an unstated age is not grounds for exclusion. Harm from another child under 5 counts as unintentional. A child unhurt in an impact that injured only an adult (e.g. head-butting a parent) is `near_miss`. | schema.py, README, HAND_REVIEW |
| 2026-11-10 | **Skip rule:** in human coding, `injury_event_check` comes first, and a non-event leaves every other field `not_stated` (the app locks them). The model still codes every field. Field-level agreement covers only posts the human coding treats as events. | schema.py (`HUMAN_SKIP_VALUES`), coding app, README |
| 2026-11-10 | **No field-by-field review in the app.** The study lead checks the coder's work outside the app (Progress → Download submitted coding), and the coder corrects it with Reopen to edit. The reference standard is the coder's final submitted coding. | HAND_REVIEW |
| 2026-11-10 | **Deep tier: 120 posts**, stratified by Phase 1 mechanism (floor 8 per group, seed 2026), with the pilot and practice posts excluded. It is loaded in the app as sample `deeptier`, in a seeded shuffled coding order so the mechanisms are interleaved. The study lead checks the first 20–25 posts before the coder continues. | `data/interim/deeptier120_stratified_manifest.json` (`coding_order`) |

## Current stage and next steps

1. Done: the practice round (13 posts, median 19 min per post) led to schema 1.6.0, which is live in the app.
2. **Now:** the coder codes the 120-post deep tier (sample `deeptier`) blind. The study lead checks the first 20–25 posts (Progress → Download submitted coding) before the coder continues, then checks the rest outside the app.
3. In parallel, the study lead runs the **full model run** on the 1,237 posts at 1.6.0. The coder never sees model output.
4. Keep the schema frozen at 1.6.0 for the whole deep tier.
5. Compute agreement: `injury_event_check` across all 120 posts; the other fields across posts the human coding treats as events.
6. Later: a targeted tier on `supervision_status`, `object_involved`, `mechanism_detail` and `care_pathway_outcome`.

Open: the ICECI value lists are *adapted*, not transcribed: check them against ICECI and the ICD-11 external-cause extension codes before the paper claims ICECI coding.

## Commands

```bash
python3 -m unittest discover -s tests            # schema tests
python3 src/extract/sample_pilot.py --help     # stratified samples (seeded)
python3 src/extract/export_coding_app.py --posts <sample.jsonl> --sample <name>
python3 src/extract/render_codebook_md.py      # readable codebook -> reports/codebook_<version>.md
cd coding_app && npm test                      # app smoke test, rules and sign-in tests on the emulators
cd coding_app && npm run deploy                # needs firebase login
```
