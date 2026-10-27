# Phase 2 pilot — hand-review protocol

> **Current pilot review: use the Markdown file.** With one reviewer, the pilot is checked using a single Markdown file: a definitions reference, then each post with the model's answers, and a **My notes** space per post. Build it with:
>
> ```bash
> python src/extract/make_review_md.py --in data/interim/pilot60_stratified_extractions_gpt55.jsonl
> ```
>
> It writes `reports/extraction_qa/hand_review/<input stem>_review.md` (git-ignored, because it holds full post text). Open it in a Markdown preview so the tables render and the field links work. The workbook protocol below is kept for when more than one reviewer is involved; its reasoning (read the post first, prefer `not_stated`, where to spend effort) applies to both.

This protocol is for whoever checks the pilot extractions against the posts. The workbook's `Instructions` sheet is the short version. This file gives the reasons behind each step.

**Read the study guide first:** [Phase 2 Study Guide and Codebook](https://claude.ai/artifact/AmnHgTaUPuqYjf5fYBct3W). It covers what the study is, the status decision, the coding notes that settle the common hard cases, and a worked example. This protocol assumes you have read it. If the link doesn't open, ask the study lead to share it with you.

**Point of contact:** Nipuna Cooray (study lead). Raise questions, ambiguities and difficult days with them; agree a daily cap before you start.

**Getting the workbook.** It contains full post text, so it is never committed (`reports/` is git-ignored). Either build it yourself (below; this needs the pilot output files) or get a copy from the study lead. Keep it on your own machine, and don't email or upload it unless the study lead says otherwise.

## What this review is, and is not

This review checks one model's extraction against each post, cell by cell. It has three purposes:

1. **Lock the schema.** Find fields, definitions and values that don't work before the full run.
2. **Confirm the model before the full run.** In particular, check whether its inferences are justified by the post.
3. **Settle the audit edge cases.** Decide how borderline posts count for `injury_event_check`, and therefore for Phase 1.

It is **not** the reliability study the manuscript cites. The pilot posts shaped the schema, so they can't also validate it. Reliability comes later: a fresh **stratified** sample from the full run, coded **blind** by two people, with agreement statistics reported. See the last section.

## Build the workbook

```bash
python src/extract/make_review_workbook.py --primary gpt55 \
  --run gpt55=data/interim/pilot60_stratified_extractions_gpt55.jsonl \
  --run nano=data/interim/pilot60_stratified_extractions_nano.jsonl \
  --run mini54=data/interim/pilot60_stratified_extractions_mini54.jsonl
```

This writes `reports/extraction_qa/hand_review/hand_review_gpt55.xlsx`. It needs `openpyxl`. Rebuilding overwrites the file, so copy a filled-in workbook elsewhere before rebuilding.

| Sheet | Use |
|---|---|
| `Instructions` | Short protocol, verdict codes and an example of a filled-in row. Enter your name in **Reviewer**. |
| `Summary` | Live progress and per-field accuracy. It updates as you fill in verdicts; don't edit it. |
| `Posts` | Full post text. Long posts continue on the following row(s). |
| `Review` | One row per post × field. You fill in the yellow columns N–P. |
| `Codebook` | Definition and allowed values for every field. |
| `Decision log` | Rules you decide. It is seeded with the open questions from the pilot. |

Only edit **yellow** cells. Everything else is generated.

## Procedure, per post

1. **Read the post in `Posts` first, before looking at any model output.** Form your own view: what happened, who was supervising and what they were doing, where it happened, who saw it, what help was sought, and whether this is really a child injury event. Reading first is the only protection against being steered by the model.
2. **Filter `Review` column A to that post.** For each row, check the model's value, status and quote against the post, then choose a **Verdict**.
3. **Columns L–M show the other models' values, and only where they disagree.** They're left blank elsewhere on purpose. Where they disagree, ask whether the model is wrong or the definition is ambiguous. If it's ambiguous, record a rule in the `Decision log`.
4. **For `wrong_value` or `missed`, enter the right value in "Correct value".** Enumerated fields offer a dropdown. For `help_seeking`, type the values joined with `|`. Use **Note** for anything a second reader would need.

## Verdicts

| Verdict | Use when |
|---|---|
| `correct` | Right value and right status, and the quote supports it. A `not_stated` cell is correct if the post really doesn't say. |
| `partially_correct` | A multi-valued or free-text cell that is part right, such as one of two help sources. Put the full right answer in "Correct value". It counts as not correct in accuracy. |
| `wrong_value` | The post supports a different value. |
| `over_inferred` | The model filled the field but the post doesn't support it; it should be `not_stated`. |
| `missed` | `not_stated`, but the post plainly answers it. |
| `wrong_status` | Right value, but `stated` vs. `inferred` is wrong. |
| `span_mismatch` | The quote is real but doesn't support the value. |
| `unsure` | You can't decide. Explain in Note and add a Decision log entry. It is excluded from accuracy. |

`over_inferred` and `missed` matter most. They're where the models differed in the pilot, and they decide both the model choice and which fields need redefining. Apply the brief's rule strictly: **prefer `not_stated` over a guess**, and give no credit for inferences drawn from typical parenting context rather than from this post.

## Where to spend effort

Tier 1 rows are shaded and should be reviewed on **every** post. They are:

- the priority fields: `supervision_status`, `injury_event_check`, `mechanism_iceci`, `care_pathway_outcome`, `witnessed`, `fall_height`, `landing_surface`
- any field where the models disagree

Tier 2 rows need only a quick check. If time is short, finish tier 1 on all posts before starting tier 2 on any.

- **`supervision_status`:** check every *inferred* value against the Codebook glosses. The prevention argument rests on this field, and `gpt-5.5` infers 43% of its values.
- **`injury_event_check`:** the 24 posts that any model flagged are the ones that matter. Answer the open questions in the Decision log; the same rules then apply to Phase 1's case definition. Status rule (schema 1.3.0): `stated` if the post explicitly describes the incident and the harm, or explicitly says no harm occurred; `inferred` if the judgement rests on the post as a whole. This field is never `not_stated`; a post you can't judge is `unclear`.
- **Free-text fields:** check the quote supports the value. Don't spend time on wording.

## Decision log

Every time a definition doesn't settle a case, write one row:
- **Situation:** what was unclear.
- **Rule decided:** the general rule, not just this post's answer.
- **Example post ID**.
- **Schema change needed?** `Y` if the rule should change a definition or a value list in `schema.py`.

Rows marked `OPEN` are questions the pilot raised but didn't settle. Each needs a rule. The `EXAMPLE` row shows the format.

The log becomes the codebook, the `schema.py` changes and the Methods text, so write rules that someone else could apply.

## Pacing

- Do **posts 1–10 first**, writing Decision log rules as you go; most rules come out of the first batch. Then do the remaining 50 in two or three sittings.
- Expect about 5–6 minutes per post, roughly 5–6 hours in total.
- Tick `Posts!F` as you finish each post.

## Afterwards

1. `Summary` gives per-field accuracy plus the `over_inferred` and `missed` counts. Those decide whether `gpt-5.5` is good enough and which fields need work.
2. The Decision log rows marked `Schema change needed = Y` become `schema.py` changes. Bump `SCHEMA_VERSION` and add a dated `SCHEMA_HISTORY` entry.
3. Re-run the same 60 posts on the chosen model (about $7 on `gpt-5.5`) to confirm the fixes, then run the full 1,237 posts.

## The formal reliability study (later)

This follows the study guide's "What coders do" section, which is the authoritative design. After the full run:

1. **Timing check.** Time blind coding on a small set of posts. Verification is faster than coding from scratch, so times from this review underestimate it. That figure sets how large the deep tier can be.
2. **Deep tier.** Draw a **stratified** sample that over-samples rarer mechanisms rather than reproducing the corpus distribution. Exclude the 60 pilot posts, because this reviewer has seen the model's output for them. Two coders each code all fields **blind**, from the narrative alone, using the locked codebook.
3. **Adjudication.** Work through disagreements and log each one with its resolution.
4. **Statistics.** Cohen's kappa or PABAK for enumerated fields, and span-level precision and recall for free-text fields: between the coders, and between each coder and the model.
5. **Targeted tier.** A larger sample on the four fields the conclusions rest on: `supervision_status`, `object_involved`, `mechanism_detail` and `care_pathway_outcome`.

The coding there differs from this review. Coders write each cell's value, status and quote themselves rather than judging the model's, so that study needs a blind workbook with no model columns; ask for one when you get there.
