#!/usr/bin/env python3
"""
Phase 2 extraction schema: Haddon-matrix fields, controlled vocabularies and
client-side guardrails. Single source of truth for the prompt, CSV header,
codebook and QA script.

No OpenAI dependency — pure data + validation logic, so it can be unit-tested
and imported by validate_extractions.py.

ICECI PROVENANCE WARNING
------------------------
The controlled vocabularies below are ADAPTED FROM the ICECI framework
(International Classification of External Causes of Injuries) and extended with
child-specific values ICECI's core does not carry (button batteries, sleep
surfaces, developmental triggers, supervision granularity). They have NOT been
transcribed from the canonical ICECI data dictionary. Before any manuscript
claims ICECI coding, check each enumeration against the published ICECI
documentation and state the adaptation explicitly in the Methods. WHO no longer
maintains ICECI and has carried its elements into ICD-11 as extension codes, so
the check should also cover the ICD-11 external-cause extension codes.

SCHEMA VERSIONING
-----------------
SCHEMA_VERSION is written into every output record. Bump it on ANY change to
field names or enumerations, and add a dated entry to SCHEMA_HISTORY.
Undocumented drift between coding rounds invalidates the reliability statistics.
"""

import difflib
import html
import re
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.4.0"
SCHEMA_HISTORY = [
    ("1.0.0", "2026-10-21", "Initial 24-field Haddon/ICECI-adapted schema for the n=50 pilot."),
    ("1.1.0", "2026-10-21", "After n=50 nano/mini pilot: removed developmental_trigger value 'none_stated' "
                            "(confused with status not_stated by both models); tightened definitions of "
                            "developmental_trigger, stated_counterfactual, countermeasure_adopted; prompt asks "
                            "for one contiguous span; guardrails recover prefixed/padded keys (key_recovered) "
                            "and accept ellipsis-joined verbatim segments (elided_span)."),
    ("1.2.0", "2026-10-21", "Added audit field injury_event_check (outside the 24 extraction fields): the "
                            "extraction model's independent judgement of Phase 1's is_injury_event, using "
                            "Phase 1's definition, to flag candidate Phase 1 false positives. System prompt "
                            "rule 1 no longer states the injury determination is settled."),
    ("1.3.0", "2026-10-22", "injury_event_check definition gains a status rule (stated = incident and harm, or "
                            "their absence, described explicitly; inferred = judged from the post as a whole; "
                            "never not_stated, use 'unclear') and says the judgement goes in value, not status. "
                            "Prompt text only; the n=60 pilot outputs under hand review are 1.2.0."),
    ("1.4.0", "2026-10-28", "injury_event_check gains 'near_miss' (a real hazardous incident with no injury "
                            "and none suspected). Phase 1's prompt never addressed near misses and kept some; "
                            "at 1.2.0 they were lumped with off-topic posts under not_an_injury_event. "
                            "The validator reports them separately from candidate false positives."),
]

FREE_TEXT_MAX = 300
STATUSES = ("stated", "inferred", "not_stated")

# ---------------- Field ordering (phase, cell, field) ----------------
# Prompt, CSV header and codebook all derive from this one ordering.

FIELDS: List[Tuple[str, str, str]] = [
    # Pre-event
    ("pre_event",  "host",                 "activity_when_injured"),
    ("pre_event",  "host",                 "developmental_trigger"),
    ("pre_event",  "agent",                "object_involved"),
    ("pre_event",  "agent",                "object_category"),
    ("pre_event",  "physical_environment", "place_of_occurrence"),
    ("pre_event",  "physical_environment", "micro_location"),
    ("pre_event",  "physical_environment", "hazard_state"),
    ("pre_event",  "social_environment",   "supervision_status"),
    ("pre_event",  "social_environment",   "supervisor_activity"),
    ("pre_event",  "social_environment",   "others_present"),
    # Event
    ("event",      "host",                 "child_action"),
    ("event",      "agent",                "mechanism_iceci"),
    ("event",      "agent",                "mechanism_detail"),
    ("event",      "physical_environment", "fall_height"),
    ("event",      "physical_environment", "landing_surface"),
    ("event",      "social_environment",   "witnessed"),
    # Post-event
    ("post_event", "host",                 "presenting_signs"),
    ("post_event", "social_environment",   "first_response"),
    ("post_event", "social_environment",   "help_seeking"),
    ("post_event", "social_environment",   "care_pathway_outcome"),
    # Interpretive (outside Haddon; drives the narrative synthesis)
    ("interpretive", "narrative",          "parent_attributed_cause"),
    ("interpretive", "narrative",          "stated_counterfactual"),
    ("interpretive", "narrative",          "countermeasure_adopted"),
    ("interpretive", "narrative",          "reason_for_posting"),
    # Phase 1 audit (outside the 24 extraction fields; not a Haddon cell)
    ("audit",      "phase1_check",         "injury_event_check"),
]
FIELD_NAMES = [f for _, _, f in FIELDS]
AUDIT_FIELD = "injury_event_check"
EXTRACTION_FIELD_NAMES = [f for f in FIELD_NAMES if f != AUDIT_FIELD]

PHASE_TITLES = {
    "pre_event": "PRE-EVENT",
    "event": "EVENT",
    "post_event": "POST-EVENT",
    "interpretive": "INTERPRETIVE (outside Haddon)",
    "audit": "PHASE 1 AUDIT (your independent judgement)",
}

# ---------------- Controlled vocabularies ----------------

VOCAB: Dict[str, Tuple[str, ...]] = {
    "activity_when_injured": (
        "play_unstructured", "play_with_equipment", "climbing",
        "eating_or_drinking", "feeding_being_fed", "sleeping_or_in_sleep_space",
        "bathing_or_washing", "nappy_change_or_dressing", "being_carried_or_held",
        "travelling_in_vehicle", "travelling_pram_or_carrier", "mobility_practice",
        "supervised_activity_or_care", "other_specified", "unspecified",
    ),
    "developmental_trigger": (
        "newly_rolling", "newly_sitting", "newly_crawling",
        "newly_pulling_to_stand", "newly_walking", "newly_climbing",
        "newly_opening_containers", "mouthing_phase", "unspecified",
    ),
    "object_category": (
        "furniture", "stairs_or_steps", "floor_or_ground_surface",
        "building_element", "bed_or_sleep_surface", "pram_cot_or_highchair",
        "household_appliance", "cookware_or_tableware", "hot_liquid_or_food",
        "heating_or_cooking_device", "tap_or_bath_water", "toy", "small_object_or_part",
        "button_battery", "magnet", "medicine_or_supplement",
        "cleaning_or_household_chemical", "cosmetic_or_personal_care_product",
        "plant_or_natural_substance", "animal", "tool_or_sharp_implement",
        "cord_blind_or_strap", "bag_wrapping_or_soft_material", "water_body_or_container",
        "motor_vehicle", "wheeled_non_motor", "person", "none_identified",
        "other_specified", "unspecified",
    ),
    "place_of_occurrence": (
        "home_indoor", "home_outdoor", "other_residence",
        "childcare_or_preschool", "school_or_educational", "shop_or_commercial",
        "healthcare_facility", "street_or_footpath", "playground_or_park",
        "sports_or_recreation_area", "pool_or_aquatic_area", "vehicle", "rural_or_farm",
        "other_specified", "unspecified",
    ),
    "supervision_status": (
        "direct_supervision", "proximate_supervision",
        "intermittent_supervision", "no_supervision", "supervised_by_other_adult",
        "supervised_by_minor", "unspecified",
    ),
    "others_present": (
        "alone", "sibling_present", "other_child_present",
        "one_adult_present", "multiple_adults_present", "unspecified",
    ),
    "mechanism_iceci": (
        "blunt_force_struck_by", "blunt_force_struck_against",
        "fall_same_level", "fall_from_height", "fall_from_furniture", "fall_down_stairs",
        "fall_from_carer_arms", "crushing_or_entrapment", "piercing_or_penetrating",
        "cutting", "mechanical_suffocation", "choking_airway_obstruction",
        "drowning_or_submersion", "thermal_heat_contact", "thermal_scald_liquid",
        "thermal_cold", "chemical_ingestion_or_contact", "medicine_ingestion",
        "foreign_body_insertion_or_ingestion", "electricity", "bite_or_sting",
        "transport_related", "other_specified", "unspecified",
    ),
    "witnessed": (
        "directly_witnessed", "heard_not_seen", "found_after_event",
        "witnessed_by_other", "unspecified",
    ),
    "help_seeking": (
        "none", "forum_advice_only", "family_or_friend", "pharmacist",
        "nurse_helpline", "poisons_information_centre", "telehealth_or_online",
        "general_practitioner", "emergency_department", "ambulance_called", "unspecified",
    ),
    "care_pathway_outcome": (
        "no_care_sought", "advice_only_no_visit",
        "primary_care_visit", "ed_presentation_discharged", "ed_presentation_admitted",
        "hospital_admission", "specialist_referral", "ongoing_or_pending", "unspecified",
    ),
    "reason_for_posting": (
        "seeking_reassurance", "seeking_triage_advice",
        "seeking_prevention_advice", "expressing_guilt_or_distress", "warning_others",
        "sharing_outcome_update", "seeking_product_information", "other_specified",
        "unspecified",
    ),
    "injury_event_check": (
        "actual_injury_event", "suspected_injury_event", "near_miss", "not_an_injury_event", "unclear",
    ),
}

# Audit value that disagrees with Phase 1 is_injury_event == true (candidate false positive)
AUDIT_NEGATIVE = {"not_an_injury_event"}
# Outside Phase 1's "harmful incident" definition, but whether it is in scope is a study decision,
# so it is reported separately rather than counted as a false positive
AUDIT_NEAR_MISS = {"near_miss"}

MULTI_VALUED = {"help_seeking"}

FALL_MECHANISMS = {
    "fall_same_level", "fall_from_height", "fall_from_furniture",
    "fall_down_stairs", "fall_from_carer_arms",
}
CONDITIONAL_FALL_FIELDS = ("fall_height", "landing_surface")

# Fields where a high not_stated rate reflects the originating-post-only scope
# (comment threads were never collected), not a schema failure. The QA 80% cut
# rule is NOT applied to these mechanically — judge them on captured-value accuracy.
SCOPE_LIMITED_FIELDS = {"care_pathway_outcome", "countermeasure_adopted", "presenting_signs"}

# Fields whose true value is genuinely rare: most posts do not link the event to a
# new skill or voice a counterfactual. High not_stated is the expected answer, so the
# 80% cut rule reports them separately; judge them on precision of captured values.
# Decided after reading the v1.0.0 pilot by hand (4-5 of 50 posts carry a real trigger).
RARE_BY_DESIGN_FIELDS = {"developmental_trigger", "stated_counterfactual"}

# ---------------- Field definitions (rendered into the prompt) ----------------

DEFINITIONS: Dict[str, str] = {
    "activity_when_injured": "What the child was doing immediately before the injury.",
    "developmental_trigger": "A RECENTLY acquired skill the post explicitly links to how the event happened (e.g. 'just started rolling'). Mentions of age, development or diagnoses that are not linked to the event are not_stated. Most posts have none.",
    "object_involved": "The specific product, object or substance involved, as a short noun phrase (e.g. 'changing table', 'coin battery from TV remote').",
    "object_category": "Category of object_involved.",
    "place_of_occurrence": "Type of place where the injury happened. Rooms (kitchen, bathroom) belong in micro_location, not here.",
    "micro_location": "The specific spot within the place (e.g. 'kitchen', 'top of stairs', 'bathtub').",
    "hazard_state": "The condition of the object or environment that made it hazardous (e.g. 'stair gate left open', 'hot coffee at table edge').",
    "supervision_status": "Supervision at the moment of injury, as described in THIS post only.",
    "supervisor_activity": "What the responsible caregiver was doing at the moment of injury.",
    "others_present": "Who else was present at the time of injury.",
    "child_action": "What the child did that led to the injury (e.g. 'rolled off', 'pulled the cord').",
    "mechanism_iceci": "How the injury was physically produced.",
    "mechanism_detail": "Short description of the mechanism in this event.",
    "fall_height": "Height fallen, as described (e.g. '3 feet', 'height of the bed'). ONLY for fall mechanisms; otherwise not_stated.",
    "landing_surface": "Surface the child landed on (e.g. 'hardwood floor'). ONLY for fall mechanisms; otherwise not_stated.",
    "witnessed": "Whether the poster saw the event happen.",
    "presenting_signs": "Signs and symptoms observed after the injury.",
    "first_response": "What the caregiver did immediately after the injury.",
    "help_seeking": "ALL sources of help or advice actually sought (list). 'forum_advice_only' only if posting here is the sole help sought.",
    "care_pathway_outcome": "The most intensive point of care actually reached.",
    "parent_attributed_cause": "What the poster says caused the injury, in their own framing.",
    "stated_counterfactual": "What the poster says would have prevented THIS injury or what they wish they had done differently before it (e.g. 'should never have stepped away', 'didn't think to plug the sink'). Questions, requests for advice and general wishes are not counterfactuals.",
    "countermeasure_adopted": "A concrete change the poster has made or says they will make to prevent a recurrence (e.g. 'anchored the bookshelf'). Treatments of the injury itself are not countermeasures.",
    "reason_for_posting": "The main reason the poster wrote the post.",
    # Same construct as Phase 1's is_injury_event (classify_injuries_openai.py SYSTEM_PROMPT), so the
    # audit measures Phase 1's error rather than a different definition.
    "injury_event_check": "Ignoring any earlier screening: does this post describe a REAL accidental harmful incident to a child (e.g. fall, burn, ingestion, bite, fracture, head injury, choking, poisoning) that actually happened or is clearly suspected? Near miss: a real hazardous incident happened (e.g. caught before falling, object taken out of the mouth before swallowing, pulled out of water) but the post says or implies no injury occurred and none is suspected; if harm may have occurred, it is suspected_injury_event. Not an injury event: sleep/nap, feeding, behaviour or general parenting topics; purely hypothetical or advice-only posts with no actual incident; metaphor ('fell behind'); injuries only to adults. Span: the passage describing the incident, or for not_an_injury_event the passage showing what the post is about. This judgement does not change how you fill the other fields. Put the judgement in value, not in status. Status: stated if the post explicitly describes the incident and the harm (or explicitly says no harm occurred); inferred if your judgement rests on the post as a whole. Always give a value: use 'unclear' rather than not_stated.",
}

# Glosses for values whose meaning is not self-evident. Rendered in the prompt.
VALUE_NOTES: Dict[str, Dict[str, str]] = {
    "injury_event_check": {
        "actual_injury_event": "the post describes an injury incident that happened",
        "suspected_injury_event": "the post describes an incident where injury is clearly suspected (e.g. possible ingestion)",
        "near_miss": "a real hazardous incident to a child with no injury and none suspected",
        "not_an_injury_event": "no actual hazardous incident to a child (off-topic, hypothetical, advice-only, adult-only)",
        "unclear": "cannot tell from the post",
    },
    "supervision_status": {
        "direct_supervision": "caregiver watching the child and within reach",
        "proximate_supervision": "caregiver nearby (same room/within reach) but not watching at that moment",
        "intermittent_supervision": "caregiver in another room, checking periodically",
        "no_supervision": "no caregiver attending (e.g. adults asleep or out of the room)",
        "supervised_by_other_adult": "an adult other than the poster was responsible",
        "supervised_by_minor": "an older child was responsible",
    },
    "witnessed": {
        "heard_not_seen": "poster heard it (thud, cry) but did not see it",
        "found_after_event": "poster discovered the child after it happened",
        "witnessed_by_other": "someone other than the poster saw it",
    },
    "care_pathway_outcome": {
        "advice_only_no_visit": "phone/online advice only, no in-person visit",
        "ed_presentation_discharged": "went to emergency department, sent home",
        "ed_presentation_admitted": "went to emergency department, admitted",
        "hospital_admission": "admitted to hospital not via a described ED visit",
        "ongoing_or_pending": "care is still under way or a decision is pending",
    },
}

# ---------------- Field types ----------------

def field_type(name: str) -> str:
    if name in MULTI_VALUED:
        return "multi_enum"
    if name in VOCAB:
        return "enum"
    return "text"

FIELD_PHASE = {f: (p, c) for p, c, f in FIELDS}

# ---------------- Prompt spec rendering ----------------
# Generated from the schema, never hand-maintained, so prompt and validator cannot drift.

def render_field_spec() -> str:
    lines = []
    current_phase = None
    for phase, cell, name in FIELDS:
        if phase != current_phase:
            lines.append("")
            lines.append(f"== {PHASE_TITLES[phase]} ==")
            current_phase = phase
        ftype = field_type(name)
        lines.append(f"- {name} [{cell}; {ftype}]: {DEFINITIONS[name]}")
        if ftype in ("enum", "multi_enum"):
            label = "allowed values (list of one or more)" if ftype == "multi_enum" else "allowed values"
            lines.append(f"    {label}: {', '.join(VOCAB[name])}")
            for v, note in VALUE_NOTES.get(name, {}).items():
                lines.append(f"      {v} = {note}")
        else:
            lines.append(f"    free text, <= {FREE_TEXT_MAX} chars, short paraphrase")
    return "\n".join(lines).strip()

# ---------------- Enumeration snapping ----------------
# A semantically wrong value quietly snapped to a plausible neighbour is worse
# than one that is flagged. Snapping is therefore only allowed when the
# near-miss is UNAMBIGUOUS; ties are refused (enum_ambiguous) rather than
# guessed, because guessing would invent distinctions the analysis rests on
# (e.g. 'ed_presentation' -> admitted vs discharged; bare 'fall' -> five subtypes).
# A high enum_violation rate on a field means the prompt or vocabulary needs
# fixing — that signal must survive to the QA report, so nothing here is silent.

SNAP_STOPWORDS = {"or", "and", "the", "a"}
SNAP_SIMILARITY = 0.82

def _norm_enum(value: Any) -> str:
    s = str(value).strip().lower().replace("-", " ").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")

def _tokens(s: str) -> set:
    return {t for t in s.split("_") if t and t not in SNAP_STOPWORDS}

def snap_enum(field: str, value: Any) -> Tuple[Optional[str], str, str]:
    """
    Resolve a raw model value against VOCAB[field].
    Returns (resolved_value_or_None, code, detail) where code is one of
    'exact', 'enum_snapped', 'enum_ambiguous', 'enum_violation'.
    """
    allowed = VOCAB[field]
    v = _norm_enum(value)
    if v in allowed:
        return v, "exact", ""

    # 1. Token containment
    vt = _tokens(v)
    if vt:
        cands = []
        for cand in allowed:
            ct = _tokens(cand)
            if vt <= ct or ct <= vt:
                cands.append((len(vt ^ ct), len(cand), cand))
        if cands:
            cands.sort()
            if len(cands) > 1 and cands[0][0] == cands[1][0]:
                tied = [c for d, _, c in cands if d == cands[0][0]]
                return None, "enum_ambiguous", f"{value!r} equidistant from {tied}"
            return cands[0][2], "enum_snapped", f"{value!r} -> {cands[0][2]}"

    # 2. String similarity
    scored = sorted(
        ((difflib.SequenceMatcher(None, v, cand).ratio(), cand) for cand in allowed),
        reverse=True,
    )
    best_ratio, best = scored[0]
    if best_ratio >= SNAP_SIMILARITY:
        if len(scored) > 1 and scored[1][0] == best_ratio:
            tied = [c for r, c in scored if r == best_ratio]
            return None, "enum_ambiguous", f"{value!r} equally similar to {tied}"
        return best, "enum_snapped", f"{value!r} -> {best} (similarity {best_ratio:.2f})"

    return None, "enum_violation", f"{value!r} not in vocabulary"

# ---------------- Span grounding ----------------

_QUOTE_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})

def _norm_span(s: str) -> str:
    # Whitespace + case normalised, per brief. Also folds curly quotes and decodes
    # HTML entities: models routinely straighten quotes when copying, and ~60% of
    # confirmed posts contain curly quotes, so a literal match gives false positives.
    s = html.unescape(s).translate(_QUOTE_FOLD)
    return " ".join(s.lower().split())

def span_in_text(span: str, source_text: str) -> bool:
    return _norm_span(span) in _norm_span(source_text)

# Models (gpt-4.1-mini especially) splice genuine fragments with "..." or line breaks.
# Every fragment being verbatim makes the span grounded, so it is accepted — but logged
# as elided_span so the rate stays visible. Fragments under MIN_SEGMENT_WORDS words are
# too short to prove anything and make the whole span unsupported.
_ELISION_SPLIT = re.compile(r"\.\.\.+|…|\n")
MIN_SEGMENT_WORDS = 3

def span_grounding(span: str, source_text: str) -> str:
    """Returns 'verbatim', 'elided' or 'unsupported'."""
    src = _norm_span(source_text)
    if _norm_span(span) in src:
        return "verbatim"
    segs = [_norm_span(s).strip(" .,;:") for s in _ELISION_SPLIT.split(span)]
    segs = [s for s in segs if s]
    if len(segs) > 1 and all(len(s.split()) >= MIN_SEGMENT_WORDS and s in src for s in segs):
        return "elided"
    return "unsupported"

# ---------------- Key recovery ----------------
# gpt-5-nano occasionally prefixed every key with its phase ('pre_event.activity_when_injured'),
# nested fields under phase objects, or padded keys with whitespace — losing all 24 cells
# as malformed. These are unambiguous, so map them back and log key_recovered.

def _recover_keys(rec: Dict[str, Any], log) -> Dict[str, Any]:
    out = {k: v for k, v in rec.items() if k in FIELD_NAMES}
    pending = [(k, v) for k, v in rec.items() if k not in FIELD_NAMES]
    while pending:
        k, v = pending.pop(0)
        if isinstance(v, dict) and _norm_enum(k) in PHASE_TITLES and not ({"value", "status"} & set(v)):
            pending.extend((f"{k}.{kk}", vv) for kk, vv in v.items())
            continue
        cand = str(k).split(".")[-1].strip().lower()
        if cand in FIELD_NAMES and cand not in out:
            out[cand] = v
            log(cand, "key_recovered", f"from key {k!r}")
        else:
            out[k] = v

    # A single cell keyed by a phase name ('pre_event': {...}) is recovered only by
    # elimination — when exactly one field of that phase is missing. Otherwise it stays extra.
    for k in [k for k in out if k not in FIELD_NAMES]:
        phase = _norm_enum(k)
        if phase in PHASE_TITLES and isinstance(out[k], dict) and {"value", "status"} & set(out[k]):
            missing = [f for p, _, f in FIELDS if p == phase and f not in out]
            if len(missing) == 1:
                out[missing[0]] = out.pop(k)
                log(missing[0], "key_recovered", f"from phase key {k!r} (only missing {phase} field)")
    return out

# ---------------- Normalizer ----------------

def blank_cell() -> Dict[str, Any]:
    return {"value": None, "evidence_span": None, "status": "not_stated"}

def _is_empty(x: Any) -> bool:
    return x is None or (isinstance(x, str) and not x.strip()) or (isinstance(x, (list, tuple)) and not x)

def normalize_extraction(rec: Dict[str, Any], source_text: str) -> Dict[str, Any]:
    """
    Apply guardrails to a raw model dict. source_text = title + "\\n" + selftext.
    Returns {"fields": {24 cells}, "violations": [{"field", "code", "detail"}]}.
    Violations are never silently repaired away — their rates are the QA output.
    """
    if isinstance(rec, dict) and isinstance(rec.get("fields"), dict) and not (set(rec) & set(FIELD_NAMES)):
        rec = rec["fields"]
    rec = rec if isinstance(rec, dict) else {}

    fields: Dict[str, Dict[str, Any]] = {}
    violations: List[Dict[str, Any]] = []

    def log(field, code, detail=""):
        violations.append({"field": field, "code": code, "detail": detail})

    rec = _recover_keys(rec, log)

    for extra in sorted(set(rec) - set(FIELD_NAMES)):
        log(extra, "extra_field", "key not in schema; ignored")

    for name in FIELD_NAMES:
        raw = rec.get(name)

        # 7.1 Malformed cell
        if not isinstance(raw, dict):
            fields[name] = blank_cell()
            log(name, "malformed_cell", f"expected object, got {type(raw).__name__}")
            continue

        value = raw.get("value")
        span = raw.get("evidence_span")
        span = span.strip() if isinstance(span, str) and span.strip() else None

        # 7.2 Bad status
        status = _norm_enum(raw.get("status")) if raw.get("status") is not None else ""
        if status not in STATUSES:
            log(name, "bad_status", f"{raw.get('status')!r}")
            status = "not_stated"

        # 7.3 not_stated carries no value — enforce, don't trust
        if status == "not_stated" or _is_empty(value):
            fields[name] = blank_cell()
            continue

        ftype = field_type(name)

        # 7.4 Enumeration handling
        if ftype == "enum":
            if isinstance(value, list):
                value = value[0] if len(value) == 1 else value
            resolved, code, detail = snap_enum(name, value)
            if code != "exact":
                log(name, code, detail)
            value = resolved if resolved is not None else "unspecified"
        elif ftype == "multi_enum":
            members = value if isinstance(value, list) else [value]
            kept: List[str] = []
            for m in members:
                if _is_empty(m):
                    continue
                resolved, code, detail = snap_enum(name, m)
                if code != "exact":
                    log(name, code, detail)
                if resolved is not None and resolved not in kept:
                    kept.append(resolved)
            value = kept or ["unspecified"]
        # 7.5 Free text
        else:
            value = str(value).strip()[:FREE_TEXT_MAX]
            if not value:
                fields[name] = blank_cell()
                continue

        # 7.6 Span grounding (the hallucination check)
        grounding = span_grounding(span, source_text) if span is not None else None
        if grounding == "elided":
            log(name, "elided_span", span[:120])
        if status == "stated":
            if span is None:
                status = "inferred"
                log(name, "stated_without_span", "")
            elif grounding == "unsupported":
                status = "inferred"
                log(name, "unsupported_span", span[:120])
        elif grounding == "unsupported":
            # Not in the brief's downgrade rule (inferred has nowhere lower to go),
            # but an invented quote on an inferred cell is still a hallucination.
            log(name, "inferred_span_unsupported", span[:120])

        fields[name] = {"value": value, "evidence_span": span, "status": status}

    # 7.7 Conditional coherence
    mech = fields["mechanism_iceci"]["value"]
    if mech not in FALL_MECHANISMS:
        for name in CONDITIONAL_FALL_FIELDS:
            if fields[name]["status"] != "not_stated":
                log(name, "conditional_violation",
                    f"cleared {fields[name]['value']!r}; mechanism_iceci={mech!r}")
                fields[name] = blank_cell()

    # 7.8 Cross-field coherence
    cat = fields["object_category"]["value"]
    if cat not in (None, "none_identified", "unspecified") and fields["object_involved"]["status"] == "not_stated":
        log("object_category", "category_without_object", f"object_category={cat!r}")

    return {"fields": fields, "violations": violations}
