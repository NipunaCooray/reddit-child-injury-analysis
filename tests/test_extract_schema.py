#!/usr/bin/env python3
"""
Acceptance tests for src/extract/schema.py (Phase 2 brief §12).
Run:  python -m unittest tests/test_extract_schema.py   (or: pytest tests/)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "extract"))

from schema import (FIELD_NAMES, EXTRACTION_FIELD_NAMES, AUDIT_FIELD, normalize_extraction,  # noqa: E402
                    snap_enum, render_field_spec, VOCAB)

POST_A_TITLE = "Toddler fell off the changing table"
POST_A_BODY = (
    "I turned around to grab a wipe and she rolled right off onto the hardwood floor.\n"
    "She screamed straight away and has a big egg on her forehead. We went to ED and\n"
    "they did a CT, all clear, home same night. I feel awful, I should never have\n"
    "stepped away."
)
POST_A = POST_A_TITLE + "\n" + POST_A_BODY

def cell(value, span=None, status="stated"):
    return {"value": value, "evidence_span": span, "status": status}

def codes_for(result, field):
    return [v["code"] for v in result["violations"] if v["field"] == field]

class TestGuardrailsPostA(unittest.TestCase):

    def test_1_unsupported_span_downgrades(self):
        r = normalize_extraction(
            {"supervision_status": cell("direct_supervision", "I was holding her the whole time")}, POST_A)
        self.assertIn("unsupported_span", codes_for(r, "supervision_status"))
        self.assertEqual(r["fields"]["supervision_status"]["status"], "inferred")

    def test_2_stated_without_span_downgrades(self):
        r = normalize_extraction({"place_of_occurrence": cell("home_indoor", None)}, POST_A)
        self.assertIn("stated_without_span", codes_for(r, "place_of_occurrence"))
        self.assertEqual(r["fields"]["place_of_occurrence"]["status"], "inferred")

    def test_3_not_stated_blanks_value(self):
        r = normalize_extraction(
            {"object_involved": cell("changing table", "changing table", "not_stated")}, POST_A)
        c = r["fields"]["object_involved"]
        self.assertIsNone(c["value"])
        self.assertIsNone(c["evidence_span"])
        self.assertEqual(c["status"], "not_stated")

    def test_4_multi_enum_drops_bad_member(self):
        r = normalize_extraction(
            {"help_seeking": cell(["emergency_department", "witch_doctor"], "We went to ED")}, POST_A)
        self.assertIn("enum_violation", codes_for(r, "help_seeking"))
        self.assertEqual(r["fields"]["help_seeking"]["value"], ["emergency_department"])

    def test_5_bare_string_is_malformed(self):
        r = normalize_extraction({"witnessed": "directly_witnessed"}, POST_A)
        self.assertIn("malformed_cell", codes_for(r, "witnessed"))
        self.assertEqual(r["fields"]["witnessed"],
                         {"value": None, "evidence_span": None, "status": "not_stated"})

    def test_6_conditional_violation_clears_fall_field(self):
        r = normalize_extraction({
            "mechanism_iceci": cell("blunt_force_struck_against", "she rolled right off"),
            "landing_surface": cell("hardwood floor", "onto the hardwood floor"),
        }, POST_A)
        self.assertIn("conditional_violation", codes_for(r, "landing_surface"))
        self.assertEqual(r["fields"]["landing_surface"]["status"], "not_stated")
        self.assertIsNone(r["fields"]["landing_surface"]["value"])

    def test_fall_mechanism_keeps_fall_fields(self):
        r = normalize_extraction({
            "mechanism_iceci": cell("fall_from_furniture", "rolled right off"),
            "landing_surface": cell("hardwood floor", "onto the hardwood floor"),
        }, POST_A)
        self.assertEqual(codes_for(r, "landing_surface"), [])
        self.assertEqual(r["fields"]["landing_surface"]["value"], "hardwood floor")

    def test_curly_quotes_and_whitespace_still_ground(self):
        text = "Title\nShe said “mama”   and then  FELL off the bed."
        r = normalize_extraction({"child_action": cell("fell", 'she said "mama" and then fell')}, text)
        self.assertEqual(codes_for(r, "child_action"), [])
        self.assertEqual(r["fields"]["child_action"]["status"], "stated")

    def test_all_fields_always_returned(self):
        r = normalize_extraction({}, POST_A)
        self.assertEqual(list(r["fields"]), FIELD_NAMES)
        self.assertEqual(len(EXTRACTION_FIELD_NAMES), 24)
        self.assertEqual(FIELD_NAMES, EXTRACTION_FIELD_NAMES + [AUDIT_FIELD])

class TestSnapping(unittest.TestCase):
    CASES = [
        ("mechanism_iceci", "choking", "choking_airway_obstruction", "enum_snapped"),
        ("mechanism_iceci", "thermal_scald", "thermal_scald_liquid", "enum_snapped"),
        ("mechanism_iceci", "fall", None, "enum_ambiguous"),
        ("mechanism_iceci", "witchcraft", None, "enum_violation"),
        ("object_category", "button batteries", "button_battery", "enum_snapped"),
        ("help_seeking", "emergency_dept", "emergency_department", "enum_snapped"),
        ("care_pathway_outcome", "ed_presentation", None, "enum_ambiguous"),
        ("place_of_occurrence", "home indoor", "home_indoor", "exact"),
        ("place_of_occurrence", "kitchen", None, "enum_violation"),
        ("supervision_status", "no supervision", "no_supervision", "exact"),
    ]

    def test_snap_cases(self):
        for field, raw, expected, code in self.CASES:
            with self.subTest(field=field, raw=raw):
                value, got_code, _ = snap_enum(field, raw)
                self.assertEqual(got_code, code)
                self.assertEqual(value, expected)

    def test_ambiguous_fall_cascades_to_conditional(self):
        # Refusing to guess 'fall' leaves mechanism unspecified, so fall fields are cleared —
        # both violations are logged and the cleared value is kept in the detail.
        r = normalize_extraction({
            "mechanism_iceci": cell("fall", "rolled right off"),
            "fall_height": cell("changing table height", "changing table"),
        }, POST_A)
        self.assertIn("enum_ambiguous", codes_for(r, "mechanism_iceci"))
        self.assertEqual(r["fields"]["mechanism_iceci"]["value"], "unspecified")
        self.assertIn("conditional_violation", codes_for(r, "fall_height"))

class TestCleanPass(unittest.TestCase):
    TITLE = "PSA: button battery from the TV remote - 18mo in hospital"
    BODY = (
        "I was asleep on the couch and my 18 month old got hold of the TV remote. "
        "He had been opening everything lately, cupboards, drawers, all of it. "
        "When I woke up I found him chewing on the remote in the living room and the battery cover was off "
        "and one battery was missing. His older brother was at school. "
        "He was drooling a lot and kept gagging. I checked his mouth and couldn't see anything. "
        "I called Poisons Information and they said go to the ED immediately. "
        "X-ray showed the battery in his oesophagus and he was admitted overnight after they removed it. "
        "I keep thinking if I had taped the battery cover this would never have happened. "
        "We have now taped every remote in the house. "
        "Please check your remotes and anything with a coin battery."
    )

    def full_extraction(self):
        return {
            "activity_when_injured": cell("play_unstructured", "got hold of the TV remote", "inferred"),
            "developmental_trigger": cell("newly_opening_containers", "He had been opening everything lately"),
            "object_involved": cell("button battery from TV remote", "one battery was missing"),
            "object_category": cell("button_battery", "button battery from the TV remote"),
            "place_of_occurrence": cell("home_indoor", "in the living room", "inferred"),
            "micro_location": cell("living room", "in the living room"),
            "hazard_state": cell("remote battery cover not secured", "the battery cover was off"),
            "supervision_status": cell("no_supervision", "I was asleep on the couch"),
            "supervisor_activity": cell("asleep on the couch", "I was asleep on the couch"),
            "others_present": cell("alone", "His older brother was at school", "inferred"),
            "child_action": cell("chewed remote and swallowed battery", "chewing on the remote"),
            "mechanism_iceci": cell("foreign_body_insertion_or_ingestion", "battery in his oesophagus"),
            "mechanism_detail": cell("swallowed button battery lodged in oesophagus", "battery in his oesophagus"),
            "fall_height": cell(None, None, "not_stated"),
            "landing_surface": cell(None, None, "not_stated"),
            "witnessed": cell("found_after_event", "When I woke up I found him chewing on the remote"),
            "presenting_signs": cell("drooling and gagging", "He was drooling a lot and kept gagging"),
            "first_response": cell("checked mouth", "I checked his mouth"),
            "help_seeking": cell(["poisons_information_centre", "emergency_department"],
                                 "I called Poisons Information and they said go to the ED immediately"),
            "care_pathway_outcome": cell("ed_presentation_admitted", "he was admitted overnight"),
            "parent_attributed_cause": cell("unsecured battery cover", "the battery cover was off"),
            "stated_counterfactual": cell("taping the battery cover",
                                          "if I had taped the battery cover this would never have happened"),
            "countermeasure_adopted": cell("taped every remote", "We have now taped every remote in the house"),
            "reason_for_posting": cell("warning_others", "Please check your remotes"),
            "injury_event_check": cell("actual_injury_event", "X-ray showed the battery in his oesophagus"),
        }

    def test_clean_pass_zero_violations(self):
        r = normalize_extraction(self.full_extraction(), self.TITLE + "\n" + self.BODY)
        self.assertEqual(r["violations"], [])
        f = r["fields"]
        self.assertEqual(f["supervision_status"]["value"], "no_supervision")
        self.assertEqual(f["witnessed"]["value"], "found_after_event")
        self.assertEqual(f["help_seeking"]["value"], ["poisons_information_centre", "emergency_department"])
        self.assertEqual(f["care_pathway_outcome"]["value"], "ed_presentation_admitted")

    def test_omitted_fields_only_malformed(self):
        rec = self.full_extraction()
        del rec["first_response"], rec["micro_location"]
        r = normalize_extraction(rec, self.TITLE + "\n" + self.BODY)
        self.assertEqual({v["code"] for v in r["violations"]}, {"malformed_cell"})
        self.assertEqual(sorted(v["field"] for v in r["violations"]), ["first_response", "micro_location"])

class TestV11Guardrails(unittest.TestCase):

    def test_phase_prefixed_keys_recovered(self):
        r = normalize_extraction({"pre_event.micro_location": cell("changing table", "changing table"),
                                  " supervisor_activity": cell("grabbing a wipe", "grab a wipe")}, POST_A)
        self.assertEqual(r["fields"]["micro_location"]["value"], "changing table")
        self.assertEqual(r["fields"]["supervisor_activity"]["value"], "grabbing a wipe")
        self.assertEqual(sorted(codes_for(r, "micro_location") + codes_for(r, "supervisor_activity")),
                         ["key_recovered", "key_recovered"])
        self.assertFalse([v for v in r["violations"] if v["code"] == "extra_field"])

    def test_phase_nested_keys_recovered(self):
        r = normalize_extraction({"pre_event": {"micro_location": cell("changing table", "changing table")}}, POST_A)
        self.assertEqual(r["fields"]["micro_location"]["value"], "changing table")
        self.assertEqual(codes_for(r, "micro_location"), ["key_recovered"])

    def test_phase_key_single_cell_recovered_by_elimination(self):
        rec = {f: cell(None, None, "not_stated") for f in FIELD_NAMES if f != "activity_when_injured"}
        rec["pre_event"] = cell("play_unstructured", None, "inferred")
        r = normalize_extraction(rec, POST_A)
        self.assertEqual(r["fields"]["activity_when_injured"]["value"], "play_unstructured")
        self.assertEqual([v["code"] for v in r["violations"]], ["key_recovered"])

    def test_phase_key_single_cell_ambiguous_stays_extra(self):
        r = normalize_extraction({"pre_event": cell("play_unstructured", None, "inferred")}, POST_A)
        self.assertIn("extra_field", codes_for(r, "pre_event"))
        self.assertEqual(r["fields"]["activity_when_injured"]["status"], "not_stated")

    def test_exact_key_wins_over_recovered(self):
        r = normalize_extraction({"micro_location": cell("changing table", "changing table"),
                                  "pre_event.micro_location": cell("floor", "hardwood floor")}, POST_A)
        self.assertEqual(r["fields"]["micro_location"]["value"], "changing table")
        self.assertIn("extra_field", [v["code"] for v in r["violations"]])

    def test_elided_verbatim_segments_accepted(self):
        r = normalize_extraction({"presenting_signs": cell(
            "screaming, forehead lump", "She screamed straight away ... has a big egg on her forehead")}, POST_A)
        self.assertEqual(r["fields"]["presenting_signs"]["status"], "stated")
        self.assertEqual(codes_for(r, "presenting_signs"), ["elided_span"])

    def test_elided_with_invented_segment_unsupported(self):
        r = normalize_extraction({"presenting_signs": cell(
            "screaming", "She screamed straight away ... and then she vomited twice")}, POST_A)
        self.assertEqual(r["fields"]["presenting_signs"]["status"], "inferred")
        self.assertEqual(codes_for(r, "presenting_signs"), ["unsupported_span"])

    def test_elided_short_segment_unsupported(self):
        r = normalize_extraction({"presenting_signs": cell("screaming", "She screamed straight away ... egg")}, POST_A)
        self.assertEqual(codes_for(r, "presenting_signs"), ["unsupported_span"])

    def test_none_stated_removed_from_developmental_trigger(self):
        self.assertNotIn("none_stated", VOCAB["developmental_trigger"])

class TestAudit(unittest.TestCase):

    def test_audit_near_miss_snaps(self):
        self.assertEqual(snap_enum(AUDIT_FIELD, "no_injury_event")[:2], ("not_an_injury_event", "enum_snapped"))
        self.assertEqual(snap_enum(AUDIT_FIELD, "injury_event")[1], "enum_ambiguous")

    def test_audit_cell_grounded_like_any_field(self):
        r = normalize_extraction({AUDIT_FIELD: cell("not_an_injury_event", "she slept through the night")}, POST_A)
        self.assertEqual(r["fields"][AUDIT_FIELD]["status"], "inferred")
        self.assertIn("unsupported_span", codes_for(r, AUDIT_FIELD))

class TestSamplerBoost(unittest.TestCase):

    def test_boost_exempt_from_trimming(self):
        from sample_pilot import allocate
        sizes = {"fall": 405, "choking": 252, "rare": 6, "rarer": 3}
        alloc = allocate(sizes, 14, 4, {"fall": 6})
        self.assertEqual(alloc, {"fall": 6, "choking": 2, "rare": 3, "rarer": 3})

    def test_pilot50_trim_matches_v10(self):
        from sample_pilot import allocate
        sizes = {"fall": 405, "choking_or_suffocation": 252, **{f"s{i}": 10 for i in range(11)}}
        alloc = allocate(sizes, 50, 4)
        self.assertEqual((alloc["fall"], alloc["choking_or_suffocation"], sum(alloc.values())), (3, 3, 50))

class TestPromptSpec(unittest.TestCase):

    def test_spec_lists_every_field_and_value(self):
        spec = render_field_spec()
        for name in FIELD_NAMES:
            self.assertIn(f"- {name} [", spec)
        for values in VOCAB.values():
            for v in values:
                self.assertIn(v, spec)

if __name__ == "__main__":
    unittest.main()
