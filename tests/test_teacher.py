#!/usr/bin/env python3
"""Guards + type-selection for the teacher pipeline — the parts that keep fabricated
numbers out of the SFT set. Pure CPU, no GPU/vLLM. Run: python tests/test_teacher.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from finetune import teacher as tg

SRC = ("TITLE: Confocal imaging of dendrites\n\n"
       "METHODS: Samples were fixed in 4% paraformaldehyde and imaged at 63x with a "
       "numerical aperture of 1.4. Z-stacks spanned 20 um at 0.3 um steps.\n\n"
       "RESULTS: Fluorescence intensity decreased with imaging depth owing to spherical "
       "aberration and signal attenuation. Resolution was 240 nm laterally.")


def check(name, cond):
    print(("ok  " if cond else "FAIL ") + name)
    assert cond, name


# ── span grounding ───────────────────────────────────────────────────────────
check("verbatim span grounds",
      tg.span_grounded("Fluorescence intensity decreased with imaging depth", SRC))
check("span normalises whitespace",
      tg.span_grounded("Fluorescence intensity   decreased\n with imaging depth", SRC))
check("fabricated span rejected",
      not tg.span_grounded("images were acquired on a two-photon microscope", SRC))
check("trivially short span rejected",
      not tg.span_grounded("20 um", SRC))
check("empty span rejected", not tg.span_grounded("", SRC))

# regression: real drops from the 122B quality probe were cosmetic-Unicode false
# positives — a genuinely verbatim quote the model re-punctuated (curly quotes,
# en-dash), or two real quotes the model stitched with '...'. These MUST ground;
# a fabricated stitch MUST NOT. \u escapes keep the invisible source chars explicit.
CURLY = ("METHODS: deactivating “scale bar” settings are necessary because "
         "graphics interfere with image structure detection. The section spanned "
         "0.3–20 µm in depth.")  # curly quotes + en-dash + thin space + micro
check("curly-quote span grounds against straight-quote text",
      tg.span_grounded('deactivating "scale bar" settings are necessary', CURLY))
check("en-dash span grounds against straight-hyphen span",
      tg.span_grounded("section spanned 0.3-20", CURLY))
check("thin-space span grounds against normalised text",
      tg.span_grounded("spanned 0.3-20 µm in depth", CURLY))
check("ellipsis-stitched real quotes ground (each fragment verbatim)",
      tg.span_grounded("Fluorescence intensity decreased with imaging depth ... "
                       "Resolution was 240 nm laterally", SRC))
check("ellipsis stitch with a fabricated fragment still rejected",
      not tg.span_grounded("Fluorescence intensity decreased with imaging depth ... "
                           "resolution was 90 nm laterally", SRC))

# ── numeric grounding ────────────────────────────────────────────────────────
ok, bad = tg.numbers_grounded("The NA was 1.4 and steps were 0.3 um.", SRC)
check("real numbers pass", ok and bad is None)
ok, bad = tg.numbers_grounded("Lateral resolution reached 90 nm.", SRC)
check("fabricated number caught", (not ok) and bad == "90")
ok, _ = tg.numbers_grounded("Imaged at 63x over 20 um.", SRC)
check("integer value match passes", ok)
ok, _ = tg.numbers_grounded("No numbers here.", SRC)
check("answer with no numbers passes", ok)

# ── full pair check ──────────────────────────────────────────────────────────
good = {"type": "artifact_diagnosis",
        "question": "Why do confocal z-stacks dim with depth?",
        "answer": "Because of spherical aberration and signal attenuation.",
        "supporting_span": "spherical aberration and signal attenuation"}
kept, reason = tg.check_pair(good, SRC)
check("grounded pair kept", kept and reason == "")

bad_num = dict(good, answer="Resolution collapsed to 900 nm with depth.",
               supporting_span="Fluorescence intensity decreased with imaging depth")
kept, reason = tg.check_pair(bad_num, SRC)
check("pair with fabricated number dropped", (not kept) and reason.startswith("num:"))

bad_span = dict(good, supporting_span="the microscope used a tunable laser source")
kept, reason = tg.check_pair(bad_span, SRC)
check("pair with ungrounded span dropped", (not kept) and reason == "span")

empty_q = dict(good, question="")
kept, reason = tg.check_pair(empty_q, SRC)
check("pair with empty question dropped", (not kept) and reason == "empty")

# ── type selection ───────────────────────────────────────────────────────────
t = tg.select_types({"A", "E"})
check("A×E paper gets modality + interpretation, no prep",
      "modality_selection" in t and "result_interpretation" in t and "sample_prep" not in t)
check("<=6 types selected", len(tg.select_types(set("ABCDEJ"))) == 6)
check("artifact_diagnosis ranks first when B+C present",
      tg.select_types({"B", "C"})[0] == "artifact_diagnosis")
check("empty facets -> no types", tg.select_types(set()) == [])

# ── prompts build without error and embed the rules ──────────────────────────
gp = tg.gen_prompt(SRC, ["artifact_diagnosis", "modality_selection"])
check("gen prompt lists requested types", "artifact_diagnosis" in gp and "PAPER TEXT" in gp)
jp = tg.judge_prompt(SRC, [good])
check("judge prompt embeds Q and A", good["question"] in jp and "SOURCE:" in jp)

print("\nall teacher guard tests passed")
