#!/usr/bin/env python3
"""Guards + type-selection for the teacher pipeline — the parts that keep fabricated
numbers out of the SFT set. Pure CPU, no GPU/vLLM. Run: pytest tests/test_teacher.py"""
from finetune import teacher as tg

SRC = ("TITLE: Confocal imaging of dendrites\n\n"
       "METHODS: Samples were fixed in 4% paraformaldehyde and imaged at 63x with a "
       "numerical aperture of 1.4. Z-stacks spanned 20 um at 0.3 um steps.\n\n"
       "RESULTS: Fluorescence intensity decreased with imaging depth owing to spherical "
       "aberration and signal attenuation. Resolution was 240 nm laterally.")

# regression: real drops from the 122B quality probe were cosmetic-Unicode false
# positives — a genuinely verbatim quote the model re-punctuated (curly quotes,
# en-dash), or two real quotes the model stitched with '...'. These MUST ground;
# a fabricated stitch MUST NOT. \u escapes keep the invisible source chars explicit.
CURLY = ("METHODS: deactivating “scale bar” settings are necessary because "
         "graphics interfere with image structure detection. The section spanned "
         "0.3–20 µm in depth.")  # curly quotes + en-dash + thin space + micro

good = {"type": "artifact_diagnosis",
        "question": "Why do confocal z-stacks dim with depth?",
        "answer": "Because of spherical aberration and signal attenuation.",
        "supporting_span": "spherical aberration and signal attenuation"}


# ── span grounding ───────────────────────────────────────────────────────────
def test_verbatim_span_grounds():
    assert tg.span_grounded("Fluorescence intensity decreased with imaging depth", SRC), \
        "verbatim span grounds"


def test_span_normalises_whitespace():
    assert tg.span_grounded("Fluorescence intensity   decreased\n with imaging depth", SRC), \
        "span normalises whitespace"


def test_fabricated_span_rejected():
    assert not tg.span_grounded("images were acquired on a two-photon microscope", SRC), \
        "fabricated span rejected"


def test_trivially_short_span_rejected():
    assert not tg.span_grounded("20 um", SRC), \
        "trivially short span rejected"


def test_empty_span_rejected():
    assert not tg.span_grounded("", SRC), "empty span rejected"


def test_curly_quote_span_grounds():
    assert tg.span_grounded('deactivating "scale bar" settings are necessary', CURLY), \
        "curly-quote span grounds against straight-quote text"


def test_en_dash_span_grounds():
    assert tg.span_grounded("section spanned 0.3-20", CURLY), \
        "en-dash span grounds against straight-hyphen span"


def test_thin_space_span_grounds():
    assert tg.span_grounded("spanned 0.3-20 µm in depth", CURLY), \
        "thin-space span grounds against normalised text"


def test_ellipsis_stitched_real_quotes_ground():
    assert tg.span_grounded("Fluorescence intensity decreased with imaging depth ... "
                            "Resolution was 240 nm laterally", SRC), \
        "ellipsis-stitched real quotes ground (each fragment verbatim)"


def test_ellipsis_stitch_with_fabricated_fragment_rejected():
    assert not tg.span_grounded("Fluorescence intensity decreased with imaging depth ... "
                                "resolution was 90 nm laterally", SRC), \
        "ellipsis stitch with a fabricated fragment still rejected"


# ── numeric grounding ────────────────────────────────────────────────────────
def test_real_numbers_pass():
    ok, bad = tg.numbers_grounded("The NA was 1.4 and steps were 0.3 um.", SRC)
    assert ok and bad is None, "real numbers pass"


def test_fabricated_number_caught():
    ok, bad = tg.numbers_grounded("Lateral resolution reached 90 nm.", SRC)
    assert (not ok) and bad == "90", "fabricated number caught"


def test_integer_value_match_passes():
    ok, _ = tg.numbers_grounded("Imaged at 63x over 20 um.", SRC)
    assert ok, "integer value match passes"


def test_answer_with_no_numbers_passes():
    ok, _ = tg.numbers_grounded("No numbers here.", SRC)
    assert ok, "answer with no numbers passes"


# ── full pair check ──────────────────────────────────────────────────────────
def test_grounded_pair_kept():
    kept, reason = tg.check_pair(good, SRC)
    assert kept and reason == "", "grounded pair kept"


def test_pair_with_fabricated_number_dropped():
    bad_num = dict(good, answer="Resolution collapsed to 900 nm with depth.",
                   supporting_span="Fluorescence intensity decreased with imaging depth")
    kept, reason = tg.check_pair(bad_num, SRC)
    assert (not kept) and reason.startswith("num:"), "pair with fabricated number dropped"


def test_pair_with_ungrounded_span_dropped():
    bad_span = dict(good, supporting_span="the microscope used a tunable laser source")
    kept, reason = tg.check_pair(bad_span, SRC)
    assert (not kept) and reason == "span", "pair with ungrounded span dropped"


def test_pair_with_empty_question_dropped():
    empty_q = dict(good, question="")
    kept, reason = tg.check_pair(empty_q, SRC)
    assert (not kept) and reason == "empty", "pair with empty question dropped"


# ── type selection ───────────────────────────────────────────────────────────
def test_axe_paper_gets_modality_and_interpretation():
    t = tg.select_types({"A", "E"})
    assert "modality_selection" in t and "result_interpretation" in t and "sample_prep" not in t, \
        "A×E paper gets modality + interpretation, no prep"


def test_at_most_six_types_selected():
    assert len(tg.select_types(set("ABCDEJ"))) == 6, "<=6 types selected"


def test_artifact_diagnosis_ranks_first_with_bc():
    assert tg.select_types({"B", "C"})[0] == "artifact_diagnosis", \
        "artifact_diagnosis ranks first when B+C present"


def test_empty_facets_no_types():
    assert tg.select_types(set()) == [], "empty facets -> no types"


# ── prompts ──────────────────────────────────────────────────────────────────
def test_gen_prompt_lists_requested_types():
    gp = tg.gen_prompt(SRC, ["artifact_diagnosis", "modality_selection"])
    assert "artifact_diagnosis" in gp and "PAPER TEXT" in gp, "gen prompt lists requested types"


def test_judge_prompt_embeds_q_and_a():
    jp = tg.judge_prompt(SRC, [good])
    assert good["question"] in jp and "SOURCE:" in jp, "judge prompt embeds Q and A"
