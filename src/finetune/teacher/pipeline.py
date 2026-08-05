"""Teacher-generation: synthetic *grounded* QA for the microscopy SFT stage.

select -> re-parse sections -> generate 6 grounded pairs each -> DROP anything not
literally in the source -> judge -> SFT JSONL. The guards are the whole point: a
plain "generate QA with an LLM" pass quietly poisons the student with fabricated
numbers.

Every stage is resumable and writes to its OWN state DB. The source corpus DB is
opened READ-ONLY only — we never take a second writer on it. All paths, the model
endpoint, and every parameter come from configs/teacher.yaml (see load_config).

  select    : stratify is_micro=1 papers on facet A x E (paper_topics_bert)  -> sft_paper
  sections  : re-parse JATS <sec> methods+results into a ~4k ctx             -> paper_ctx
  generate  : model reads ctx -> 6 grounded QA pairs/paper (JSON)            -> gen
  guard     : span-substring + numeric checks (CPU, free) drop failures      -> pair
  judge     : 2nd model pass, per-paper, adversarial "is it supported?"       -> pair.judged
  export    : kept (& judged) pairs -> the SFT train_file (chat JSONL)
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .jats import _ABS, _BODY, _TITLE, _clean
from .progress import Bar
from .qwen_client import QwenClient, QwenError, extract_json

# ── validated guard/selection constants (do not tune without re-validating) ───
MIN_SPAN_CHARS = 12          # a "span" shorter than this grounds nothing meaningful
PAIRS_PER_PAPER = 6          # default cap for select_types; the pipeline overrides
                             # the request count from config (generate.pairs_per_paper)

# ── QA types: name -> (instruction, enabling facets) ──────────────────────────
# A paper gets the (up to 6) types whose enabling facets it actually carries.
TYPES = {
    "modality_selection":    ("Which imaging technique/modality suits a stated goal, and why.", {"A", "E"}),
    "sample_prep":           ("A sample-preparation / labelling / protocol choice and its rationale.", {"B"}),
    "instrumentation":       ("An instrument/optics setting, or a resolution / acquisition limit.", {"C"}),
    "artifact_diagnosis":    ("Given an imaging observation described in the text, identify its likely cause.", {"B", "C"}),
    "image_analysis":        ("An image-processing / quantification / analysis method or parameter.", {"D"}),
    "result_interpretation": ("Interpret a reported result or measurement in its context.", {"E", "J"}),
    "comparative_tradeoff":  ("A trade-off the PAPER ITSELF draws between two techniques/conditions.", {"A"}),
    "limitations":           ("A stated limitation, failure mode, or caveat of the approach.", {"J"}),
}
# highest teaching value first — artifact diagnosis is what people bring to a specialist
PRIORITY = ["artifact_diagnosis", "modality_selection", "result_interpretation",
            "instrumentation", "image_analysis", "sample_prep", "limitations",
            "comparative_tradeoff"]

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_WS = re.compile(r"\s+")
# fold cosmetic Unicode so a genuinely verbatim quote is not dropped on curly-vs-
# straight quotes / apostrophes or en–em–minus dashes that the transcription and the
# model disagree on. NFKC alone does NOT unify these.
_QUOTES = {ord(c): "'" for c in "‘’‛′´`"}
_QUOTES.update({ord(c): '"' for c in "“”‟″"})
_DASH = {ord(c): "-" for c in "‐‑‒–—―−"}
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")
# section anchors: sec-type attr OR a <title>…keyword…</title>
_SEC_METHODS = re.compile(
    r'(?:sec-type="[^"]*(?:method|material)[^"]*"'
    r'|<title[^>]*>\s*(?:materials?\s+and\s+methods?|methods?(?:\s+and\s+materials?)?|experimental(?:\s+section)?)\s*</title>)',
    re.I | re.S)
_SEC_RESULTS = re.compile(
    r'(?:sec-type="[^"]*result[^"]*"'
    r'|<title[^>]*>\s*results?(?:\s+and\s+discussion)?\s*</title>)',
    re.I | re.S)


# ── config ────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    store: str
    source_db: str
    state_db: str
    export_file: str
    model_base_url: str
    model_name: str
    model_concurrency: int
    select_n: int
    select_seed: int
    sections_workers: int
    abstract_chars: int
    methods_chars: int
    results_chars: int
    pairs_per_paper: int
    gen_temperature: float
    gen_max_tokens: int
    gen_chunk: int
    judge_temperature: float
    judge_max_tokens: int
    judge_chunk: int
    export_require_judge: bool


# CLI-overridable fields: key in the `overrides` mapping -> Config field it replaces.
_OVERRIDES = {
    "store": "store",
    "source_db": "source_db",
    "state_db": "state_db",
    "export_file": "export_file",
    "base_url": "model_base_url",
    "model": "model_name",
    "concurrency": "model_concurrency",
}


def load_config(config_path: str | Path, overrides: dict | None = None) -> Config:
    """Load configs/teacher.yaml into a Config. Any key in `overrides` (e.g. parsed
    from CLI flags) whose value is not None replaces the corresponding config value."""
    import yaml
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
    p, m = raw["paths"], raw["model"]
    sel, sec = raw["select"], raw["sections"]
    gen, jdg, exp = raw["generate"], raw["judge"], raw["export"]
    cfg = Config(
        store=p["store"], source_db=p["source_db"], state_db=p["state_db"],
        export_file=p["export_file"],
        model_base_url=m["base_url"], model_name=m["name"], model_concurrency=m["concurrency"],
        select_n=sel["n"], select_seed=sel["seed"],
        sections_workers=sec["workers"], abstract_chars=sec["abstract_chars"],
        methods_chars=sec["methods_chars"], results_chars=sec["results_chars"],
        pairs_per_paper=gen["pairs_per_paper"], gen_temperature=gen["temperature"],
        gen_max_tokens=gen["max_tokens"], gen_chunk=gen["chunk"],
        judge_temperature=jdg["temperature"], judge_max_tokens=jdg["max_tokens"],
        judge_chunk=jdg["chunk"],
        export_require_judge=exp["require_judge"],
    )
    for key, field in _OVERRIDES.items():
        if overrides and overrides.get(key) is not None:
            setattr(cfg, field, overrides[key])
    return cfg


# ── selection: which QA types a paper's facet set supports ────────────────────
def select_types(facets: set, cap: int = PAIRS_PER_PAPER) -> list:
    """Types whose enabling facets intersect the paper's facet set, ranked by
    PRIORITY, capped at `cap`. Optical papers always carry A, so
    modality_selection is effectively always available."""
    avail = [t for t in PRIORITY if TYPES[t][1] & facets]
    return avail[:cap]


# ── guards (pure, CPU, unit-tested) ───────────────────────────────────────────
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").translate(_QUOTES).translate(_DASH)
    return _WS.sub(" ", s).strip()


def span_grounded(span: str, source: str) -> bool:
    """The supporting span must ground verbatim in the source (whitespace / unicode /
    quote / dash-normalised; case preserved — units and names matter). A model often
    stitches two real, non-adjacent quotes with '...'; split on the ellipsis and
    require EACH fragment (>= MIN_SPAN_CHARS) to be a literal substring. Fabrication
    still cannot pass — every fragment must appear verbatim."""
    src = _norm(source)
    frags = [f for f in (_norm(p) for p in _ELLIPSIS.split(span or "")) if len(f) >= MIN_SPAN_CHARS]
    return bool(frags) and all(f in src for f in frags)


def numbers_grounded(answer: str, source: str):
    """Every number in the answer must appear in the source, by literal string OR
    by float value. Returns (ok, offending_number_or_None) — fabricated resolutions
    / concentrations / exposure times are the most damaging failure and caught here."""
    src = _norm(source)
    src_vals = {float(m) for m in _NUM.findall(src)}
    for m in _NUM.findall(_norm(answer)):
        if m in src:
            continue
        try:
            if float(m) in src_vals:
                continue
        except ValueError:
            pass
        return False, m
    return True, None


def check_pair(pair: dict, source: str):
    """(kept, reason). reason is '' when kept."""
    ans = (pair.get("answer") or "").strip()
    span = (pair.get("span") or pair.get("supporting_span") or "").strip()
    if not ans or not (pair.get("question") or "").strip():
        return False, "empty"
    if not span_grounded(span, source):
        return False, "span"
    ok, bad = numbers_grounded(ans, source)
    if not ok:
        return False, f"num:{bad}"
    return True, ""


# ── prompts ───────────────────────────────────────────────────────────────────
SYS_GEN = (
    "You are an expert microscopy educator writing question-answer pairs to TEACH a "
    "smaller model, grounded STRICTLY in ONE paper's text.\n"
    "HARD RULES:\n"
    "- Use ONLY the provided text. No outside knowledge; assert no fact not present in it.\n"
    "- Do NOT reference figures, tables, panels, or citations — the reader will not have them.\n"
    "- No cross-paper comparison; any comparison must be one the text itself makes.\n"
    "- Each answer MUST be fully supported by a VERBATIM span copied character-for-character "
    "from the text (put it in supporting_span).\n"
    "- Every NUMBER in an answer MUST appear verbatim in the text. If you are unsure, do not "
    "state the number.\n"
    "- Questions must be self-contained (a reader without the text can understand what is asked); "
    "answers are 1-4 sentences.\n"
    'Return ONLY JSON: {"pairs":[{"type":"..","question":"..","answer":"..","supporting_span":".."}]} '
    "— exactly one object per requested type, in the order requested."
)

SYS_JUDGE = (
    "You are a strict fact-checker. For each numbered QA item you are given the SOURCE text and "
    "a question/answer. Decide whether the answer is FULLY supported by the source alone — every "
    "claim and every number must be present in the source. Be adversarial: if anything is not "
    "clearly in the source, mark it unsupported.\n"
    'Return ONLY JSON: {"verdicts":[{"n":1,"supported":true|false}]} with one verdict per item, in order.'
)


def gen_prompt(ctx: str, types: list) -> str:
    asks = "\n".join(f"  {i+1}. {t} — {TYPES[t][0]}" for i, t in enumerate(types))
    return (f"Write one QA pair for EACH of these {len(types)} types, in this order:\n{asks}\n\n"
            f"=== PAPER TEXT ===\n{ctx}\n=== END ===")


def judge_prompt(ctx: str, pairs: list) -> str:
    items = "\n".join(f"[{i+1}] Q: {p['question']}\n    A: {p['answer']}" for i, p in enumerate(pairs))
    return f"SOURCE:\n{ctx}\n\nITEMS:\n{items}"


# ── db ────────────────────────────────────────────────────────────────────────
def connect(cfg: Config) -> sqlite3.Connection:
    db = Path(cfg.state_db)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db, timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS sft_paper(pmcid TEXT PRIMARY KEY, a TEXT, e TEXT, facets TEXT);"
        "CREATE TABLE IF NOT EXISTS paper_ctx(pmcid TEXT PRIMARY KEY, context TEXT);"
        "CREATE TABLE IF NOT EXISTS gen(pmcid TEXT PRIMARY KEY, pairs_json TEXT, ok INT, ts TEXT);"
        "CREATE TABLE IF NOT EXISTS pair("
        "  id INTEGER PRIMARY KEY, pmcid TEXT, type TEXT, question TEXT, answer TEXT,"
        "  span TEXT, kept INT, reason TEXT, judged INT DEFAULT -1);"
        "CREATE INDEX IF NOT EXISTS ix_pair_pmcid ON pair(pmcid);"
        "CREATE INDEX IF NOT EXISTS ix_pair_state ON pair(kept, judged);")
    conn.commit()
    return conn


def _pmc_ro(cfg: Config) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{cfg.source_db}?mode=ro", uri=True, timeout=60)


# ── stage 1: select ───────────────────────────────────────────────────────────
def stage_select(cfg: Config):
    import random
    rng = random.Random(cfg.select_seed)
    pmc_db = Path(cfg.source_db)
    print(f"[select] loading is_micro=1 pmcids + A/E/facets from {pmc_db.name} …", flush=True)
    ro = _pmc_ro(cfg)
    micro = {r[0] for r in ro.execute("SELECT pmcid FROM paper_relevance WHERE is_micro=1")}
    print(f"[select] {len(micro):,} candidate papers; scanning paper_topics_bert …", flush=True)
    # one scan: per pmcid collect facet-presence set + argmax subtopic for A and E
    best = {}   # pmcid -> {"A":(prob,sub), "E":(prob,sub), "F":set(facets)}
    cur = ro.execute("SELECT pmcid, facet, subtopic, prob FROM paper_topics_bert")
    seen = 0
    for pmcid, facet, sub, prob in cur:
        if pmcid not in micro:
            continue
        d = best.setdefault(pmcid, {"A": (0.0, None), "E": (0.0, None), "F": set()})
        d["F"].add(facet)
        if facet in ("A", "E") and prob is not None and prob >= d[facet][0]:
            d[facet] = (prob, sub)
        seen += 1
        if seen % 2_000_000 == 0:
            print(f"[select]   scanned {seen:,} topic rows…", flush=True)
    ro.close()

    # build strata keyed on (primary A, primary E)
    strata: dict = {}
    for pmcid, d in best.items():
        key = (d["A"][1] or "none", d["E"][1] or "none")
        strata.setdefault(key, []).append((pmcid, "".join(sorted(d["F"]))))
    total = sum(len(v) for v in strata.values())
    n = min(cfg.select_n, total)
    print(f"[select] {total:,} papers across {len(strata):,} A×E strata; target {n:,}")

    # largest-remainder allocation with a floor, capped by availability
    FLOOR = 10
    alloc = {k: min(len(v), FLOOR) for k, v in strata.items()}
    remaining = n - sum(alloc.values())
    if remaining > 0:
        room = {k: len(v) - alloc[k] for k, v in strata.items()}
        room_total = sum(room.values()) or 1
        # proportional to leftover room, then hand out remainders greedily
        raw = {k: remaining * room[k] / room_total for k in strata}
        base = {k: min(room[k], int(raw[k])) for k in strata}
        for k in strata:
            alloc[k] += base[k]
        left = n - sum(alloc.values())
        for k in sorted(strata, key=lambda k: raw[k] - int(raw[k]), reverse=True):
            if left <= 0:
                break
            if alloc[k] < len(strata[k]):
                alloc[k] += 1
                left -= 1

    conn = connect(cfg)
    conn.execute("DELETE FROM sft_paper")
    picked = 0
    for (a, e), members in strata.items():
        k = alloc[(a, e)]
        if k <= 0:
            continue
        chosen = members if k >= len(members) else rng.sample(members, k)
        conn.executemany("INSERT OR REPLACE INTO sft_paper VALUES (?,?,?,?)",
                          [(pmcid, a, e, facets) for pmcid, facets in chosen])
        picked += len(chosen)
    conn.commit()
    conn.close()
    print(f"[select] wrote {picked:,} papers to sft_paper")


# ── stage 2: sections (re-parse JATS -> ~4k ctx) ──────────────────────────────
_W: dict = {}   # per-worker config, set by _init_worker in the process pool


def _init_worker(store, abstract_chars, methods_chars, results_chars):
    _W["store"] = Path(store)
    _W["abstract_chars"] = abstract_chars
    _W["methods_chars"] = methods_chars
    _W["results_chars"] = results_chars


def _section_excerpt(body: str, anchor: re.Pattern, max_chars: int) -> str:
    m = anchor.search(body)
    if not m:
        return ""
    # take a window of raw markup from the anchor, then strip tags
    window = body[m.start(): m.start() + max_chars * 3]
    return _clean(window)[:max_chars]


def build_context(xml: str, abstract_chars: int, methods_chars: int, results_chars: int) -> str:
    tm = _TITLE.search(xml)
    title = _clean(tm.group(1)) if tm else ""
    abstract = _clean(" ".join(_ABS.findall(xml)))[:abstract_chars]
    bm = _BODY.search(xml)
    body = bm.group(1) if bm else xml
    methods = _section_excerpt(body, _SEC_METHODS, methods_chars)
    results = _section_excerpt(body, _SEC_RESULTS, results_chars)
    parts = [f"TITLE: {title}", f"ABSTRACT: {abstract}"]
    if methods:
        parts.append(f"METHODS: {methods}")
    if results:
        parts.append(f"RESULTS: {results}")
    return "\n\n".join(parts)


def _ctx_row(row):
    pmcid, rel = row
    try:
        xml = (_W["store"] / rel).read_text(errors="ignore")
    except Exception:
        return pmcid, None
    ctx = build_context(xml, _W["abstract_chars"], _W["methods_chars"], _W["results_chars"])
    # require BOTH methods and results signal — a paper with neither can't ground
    # instrumentation/artifact/analysis QA. Cheap guard against thin front-matter.
    if "METHODS:" not in ctx and "RESULTS:" not in ctx:
        return pmcid, None
    return pmcid, ctx


def stage_sections(cfg: Config):
    conn = connect(cfg)
    sft_dir = Path(cfg.state_db).parent
    have = {r[0] for r in conn.execute("SELECT pmcid FROM paper_ctx")}
    want = [r[0] for r in conn.execute("SELECT pmcid FROM sft_paper") if r[0] not in have]
    ro = _pmc_ro(cfg)
    paths = dict(ro.execute(
        "SELECT pmcid, local_path FROM candidates WHERE fetch_state='fetched'"))
    ro.close()
    rows = [(p, paths[p]) for p in want if p in paths]
    print(f"[sections] {len(rows):,} papers to parse ({len(have):,} cached, "
          f"{len(want) - len(rows):,} missing local_path)", flush=True)
    bar = Bar("sections", len(rows) or 1, sft_dir / "progress_sections.json")
    n = miss = 0
    with ProcessPoolExecutor(
            max_workers=cfg.sections_workers, initializer=_init_worker,
            initargs=(cfg.store, cfg.abstract_chars, cfg.methods_chars, cfg.results_chars)) as ex:
        for pmcid, ctx in ex.map(_ctx_row, rows, chunksize=128):
            n += 1
            if ctx is None:
                miss += 1
            else:
                conn.execute("INSERT OR REPLACE INTO paper_ctx VALUES (?,?)", (pmcid, ctx))
            if n % 1000 == 0:
                conn.commit()
                bar.update(n, extra=f"empty={miss}")
    conn.commit()
    bar.close(n)
    print(f"[sections] parsed {n - miss:,} contexts ({miss:,} unusable)")


# ── stage 3: generate ─────────────────────────────────────────────────────────
def stage_generate(cfg: Config):
    conn = connect(cfg)
    sft_dir = Path(cfg.state_db).parent
    done = {r[0] for r in conn.execute("SELECT pmcid FROM gen WHERE ok=1")}
    facets = dict(conn.execute("SELECT pmcid, facets FROM sft_paper"))
    todo = [(p, c) for p, c in conn.execute("SELECT pmcid, context FROM paper_ctx")
            if p not in done]
    qwen = QwenClient(base_url=cfg.model_base_url, model=cfg.model_name,
                      max_concurrency=cfg.model_concurrency,
                      enable_thinking=False, temperature=cfg.gen_temperature)
    if not qwen.health():
        sys.exit(f"[generate] vLLM not serving {cfg.model_name} at {cfg.model_base_url} — start it first.")
    print(f"[generate] {len(todo):,} papers pending ({len(done):,} done); model={cfg.model_name}", flush=True)
    bar = Bar("generate", len(todo) or 1, sft_dir / "progress_generate.json")
    st = {"n": 0, "fail": 0}

    def flush(items):
        prompts, metas = [], []
        for pmcid, ctx in items:
            types = select_types(set(facets.get(pmcid, "")), cfg.pairs_per_paper)
            if not types:
                continue
            prompts.append(gen_prompt(ctx, types))
            metas.append((pmcid, types))

        def on_result(idx, out):
            pmcid, types = metas[idx]
            arr = None if isinstance(out, QwenError) else extract_json(out)
            if isinstance(arr, dict):
                arr = arr.get("pairs")
            ok = 1 if isinstance(arr, list) and arr else 0
            if not ok:
                st["fail"] += 1
            conn.execute("INSERT OR REPLACE INTO gen VALUES (?,?,?,?)",
                         (pmcid, json.dumps(arr, ensure_ascii=False) if ok else None,
                          ok, time.strftime("%Y-%m-%dT%H:%M:%S")))
            st["n"] += 1
            bar.update(st["n"], extra=f"fail={st['fail']:,}")

        qwen.generate_batch(prompts, system=SYS_GEN,
                            max_tokens=cfg.gen_max_tokens,
                            on_result=on_result, response_format={"type": "json_object"})
        conn.commit()

    buf = []
    for pmcid, ctx in todo:
        buf.append((pmcid, ctx))
        if len(buf) >= cfg.gen_chunk:
            flush(buf)
            buf = []
    if buf:
        flush(buf)
    bar.close(st["n"])
    print(f"[generate] {st['n']:,} papers, {st['fail']:,} unparseable")


# ── stage 4: guard (CPU) ──────────────────────────────────────────────────────
def stage_guard(cfg: Config):
    conn = connect(cfg)
    sft_dir = Path(cfg.state_db).parent
    ctx = dict(conn.execute("SELECT pmcid, context FROM paper_ctx"))
    already = {r[0] for r in conn.execute("SELECT DISTINCT pmcid FROM pair")}
    rows = [(p, j) for p, j in conn.execute("SELECT pmcid, pairs_json FROM gen WHERE ok=1")
            if p not in already]
    print(f"[guard] checking {len(rows):,} papers ({len(already):,} already guarded)", flush=True)
    bar = Bar("guard", len(rows) or 1, sft_dir / "progress_guard.json")
    kept = dropped = n = 0
    reasons: dict = {}
    for pmcid, pj in rows:
        source = ctx.get(pmcid, "")
        try:
            pairs = json.loads(pj) or []
        except Exception:
            pairs = []
        for p in pairs:
            if not isinstance(p, dict):
                continue
            ok, reason = check_pair(p, source)
            conn.execute(
                "INSERT INTO pair(pmcid,type,question,answer,span,kept,reason) VALUES (?,?,?,?,?,?,?)",
                (pmcid, p.get("type"), (p.get("question") or "").strip(),
                 (p.get("answer") or "").strip(),
                 (p.get("span") or p.get("supporting_span") or "").strip(),
                 1 if ok else 0, reason))
            if ok:
                kept += 1
            else:
                dropped += 1
                reasons[reason.split(":")[0]] = reasons.get(reason.split(":")[0], 0) + 1
        n += 1
        if n % 1000 == 0:
            conn.commit()
            bar.update(n)
    conn.commit()
    bar.close(n)
    tot = kept + dropped or 1
    print(f"[guard] kept {kept:,} / dropped {dropped:,} pairs "
          f"({100*dropped/tot:.1f}% drop) — reasons {reasons}")


# ── stage 5: judge (2nd model pass) ───────────────────────────────────────────
def stage_judge(cfg: Config):
    conn = connect(cfg)
    sft_dir = Path(cfg.state_db).parent
    ctx = dict(conn.execute("SELECT pmcid, context FROM paper_ctx"))
    # group kept-but-unjudged pairs by paper
    bypaper: dict = {}
    for pid, pmcid, q, a in conn.execute(
            "SELECT id, pmcid, question, answer FROM pair WHERE kept=1 AND judged=-1"):
        bypaper.setdefault(pmcid, []).append({"id": pid, "question": q, "answer": a})
    papers = list(bypaper.items())
    qwen = QwenClient(base_url=cfg.model_base_url, model=cfg.model_name,
                      max_concurrency=cfg.model_concurrency,
                      enable_thinking=False, temperature=cfg.judge_temperature)
    if not qwen.health():
        sys.exit(f"[judge] vLLM not serving {cfg.model_name} at {cfg.model_base_url} — start it first.")
    print(f"[judge] {len(papers):,} papers with unjudged pairs; model={cfg.model_name}", flush=True)
    bar = Bar("judge", len(papers) or 1, sft_dir / "progress_judge.json")
    st = {"n": 0, "yes": 0, "no": 0}

    def flush(items):
        prompts = [judge_prompt(ctx.get(pmcid, ""), pairs) for pmcid, pairs in items]

        def on_result(idx, out):
            pmcid, pairs = items[idx]
            arr = None if isinstance(out, QwenError) else extract_json(out)
            if isinstance(arr, dict):
                arr = arr.get("verdicts")
            verdicts = {}
            if isinstance(arr, list):
                for v in arr:
                    if isinstance(v, dict) and "n" in v:
                        try:
                            verdicts[int(v["n"])] = 1 if v.get("supported") else 0
                        except (ValueError, TypeError):
                            pass
            for i, pr in enumerate(pairs, 1):
                # unparseable verdict -> treat as unsupported (conservative)
                j = verdicts.get(i, 0)
                conn.execute("UPDATE pair SET judged=? WHERE id=?", (j, pr["id"]))
                st["yes" if j else "no"] += 1
            st["n"] += 1
            bar.update(st["n"], extra=f"yes={st['yes']:,} no={st['no']:,}")

        qwen.generate_batch(prompts, system=SYS_JUDGE,
                            max_tokens=cfg.judge_max_tokens,
                            on_result=on_result, response_format={"type": "json_object"})
        conn.commit()

    buf = []
    for item in papers:
        buf.append(item)
        if len(buf) >= cfg.judge_chunk:
            flush(buf)
            buf = []
    if buf:
        flush(buf)
    bar.close(st["n"])
    print(f"[judge] supported {st['yes']:,} / rejected {st['no']:,}")


# ── stage 6: export ───────────────────────────────────────────────────────────
def stage_export(cfg: Config):
    conn = connect(cfg)
    where = "kept=1" + (" AND judged=1" if cfg.export_require_judge else "")
    rows = conn.execute(f"SELECT question, answer FROM pair WHERE {where}")
    out = Path(cfg.export_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w") as fh:
        for q, a in rows:
            rec = {"messages": [{"role": "user", "content": q},
                                {"role": "assistant", "content": a}]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"[export] wrote {n:,} SFT pairs -> {out} (require_judge={cfg.export_require_judge})")


# ── driver ──────────────────────────────────────────────────────────────────
STAGES = ("select", "sections", "generate", "guard", "judge", "export")

_DISPATCH = {
    "select": stage_select,
    "sections": stage_sections,
    "generate": stage_generate,
    "guard": stage_guard,
    "judge": stage_judge,
    "export": stage_export,
}


def run(stage: str, config_path: str | Path, overrides: dict | None = None):
    """Run one stage, or the whole pipeline in order when stage == 'all'.

    `overrides` (typically vars(argparse-namespace)) lets CLI flags replace
    individual config values; see _OVERRIDES for the overridable keys."""
    cfg = load_config(config_path, overrides)
    for s in (STAGES if stage == "all" else (stage,)):
        _DISPATCH[s](cfg)
