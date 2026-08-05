#!/usr/bin/env bash
# One teacher-generation session per day (default 4h), lossless + resumable.
#
# Each chunk commits to the state DB, so `timeout` cutting a run mid-way loses at
# most the in-flight chunk; the next run resumes from the checkpoint. The script
# auto-picks the next stage from DB state, so you just run it once a day:
#
#   ./scripts/run_daily.sh              # 4h GPU window
#   HOURS=6 ./scripts/run_daily.sh      # longer window (weekends)
#   (cron)  0 9 * * *  cd <repo> && ./scripts/run_daily.sh >> <sft>/daily.log 2>&1
#
# Order: select→sections (CPU, once) → generate (4h/day) → guard (CPU, once) →
#        judge (4h/day) → export. At most ONE ${HOURS}h GPU block runs per invocation.
# Needs the teacher model up for generate/judge (checked; aborts cleanly if down).
# Paths/model come from configs/teacher.yaml; PY defaults to the active env's python.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}
CFG=${CFG:-"$ROOT/configs/teacher.yaml"}
HOURS=${HOURS:-4}
cd "$ROOT"

cfg(){ "$PY" -c "import yaml;print(yaml.safe_load(open('$CFG'))$1)"; }
STATE_DB=$(cfg "['paths']['state_db']")
BASE_URL=$(cfg "['model']['base_url']")
MODEL=$(cfg "['model']['name']")
SFT=$(dirname "$STATE_DB")
mkdir -p "$SFT"

RT=("$PY" "$ROOT/scripts/run_teacher.py" --config "$CFG")

log(){ echo "[run_daily $(date '+%F %H:%M:%S')] $*"; }
# health: exit 0 if $MODEL appears in $BASE_URL/models, else non-zero (connection
# failure = unhealthy). stdlib urllib via $PY — no curl dependency.
health(){ "$PY" - "$BASE_URL" "$MODEL" <<'PY'
import sys, urllib.request
base, model = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(base + "/models", timeout=10) as r:
        body = r.read().decode()
    ok = model in body
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
}

# echoes: SEL CTX GEN KEPT JUDGED  (all 0 if the db doesn't exist yet)
state(){ STATE_DB="$STATE_DB" "$PY" - <<'PY'
import os, sqlite3
p = os.environ["STATE_DB"]
if not os.path.exists(p):
    print("0 0 0 0 0"); raise SystemExit
c = sqlite3.connect(p)
def n(sql):
    try: return c.execute(sql).fetchone()[0]
    except sqlite3.OperationalError: return 0
print(n("SELECT COUNT(*) FROM sft_paper"),
      n("SELECT COUNT(*) FROM paper_ctx"),
      n("SELECT COUNT(*) FROM gen WHERE ok=1"),
      n("SELECT COUNT(*) FROM pair WHERE kept=1"),
      n("SELECT COUNT(*) FROM pair WHERE kept=1 AND judged!=-1"))
PY
}
tol(){ local t=$(( $1 / 200 )); (( t < 200 )) && t=200; echo "$t"; }  # ~0.5%, min 200 stragglers

read -r SEL CTX GEN KEPT JUDGED < <(state)

# ── select (CPU, once) — DB-driven: over-selected to net ~80k usable ──────────
if [ "$SEL" -eq 0 ]; then
  log "select (CPU)…"; "${RT[@]}" --stage select
  read -r SEL CTX GEN KEPT JUDGED < <(state)
fi

# ── sections (CPU, once) ──────────────────────────────────────────────────────
if [ ! -f "$SFT/.sections_done" ]; then
  log "sections (CPU)…"; "${RT[@]}" --stage sections && touch "$SFT/.sections_done"
  read -r SEL CTX GEN KEPT JUDGED < <(state)
fi

# ── generate (one ${HOURS}h GPU block/day) ────────────────────────────────────
if [ ! -f "$SFT/.generate_done" ]; then
  health || { log "$MODEL not serving at $BASE_URL — start it and rerun."; exit 1; }
  log "generate ${HOURS}h  (done ${GEN}/${CTX} papers)…"
  timeout "${HOURS}h" "${RT[@]}" --stage generate || true
  read -r SEL CTX GEN KEPT JUDGED < <(state)
  if [ "$(( CTX - GEN ))" -le "$(tol "$CTX")" ]; then
    touch "$SFT/.generate_done"; log "generate COMPLETE (${GEN}/${CTX})"
    log "guard (CPU)…"; "${RT[@]}" --stage guard && touch "$SFT/.guard_done"   # free; run now, stop for the day
  else
    log "generate: $(( CTX - GEN )) papers left — more sessions needed."
  fi
  exit 0
fi

# ── guard (CPU, once) — safety net if generate finished but guard was interrupted ─
if [ ! -f "$SFT/.guard_done" ]; then
  log "guard (CPU)…"; "${RT[@]}" --stage guard && touch "$SFT/.guard_done"
  read -r SEL CTX GEN KEPT JUDGED < <(state)
fi

# ── judge (one ${HOURS}h GPU block/day) ───────────────────────────────────────
if [ ! -f "$SFT/.judge_done" ]; then
  health || { log "$MODEL not serving at $BASE_URL — start it and rerun."; exit 1; }
  log "judge ${HOURS}h  (done ${JUDGED}/${KEPT} kept pairs)…"
  timeout "${HOURS}h" "${RT[@]}" --stage judge || true
  read -r SEL CTX GEN KEPT JUDGED < <(state)
  if [ "$(( KEPT - JUDGED ))" -le "$(tol "$KEPT")" ]; then
    touch "$SFT/.judge_done"; log "judge COMPLETE (${JUDGED}/${KEPT})"
  else
    log "judge: $(( KEPT - JUDGED )) pairs left — more sessions needed."; exit 0
  fi
fi

# ── export ────────────────────────────────────────────────────────────────────
if [ ! -f "$SFT/.export_done" ]; then
  log "export…"; "${RT[@]}" --stage export && touch "$SFT/.export_done"
  read -r SEL CTX GEN KEPT JUDGED < <(state)
  log "TEACHER-GENERATION COMPLETE. selected=$SEL ctx=$CTX gen=$GEN kept=$KEPT judged=$JUDGED"
fi
