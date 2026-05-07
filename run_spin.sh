#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Common knobs (edit defaults here or override with environment variables)
# Usage: ./run_spin.sh [STUDY_ID] [MODEL_NAME] [METHOD_NAME]
#   MODEL_NAME must match a key in material/models (case-insensitive).
# =============================================================================

CONDA_ENV="${CONDA_ENV:-GTAgent}"

STUDY_ID="${1:-S_T}"
MODEL_NAME="${2:-gpt-5.4-mini}"
METHOD_NAME="${3:-spin}"

NUM_WORKERS="${NUM_WORKERS:-4}"
REPEATS="${REPEATS:-1}"
TEMPERATURE="${TEMPERATURE:-1}"
RANDOM_SEED="${RANDOM_SEED:-42}"

# Optional: cap SPIN substeps (raise if the model truncates JSON)
SPIN_PERSONALITY_MAX_TOKENS="${SPIN_PERSONALITY_MAX_TOKENS:-1024}"
SPIN_ELICITATION_MAX_TOKENS="${SPIN_ELICITATION_MAX_TOKENS:-1024}"
SPIN_DECISION_MAX_TOKENS="${SPIN_DECISION_MAX_TOKENS:-1024}"

# --- Less common (usually leave unset) --------------------------------------
# N_PARTICIPANTS=20              # quick smoke test; omit for full study sample
# RUN_NAME=my_manual_run         # fixed results folder name (default is auto)
# SPIN_DATA_DIR=/path/to/data    # overrides default ../data next to this script
# SKIP_AZURE_PROFILE=1           # skip material/models; use existing API env only
# RUN_SPIN_VERBOSE=1             # print extra banner lines (seed, workers, API hint)

# This script: material/run_spin.sh → SPIN code: material/SPIN/ → API map: material/models
MATERIAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
SPIN_ROOT="${MATERIAL_ROOT}/SPIN"
export SPIN_DATA_DIR="${SPIN_DATA_DIR:-${MATERIAL_ROOT}/data}"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
cd "$SPIN_ROOT"

if [[ -z "${SKIP_AZURE_PROFILE:-}" ]]; then
  eval "$(
    python3 - "$MATERIAL_ROOT" "$MODEL_NAME" <<'PY'
import json, re, sys

def shell_sq(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def norm_azure_endpoint(raw: str) -> str:
    u = (raw or "").strip().rstrip("/")
    if not u:
        return u
    u = re.split(r"[?#]", u, maxsplit=1)[0].rstrip("/")
    for suffix in (
        "/openai/responses",
        "/openai/chat/completions",
        "/openai/deployments",
    ):
        if u.lower().endswith(suffix.lower()):
            u = u[: -len(suffix)].rstrip("/")
            break
    return u


root, model = sys.argv[1], sys.argv[2]
path = f"{root}/models"
try:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: file not found: {path}", file=sys.stderr)
    sys.exit(2)
except json.JSONDecodeError as e:
    print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
    sys.exit(2)

if model in data:
    entry = data[model]
else:
    lower = {str(k).lower(): k for k in data}
    lk = model.lower()
    if lk not in lower:
        known = ", ".join(sorted(data))
        print(
            f"Error: model {model!r} is not configured in material/models.\nKnown models: {known or '(none)'}",
            file=sys.stderr,
        )
        sys.exit(3)
    entry = data[lower[lk]]

endpoint = (entry.get("endpoint") or "").strip()
api_version = (entry.get("api_version") or "").strip()
api_key = (entry.get("api_key") or "").strip()
if not endpoint or not api_key:
    print(
        "Error: each model entry must set endpoint, api_version (empty string for OpenAI-compatible), and api_key.",
        file=sys.stderr,
    )
    sys.exit(4)

lines = []
if api_version:
    endpoint = norm_azure_endpoint(endpoint)
    lines.extend(
        [
            f"export AZURE_OPENAI_ENDPOINT={shell_sq(endpoint)}",
            f"export AZURE_OPENAI_API_VERSION={shell_sq(api_version)}",
            "unset OPENAI_BASE_URL",
            "unset AZURE_INFERENCE_DEPLOYMENT_NAME",
            f"export OPENAI_API_KEY={shell_sq(api_key)}",
            f"export AZURE_OPENAI_API_KEY={shell_sq(api_key)}",
        ]
    )
else:
    lines.extend(
        [
            f"export OPENAI_BASE_URL={shell_sq(endpoint)}",
            "unset AZURE_OPENAI_ENDPOINT",
            "unset AZURE_OPENAI_API_VERSION",
            "unset AZURE_INFERENCE_DEPLOYMENT_NAME",
            f"export OPENAI_API_KEY={shell_sq(api_key)}",
        ]
    )
print("\n".join(lines))
PY
  )"
fi

MODEL_SLUG="${MODEL_NAME//\//_}"
MODEL_SLUG="${MODEL_SLUG//-/_}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-${STUDY_ID}_${MODEL_SLUG}_${TIMESTAMP}}"
RESULTS_BASE_DIR="results/${METHOD_NAME}/${RUN_NAME}"

N_PARTICIPANTS_ARG=()
if [[ -n "${N_PARTICIPANTS:-}" ]]; then
  N_PARTICIPANTS_ARG=(--n-participants "$N_PARTICIPANTS")
fi

if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ -z "${AZURE_OPENAI_API_KEY:-}" ]]; then
  echo "Error: set api_key in material/models or export OPENAI_API_KEY / AZURE_OPENAI_API_KEY." >&2
  exit 1
fi

export RESULTS_METHOD_NAME="$METHOD_NAME"

# Compact banner (set RUN_SPIN_VERBOSE=1 for extra lines)
if [[ -t 1 ]]; then
  _dim=$'\033[2m'
  _bold=$'\033[1m'
  _rst=$'\033[0m'
else
  _dim="" _bold="" _rst=""
fi

echo ""
echo "${_bold}SPIN${_rst}  ${STUDY_ID}  ·  ${MODEL_NAME}  ·  ${METHOD_NAME}"
echo "${_dim}run${_rst}    ${RUN_NAME}"
echo "${_dim}out${_rst}    ${RESULTS_BASE_DIR}"
if [[ -n "${RUN_SPIN_VERBOSE:-}" ]]; then
  echo "${_dim}temp${_rst}   ${TEMPERATURE}  ${_dim}seed${_rst} ${RANDOM_SEED}  ${_dim}workers${_rst} ${NUM_WORKERS}"
  if [[ -n "${N_PARTICIPANTS:-}" ]]; then
    echo "${_dim}n${_rst}       ${N_PARTICIPANTS} (override)"
  fi
  if [[ -n "${AZURE_OPENAI_ENDPOINT:-}" ]]; then
    echo "${_dim}api${_rst}    azure (${AZURE_OPENAI_API_VERSION:-?})"
  elif [[ -n "${OPENAI_BASE_URL:-}" ]]; then
    echo "${_dim}api${_rst}    openai-compatible base set"
  fi
elif [[ -n "${N_PARTICIPANTS:-}" ]]; then
  echo "${_dim}n${_rst}       ${N_PARTICIPANTS} (override)"
fi
echo ""

testing_args=(
  -m evaluation.cli
  --stage testing
  --study-id "$STUDY_ID"
  --real-llm
  --provider openai
  --model "$MODEL_NAME"
  --repeats "$REPEATS"
  --run-name "$RUN_NAME"
  --num-workers "$NUM_WORKERS"
  --random-seed "$RANDOM_SEED"
  --temperature "$TEMPERATURE"
  --agent-method spin
  --spin-personality-max-tokens "$SPIN_PERSONALITY_MAX_TOKENS"
  --spin-elicitation-max-tokens "$SPIN_ELICITATION_MAX_TOKENS"
  --spin-decision-max-tokens "$SPIN_DECISION_MAX_TOKENS"
  --spin-save-raw-prompts
  --spin-save-raw-responses
)
if [[ "${#N_PARTICIPANTS_ARG[@]}" -gt 0 ]]; then
  testing_args+=("${N_PARTICIPANTS_ARG[@]}")
fi
python3 "${testing_args[@]}"

evaluation_args=(
  -m evaluation.cli
  --stage evaluation
  --study-id "$STUDY_ID"
  --run-name "$RUN_NAME"
  --skip-generation
)
python3 "${evaluation_args[@]}"

RESULT_DIR_ABS="${SPIN_ROOT}/${RESULTS_BASE_DIR}"
echo ""
if [[ -f "${RESULTS_BASE_DIR}/evaluation_results.json" ]]; then
  echo "${_bold}Done${_rst}  ${RESULT_DIR_ABS}/"
  echo "${_dim}      └${_rst} evaluation_results.json"
else
  echo "${_bold}Done${_rst}  ${RESULT_DIR_ABS}/"
  echo "Warning: evaluation_results.json missing — see evaluation output above." >&2
fi
