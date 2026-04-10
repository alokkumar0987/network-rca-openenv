import os
import json
from sentence_transformers import SentenceTransformer, util

_sim_model = SentenceTransformer('all-MiniLM-L6-v2')

_TASKS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'tasks.json')
if not os.path.exists(_TASKS_FILE):
    raise FileNotFoundError(f"Tasks file not found: {_TASKS_FILE}. Run extract_tasks.py first.")

with open(_TASKS_FILE, 'r') as f:
    _TASKS = json.load(f)

# Phase-2 validators require grader scores strictly inside (0, 1), not 0.0 or 1.0.
# Use a margin large enough to survive float/json round-trips and strict boundary checks.
_GRADER_EPS = 0.01


def _clamp_grader_score_open_interval(score: float) -> float:
    x = float(score)
    return min(1.0 - _GRADER_EPS, max(_GRADER_EPS, x))


def get_task(difficulty: str):
    if difficulty not in _TASKS:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    tasks_list = _TASKS[difficulty]
    if not tasks_list:
        raise ValueError(f"No tasks available for {difficulty}")
    return json.loads(json.dumps(tasks_list[0]))


def get_task_by_id(task_id: str):
    target = str(task_id or "").strip()
    if not target:
        raise ValueError("task_id is required")
    # Prefer explicit task.id if present, else fallback to difficulty-index convention.
    for difficulty, tasks_list in _TASKS.items():
        for i, task in enumerate(tasks_list):
            candidate = str(task.get("id", f"{difficulty}-{i}"))
            if candidate == target:
                return json.loads(json.dumps(task))
    raise ValueError(f"Unknown task_id: {target}")

def _normalize_text(value: str) -> str:
    return " ".join((value or "").lower().strip().split())

def _score_root_cause(conclusion: str, ground_truth: str) -> tuple[float, str]:
    c = _normalize_text(conclusion)
    gt = _normalize_text(ground_truth)
    if not c:
        return _clamp_grader_score_open_interval(0.0), "No conclusion provided"

    if c == gt:
        return _clamp_grader_score_open_interval(1.0), "Exact match"

    # High-confidence domain aliases/synonyms
    keyword_groups = [
        ("power outage", ["power", "outage"]),
        ("fiber cut", ["fiber", "cut"]),
        ("misconfiguration", ["misconfig", "configuration"]),
        ("node down", ["node", "down"]),
        ("interface down", ["interface", "down"]),
    ]
    for label, terms in keyword_groups:
        if label in gt and all(t in c for t in terms):
            return _clamp_grader_score_open_interval(0.9), f"Partial match ({label})"

    emb1 = _sim_model.encode(conclusion, convert_to_tensor=True)
    emb2 = _sim_model.encode(ground_truth, convert_to_tensor=True)
    similarity = float(util.cos_sim(emb1, emb2).item())
    if similarity < 0.1:
        return _clamp_grader_score_open_interval(0.0), "Conclusion unrelated to ground truth"
    score = max(0.0, min(0.8, (similarity - 0.1) / 0.9))
    return _clamp_grader_score_open_interval(score), f"Semantic similarity: {similarity:.2f}"

def _score_evidence(required_evidence: list, gathered_evidence: dict) -> tuple[float, list]:
    required = [str(x) for x in (required_evidence or [])]
    if not required:
        return _clamp_grader_score_open_interval(1.0), []
    gathered = set(str(x) for x in (gathered_evidence or {}).get("evidence_keys", []))
    missing = [k for k in required if k not in gathered]
    coverage = float((len(required) - len(missing)) / max(1, len(required)))
    return _clamp_grader_score_open_interval(coverage), missing

def _score_efficiency(step_count: int, max_steps: int) -> float:
    if max_steps <= 0:
        return _clamp_grader_score_open_interval(0.0)
    used_ratio = max(0.0, min(1.0, float(step_count) / float(max_steps)))
    return _clamp_grader_score_open_interval(1.0 - used_ratio)

def grade_episode(
    conclusion: str,
    ground_truth: str,
    required_evidence: list,
    gathered_evidence: dict,
    step_count: int,
    max_steps: int,
) -> tuple[float, str, dict]:
    root_score, root_feedback = _score_root_cause(conclusion, ground_truth)
    evidence_score, missing_evidence = _score_evidence(required_evidence, gathered_evidence)
    efficiency_score = _score_efficiency(step_count, max_steps)

    weights = {"root": 0.5, "evidence": 0.3, "efficiency": 0.2}
    final_score = (
        weights["root"] * root_score
        + weights["evidence"] * evidence_score
        + weights["efficiency"] * efficiency_score
    )
    final_score = max(0.0, min(1.0, final_score))
    final_score = _clamp_grader_score_open_interval(final_score)

    # Some validators scan all floats in the grader payload; keep components off 0.0/1.0 too.
    breakdown = {
        "root_cause_score": round(_clamp_grader_score_open_interval(root_score), 4),
        "evidence_score": round(_clamp_grader_score_open_interval(evidence_score), 4),
        "efficiency_score": round(_clamp_grader_score_open_interval(efficiency_score), 4),
        "weights": weights,
        "missing_evidence": missing_evidence,
    }
    feedback = f"{root_feedback}; evidence={evidence_score:.2f}; efficiency={efficiency_score:.2f}"
    return final_score, feedback, breakdown

def grade_conclusion(conclusion: str, ground_truth: str) -> tuple[float, str]:
    if not conclusion:
        return _clamp_grader_score_open_interval(0.0), "No conclusion provided"
    score, feedback = _score_root_cause(conclusion, ground_truth)
    return _clamp_grader_score_open_interval(score), feedback