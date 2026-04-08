import os
import json
from sentence_transformers import SentenceTransformer, util

_sim_model = SentenceTransformer('all-MiniLM-L6-v2')

_TASKS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'tasks.json')
if not os.path.exists(_TASKS_FILE):
    raise FileNotFoundError(f"Tasks file not found: {_TASKS_FILE}. Run extract_tasks.py first.")

with open(_TASKS_FILE, 'r') as f:
    _TASKS = json.load(f)

def get_task(difficulty: str):
    if difficulty not in _TASKS:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    tasks_list = _TASKS[difficulty]
    if not tasks_list:
        raise ValueError(f"No tasks available for {difficulty}")
    return json.loads(json.dumps(tasks_list[0]))

def _normalize_text(value: str) -> str:
    return " ".join((value or "").lower().strip().split())

def _score_root_cause(conclusion: str, ground_truth: str) -> tuple[float, str]:
    c = _normalize_text(conclusion)
    gt = _normalize_text(ground_truth)
    if not c:
        return 0.0, "No conclusion provided"

    if c == gt:
        return 1.0, "Exact match"

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
            return 0.9, f"Partial match ({label})"

    emb1 = _sim_model.encode(conclusion, convert_to_tensor=True)
    emb2 = _sim_model.encode(ground_truth, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()
    if similarity < 0.1:
        return 0.0, "Conclusion unrelated to ground truth"
    score = max(0.0, min(0.8, (similarity - 0.1) / 0.9))
    return score, f"Semantic similarity: {similarity:.2f}"

def _score_evidence(required_evidence: list, gathered_evidence: dict) -> tuple[float, list]:
    required = [str(x) for x in (required_evidence or [])]
    if not required:
        return 1.0, []
    gathered = set(str(x) for x in (gathered_evidence or {}).get("evidence_keys", []))
    missing = [k for k in required if k not in gathered]
    coverage = (len(required) - len(missing)) / len(required)
    return max(0.0, min(1.0, coverage)), missing

def _score_efficiency(step_count: int, max_steps: int) -> float:
    if max_steps <= 0:
        return 0.0
    used_ratio = max(0.0, min(1.0, float(step_count) / float(max_steps)))
    return 1.0 - used_ratio

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

    breakdown = {
        "root_cause_score": round(root_score, 4),
        "evidence_score": round(evidence_score, 4),
        "efficiency_score": round(efficiency_score, 4),
        "weights": weights,
        "missing_evidence": missing_evidence,
    }
    feedback = f"{root_feedback}; evidence={evidence_score:.2f}; efficiency={efficiency_score:.2f}"
    return final_score, feedback, breakdown

def grade_conclusion(conclusion: str, ground_truth: str) -> tuple[float, str]:
    if not conclusion:
        return 0.0, "No conclusion provided"
    return _score_root_cause(conclusion, ground_truth)