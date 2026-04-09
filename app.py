from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
import time
from environment import NetworkRCAEnv
from tasks import _TASKS, grade_episode
from models import Action

try:
    from baseline import run_baseline
except ImportError:
    from baseline import run_agent as run_baseline

app = FastAPI(title="Network RCA Environment")
env = None

# #region agent log
def _dbg(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "run-1") -> None:
    try:
        with open("debug-cfcb32.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "cfcb32",
                "runId": run_id,
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
# #endregion

class StepRequest(BaseModel):
    action: dict

class ResetRequest(BaseModel):
    difficulty: str = "easy"
    task_id: Optional[str] = None

class GraderRequest(BaseModel):
    conclusion: str

def _derive_required_evidence(task_data: dict, env_obj: NetworkRCAEnv) -> list:
    required = task_data.get("required_evidence", [])
    if required:
        return [str(x) for x in required]
    root_device = env_obj._extract_root_device()
    if root_device == "unknown":
        return []
    return [f"metrics:{root_device}", f"logs:{root_device}"]

@app.get("/")
def read_root():
    return {"message": "Network RCA Environment is running"}

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    global env
    env = NetworkRCAEnv()
    # OpenEnv validators may send POST /reset with an empty body.
    difficulty = req.difficulty if req is not None else "easy"
    task_id = req.task_id if req is not None else None
    # #region agent log
    _dbg("H3", "app.py:reset", "reset called", {"difficulty": difficulty, "task_id": task_id})
    # #endregion
    obs = env.reset(difficulty, task_id)
    # #region agent log
    _dbg(
        "H4",
        "app.py:reset",
        "reset selected task",
        {"selected_task_id": env.task_data.get("id", f"{env.difficulty}-0"), "difficulty": env.difficulty},
    )
    # #endregion
    return obs.dict()

@app.post("/step")
def step(req: StepRequest):
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not reset")
    try:
        action = Action(**req.action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {e}")
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not reset")
    return env.state()

@app.get("/baseline")
def baseline():
    scores = run_baseline()
    return scores

@app.post("/grader")
def grader(req: GraderRequest):
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="No active environment")
    required_evidence = _derive_required_evidence(env.task_data, env)
    gathered_evidence = {
        "evidence_keys": [f"metrics:{d}" for d in env.metrics_queried] + [f"logs:{d}" for d in env.logs_checked]
    }
    score, feedback, breakdown = grade_episode(
        conclusion=req.conclusion,
        ground_truth=env.task_data["ground_truth"],
        required_evidence=required_evidence,
        gathered_evidence=gathered_evidence,
        step_count=env.step_count,
        max_steps=env.task_data.get("max_steps", 20),
    )
    # #region agent log
    _dbg(
        "H2",
        "app.py:grader",
        "grader response generated",
        {
            "task_id": env.task_data.get("id", f"{env.difficulty}-0"),
            "score": score,
            "in_open_interval": (score > 0.0 and score < 1.0),
            "step_count": env.step_count,
            "max_steps": env.task_data.get("max_steps", 20),
            "required_evidence_count": len(required_evidence),
            "gathered_evidence_count": len(gathered_evidence["evidence_keys"]),
        },
    )
    # #endregion
    return {
        "score": score,
        "feedback": feedback,
        "breakdown": breakdown,
        "task_id": env.task_data.get("id", f"{env.difficulty}-0"),
        "required_evidence": required_evidence,
        "gathered_evidence": gathered_evidence["evidence_keys"],
    }

@app.get("/tasks")
def tasks():
    action_schema = Action.model_json_schema()
    task_list = []
    for difficulty, tasks in _TASKS.items():
        for i, task in enumerate(tasks):
            task_list.append({
                "id": task.get("id", f"{difficulty}-{i}"),
                "difficulty": difficulty,
                "index": i,
                "description": task.get("description", ""),
                "ground_truth": task.get("ground_truth", ""),
                "max_steps": task.get("max_steps", 20),
                "alarm_count": len(task.get("alarms", [])),
                "required_evidence": task.get("required_evidence", []),
                "supports_actions": ["investigate", "correlate", "query_metrics", "check_logs", "conclude"],
                "has_grader": True,
            })
    # #region agent log
    _dbg(
        "H1",
        "app.py:tasks",
        "tasks listed",
        {
            "tasks_count": len(task_list),
            "has_grader_count": sum(1 for t in task_list if t.get("has_grader") is True),
            "sample_task_ids": [t.get("id") for t in task_list[:3]],
        },
    )
    # #endregion
    return {"tasks": task_list, "action_schema": action_schema}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)