import json
import sys
import urllib.error
import urllib.request


BASE_URL = "http://127.0.0.1:7860"


def _request(method: str, path: str, body: dict | None = None) -> dict:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = resp.read().decode("utf-8")
        return json.loads(payload) if payload else {}


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    try:
        root = _request("GET", "/")
        checks.append(("/", "message" in root, "root message missing"))

        reset = _request("POST", "/reset", {"difficulty": "easy"})
        checks.append(("/reset", isinstance(reset.get("alarms"), list), "alarms not returned"))

        step = _request(
            "POST",
            "/step",
            {"action": {"action_type": "investigate", "target": "A1", "root_cause": None}},
        )
        checks.append(("/step", "observation" in step and "reward" in step, "step payload incomplete"))

        state = _request("GET", "/state")
        checks.append(("/state", isinstance(state.get("step_count"), int), "step_count missing"))

        tasks = _request("GET", "/tasks")
        checks.append(("/tasks", len(tasks.get("tasks", [])) >= 3, "tasks < 3"))

        grader = _request("POST", "/grader", {"conclusion": "power outage"})
        score = grader.get("score")
        # Hackathon / OpenEnv Phase 2: scores must be strictly inside (0, 1)
        score_ok = isinstance(score, (int, float)) and 0.0 < float(score) < 1.0
        checks.append(("/grader", score_ok, "grader score must be strictly between 0 and 1"))

        baseline = _request("GET", "/baseline")
        has_results = isinstance(baseline.get("results"), dict) and len(baseline["results"]) > 0
        checks.append(("/baseline", has_results, "baseline results missing"))
    except urllib.error.URLError as e:
        print(f"ERROR: cannot reach server at {BASE_URL} ({e})")
        print("Start server first: uvicorn app:app --host 0.0.0.0 --port 7860")
        return 1
    except Exception as e:
        print(f"ERROR: smoke test failed with exception: {e}")
        return 1

    failed = [c for c in checks if not c[1]]
    for name, ok, msg in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + ("" if ok else f" -> {msg}"))

    if failed:
        return 1
    print("All endpoint smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
