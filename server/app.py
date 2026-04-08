import json
import time
import uvicorn
from app import app


# #region agent log
def _dbg(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    with open("debug-eeaaee.log", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "sessionId": "eeaaee",
            "runId": "openenv-validate",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }) + "\n")
# #endregion


def main() -> None:
    _dbg("H1", "server/app.py:22", "main() invoked", {"entrypoint": "server.app:main"})
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    _dbg("H2", "server/app.py:27", "__main__ execution path", {})
    main()
