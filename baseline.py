"""
Hybrid agent: LangGraph workflow + OpenAI reasoning.
"""

import argparse
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

from environment import NetworkRCAEnv
from models import Action, Observation

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "150"))
MAX_RETRIES = 3
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "deepseek-v3.1:671b-cloud")


def _make_client() -> OpenAI:
    # Hackathon baseline requirement: use OpenAI API key from environment.
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


def _ollama_model_available(model: str) -> bool:
    try:
        out = subprocess.run(
            ["ollama", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        available = model in (out.stdout or "")
        return available
    except Exception:
        return False


def _call_ollama_fallback(prompt: str) -> str:
    fallback_client = OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")
    response = fallback_client.chat.completions.create(
        model=OLLAMA_FALLBACK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    content = (response.choices[0].message.content or "").strip()
    return content


def call_openai_with_retry(prompt: str, max_retries: int = MAX_RETRIES, initial_delay: int = 1) -> str:
    """Call model endpoint with exponential backoff."""
    client = _make_client()
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
            return content
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
        except Exception:
            # Fallback path: keep OPENAI_API_KEY setup and switch to Ollama when available.
            if _ollama_model_available(OLLAMA_FALLBACK_MODEL):
                try:
                    return _call_ollama_fallback(prompt)
                except Exception:
                    pass
            raise
    raise RuntimeError("Max retries exceeded")


class AgentState(TypedDict):
    env: NetworkRCAEnv
    observation: Observation
    step_count: int
    total_reward: float
    done: bool
    action_sequence: List[str]
    current_device: Optional[str]
    inferred_root_cause: Optional[str]


def investigate(state: AgentState) -> AgentState:
    obs = state["observation"]
    env = state["env"]
    if obs.alarms:
        alarm = obs.alarms[0]
        state["current_device"] = alarm.device
        action = Action(action_type="investigate", target=alarm.id, root_cause=None)
        obs, reward, done, _ = env.step(action)
        state["observation"] = obs
        state["total_reward"] += reward.value
        state["step_count"] += 1
        state["done"] = done
        state["action_sequence"].append("investigate")
    return state


def query_metrics(state: AgentState) -> AgentState:
    env = state["env"]
    device = state["current_device"]
    action = Action(action_type="query_metrics", target=device, root_cause=None)
    obs, reward, done, _ = env.step(action)
    state["observation"] = obs
    state["total_reward"] += reward.value
    state["step_count"] += 1
    state["done"] = done
    state["action_sequence"].append("query_metrics")
    return state


def check_logs(state: AgentState) -> AgentState:
    env = state["env"]
    device = state["current_device"]
    action = Action(action_type="check_logs", target=device, root_cause=None)
    obs, reward, done, _ = env.step(action)
    state["observation"] = obs
    state["total_reward"] += reward.value
    state["step_count"] += 1
    state["done"] = done
    state["action_sequence"].append("check_logs")
    return state


def infer_root_cause(state: AgentState) -> AgentState:
    obs = state["observation"]
    metrics = obs.metrics or {}
    logs = obs.logs or {}
    prompt = f"""You are a network engineer performing root cause analysis.

Metrics from devices:
{json.dumps(metrics, indent=2)}

Logs from devices:
{json.dumps(logs, indent=2)}

Based on this evidence, return ONLY valid JSON:
{{"root_cause":"short diagnosis"}}
"""
    try:
        content = call_openai_with_retry(prompt)
        data = json.loads(content)
        state["inferred_root_cause"] = data.get("root_cause", "Unknown")
    except Exception:
        state["inferred_root_cause"] = "Unknown"
    return state


def conclude(state: AgentState) -> AgentState:
    env = state["env"]
    reason = state["inferred_root_cause"]
    action = Action(action_type="conclude", target=None, root_cause=reason)
    obs, reward, done, info = env.step(action)
    state["observation"] = obs
    state["total_reward"] += reward.value
    state["step_count"] += 1
    state["done"] = done
    state["action_sequence"].append("conclude")
    return state


builder = StateGraph(AgentState)
builder.add_node("investigate", investigate)
builder.add_node("query_metrics", query_metrics)
builder.add_node("check_logs", check_logs)
builder.add_node("infer", infer_root_cause)
builder.add_node("conclude", conclude)
builder.set_entry_point("investigate")
builder.add_edge("investigate", "query_metrics")
builder.add_edge("query_metrics", "check_logs")
builder.add_edge("check_logs", "infer")
builder.add_edge("infer", "conclude")
builder.add_edge("conclude", END)
agent_graph = builder.compile()


def run_agent(difficulty: str = "hard") -> Dict[str, Any]:
    env = NetworkRCAEnv()
    obs = env.reset(difficulty)
    state: AgentState = {
        "env": env,
        "observation": obs,
        "step_count": 0,
        "total_reward": 0.0,
        "done": False,
        "action_sequence": [],
        "current_device": None,
        "inferred_root_cause": None,
    }
    final = agent_graph.invoke(state)
    out = {
        "difficulty": difficulty,
        "steps_taken": final["step_count"],
        "total_reward": float(max(0.01, min(0.99, final["total_reward"]))),
        "done": bool(final["done"]),
        "action_sequence": final["action_sequence"],
        "inferred_root_cause": final["inferred_root_cause"],
    }
    return out


def run_baseline(difficulties: List[str] | None = None, seed: int = 42, episodes: int = 1) -> Dict[str, Any]:
    # Compatibility shim for app.py /baseline endpoint.
    del seed
    difficulties = difficulties or ["easy", "medium", "hard"]
    results: Dict[str, Any] = {"episodes": episodes, "results": {}, "aggregate": {}}
    all_rewards: List[float] = []
    for d in difficulties:
        episode_reports = []
        for _ in range(episodes):
            report = run_agent(difficulty=d)
            episode_reports.append(report)
            all_rewards.append(float(report["total_reward"]))
        avg = sum(float(r["total_reward"]) for r in episode_reports) / max(len(episode_reports), 1)
        results["results"][d] = {"average_total_reward": round(avg, 6), "episodes": episode_reports}
    overall = sum(all_rewards) / max(len(all_rewards), 1) if all_rewards else 0.0
    results["aggregate"] = {"overall_average_total_reward": round(overall, 6), "difficulty_count": len(difficulties)}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph hybrid baseline runner")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"], default="hard")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    if args.difficulty == "all":
        report = run_baseline(difficulties=["easy", "medium", "hard"], episodes=args.episodes)
    else:
        report = run_agent(difficulty=args.difficulty)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()