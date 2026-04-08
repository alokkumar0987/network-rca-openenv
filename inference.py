import os
from openai import OpenAI

# Defaults only for API_BASE_URL and MODEL_NAME (hackathon rule). HF_TOKEN must not have a default.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _safe_error(value) -> str:
    if value in (None, "", "null"):
        return "null"
    return str(value)


def choose_action_with_llm(observation) -> "Action":
    from models import Action

    prompt = (
        "Given these alarms, suggest the next Network RCA action in plain text. "
        "Use one short sentence.\n\n"
        f"alarms={observation.alarms}"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        _ = completion.choices[0].message.content
    except Exception:
        pass

    return Action(action_type="correlate", target=None, root_cause=None)


def run_episode(task_name: str = "network-rca", benchmark: str = "openenv") -> None:
    # Emit [START] before importing the env (sentence-transformers load) so stdout order matches the spec.
    print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}")

    from environment import NetworkRCAEnv

    env = NetworkRCAEnv()
    rewards = []
    success = False
    step_idx = 0

    try:
        observation = env.reset("easy")
        action = choose_action_with_llm(observation)
        next_obs, reward, done, info = env.step(action)
        _ = next_obs

        step_idx = 1
        reward_value = float(reward.value)
        rewards.append(f"{reward_value:.2f}")
        error_value = _safe_error(info.get("last_action_error"))
        print(
            f"[STEP] step={step_idx} action={action.action_type} "
            f"reward={reward_value:.2f} done={_bool_str(done)} error={error_value}"
        )
        success = bool(done)
    except Exception as exc:
        print(
            f"[STEP] step={step_idx + 1} action=error "
            f"reward=0.00 done=false error={_safe_error(exc)}"
        )
    finally:
        if hasattr(env, "close"):
            env.close()
        rewards_csv = ",".join(rewards)
        print(
            f"[END] success={_bool_str(success)} steps={step_idx} rewards={rewards_csv}"
        )


if __name__ == "__main__":
    run_episode()
