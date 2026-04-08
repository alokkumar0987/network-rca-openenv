import random
from environment import NetworkRCAEnv
from models import Action

def random_action(env):
    alarms = env.task_data.get("alarms", [])
    if random.random() < 0.3 and alarms:
        causes = ["power outage", "fiber cut", "misconfiguration", "hardware failure"]
        return Action(action_type="conclude", target=None, root_cause=random.choice(causes))
    elif random.random() < 0.6 and alarms:
        target = random.choice(alarms).id
        return Action(action_type="investigate", target=target, root_cause=None)
    else:
        return Action(action_type="correlate", target=None, root_cause=None)

def run_random():
    for difficulty in ["easy", "medium", "hard"]:
        env = NetworkRCAEnv()
        env.reset(difficulty)
        done = False
        total = 0.0
        while not done:
            act = random_action(env)
            _, reward, done, _ = env.step(act)
            total += reward.value
        print(f"{difficulty} total: {total:.2f}")

if __name__ == "__main__":
    run_random()