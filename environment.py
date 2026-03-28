from typing import Tuple, Dict, Any
try:
    from .models import Observation, Action, Reward, Alarm
    from .tasks import get_task, grade_episode
except ImportError:
    from models import Observation, Action, Reward, Alarm
    from tasks import get_task, grade_episode

# Reward constants
REWARD_INVEST_RELEVANT = 0.25
REWARD_INVEST_IRRELEVANT = -0.1
REWARD_INVEST_REDUNDANT = -0.05
REWARD_CORRELATE = 0.1
REWARD_DEPENDENCY_DISCOVERY = 0.2
REWARD_QUERY_METRICS = 0.15
REWARD_CHECK_LOGS = 0.15
STEP_PENALTY = -0.2
PENALTY_WRONG_CONCLUSION = -0.3
PENALTY_TIMEOUT = -0.5
PENALTY_INSUFFICIENT_EVIDENCE = -0.3

class NetworkRCAEnv:
    def __init__(self):
        self.task_data = None
        self.step_count = 0
        self.done = False
        self.investigated = set()
        self.metrics_queried = set()
        self.logs_checked = set()
        self.difficulty = None
        self.alarm_queue = []
        self.expiring_alarms = {}
        self.termination_reason = None
        self.allowed_action_types = {"investigate", "correlate", "query_metrics", "check_logs", "conclude"}

    def reset(self, difficulty: str = "easy") -> Observation:
        self.difficulty = difficulty
        self.task_data = get_task(difficulty)
        self.step_count = 0
        self.done = False
        self.termination_reason = None
        self.investigated.clear()
        self.metrics_queried.clear()
        self.logs_checked.clear()
        self.alarm_queue = []
        self.expiring_alarms.clear()

        # Convert alarms to Alarm objects
        alarms_dict = self.task_data["alarms"]
        self.task_data["alarms"] = [Alarm(**a) for a in alarms_dict]

        # Future alarms
        for step, alarm_dict in self.task_data.get("future_alarms", []):
            alarm = Alarm(**alarm_dict)
            self.alarm_queue.append((step, alarm))
            if alarm_dict.get("expires_at") is not None:
                self.expiring_alarms[alarm.id] = alarm_dict["expires_at"]

        return Observation(
            alarms=self.task_data["alarms"],
            topology_edges=self.task_data.get("topology_edges", []),
            step_count=0,
            metrics=None,
            logs=None
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode already done. Call reset().")

        self._remove_expired_alarms()

        reward_value = 0.0
        details = []
        target_norm = action.target.strip().lower() if action.target else None
        action_type = (action.action_type or "").strip().lower()

        if action_type not in self.allowed_action_types:
            reward_value -= 0.2
            details.append(f"invalid action type {action.action_type}")
            action_type = "correlate"

        # Process action
        if action_type == "investigate":
            if target_norm is None:
                reward_value -= 0.2
                details.append("investigation target missing")
            else:
                # Match alarm ID
                alarm_match = None
                for a in self.task_data["alarms"]:
                    if a.id.strip().lower() == target_norm:
                        alarm_match = a
                        break
                if alarm_match:
                    original_id = alarm_match.id
                    if original_id not in self.investigated:
                        self.investigated.add(original_id)
                        detailed = self.task_data.get("detailed_descriptions", {}).get(original_id)
                        if detailed:
                            alarm_match.description = detailed
                            details.append("investigation revealed new details")
                        if original_id in self.task_data.get("relevant_alarm_ids", set()):
                            reward_value += REWARD_INVEST_RELEVANT
                            if original_id in self.task_data.get("dependency_alarms", set()):
                                reward_value += REWARD_DEPENDENCY_DISCOVERY
                                details.append("discovered dependency")
                            else:
                                details.append("investigated relevant alarm")
                        else:
                            reward_value += REWARD_INVEST_IRRELEVANT
                            details.append("investigated irrelevant alarm")
                    else:
                        reward_value += REWARD_INVEST_REDUNDANT
                        details.append("redundant investigation")
                # Device investigation
                elif target_norm in [d.strip().lower() for edge in self.task_data.get("topology_edges", []) for d in edge]:
                    root_device = self._extract_root_device()
                    if target_norm == root_device.strip().lower():
                        reward_value += 0.15
                        details.append(f"investigated root device {target_norm}")
                    else:
                        reward_value -= 0.05
                        details.append(f"investigated non-root device {target_norm}")
                else:
                    reward_value -= 0.2
                    details.append(f"invalid investigation target {action.target}")

        elif action_type == "correlate":
            reward_value += REWARD_CORRELATE
            details.append("correlation attempted")

        elif action_type == "query_metrics":
            if target_norm is None:
                reward_value -= 0.2
                details.append("query_metrics target missing")
            else:
                if target_norm in [d.strip().lower() for edge in self.task_data.get("topology_edges", []) for d in edge]:
                    device_name = next(d for edge in self.task_data.get("topology_edges", []) for d in edge if d.strip().lower() == target_norm)
                    if device_name not in self.metrics_queried:
                        self.metrics_queried.add(device_name)
                        metrics = self.task_data.get("metrics", {}).get(device_name, {})
                        reward_value += REWARD_QUERY_METRICS
                        details.append(f"queried metrics for {device_name}: {metrics}")
                    else:
                        reward_value += REWARD_INVEST_REDUNDANT
                        details.append("metrics already queried")
                else:
                    reward_value -= 0.2
                    details.append(f"invalid device {action.target} for query_metrics")

        elif action_type == "check_logs":
            if target_norm is None:
                reward_value -= 0.2
                details.append("check_logs target missing")
            else:
                if target_norm in [d.strip().lower() for edge in self.task_data.get("topology_edges", []) for d in edge]:
                    device_name = next(d for edge in self.task_data.get("topology_edges", []) for d in edge if d.strip().lower() == target_norm)
                    if device_name not in self.logs_checked:
                        self.logs_checked.add(device_name)
                        logs = self.task_data.get("logs", {}).get(device_name, [])
                        reward_value += REWARD_CHECK_LOGS
                        details.append(f"checked logs for {device_name}: {logs[:3]}...")
                    else:
                        reward_value += REWARD_INVEST_REDUNDANT
                        details.append("logs already checked")
                else:
                    reward_value -= 0.2
                    details.append(f"invalid device {action.target} for check_logs")

        elif action_type == "conclude":
            if action.root_cause is None:
                reward_value -= 0.5
                details.append("no root cause provided")
            else:
                required_evidence = self.task_data.get("required_evidence", [])
                if not required_evidence:
                    root_device = self._extract_root_device()
                    if root_device != "unknown":
                        required_evidence = [f"metrics:{root_device}", f"logs:{root_device}"]

                evidence_keys = [f"metrics:{d}" for d in self.metrics_queried] + [f"logs:{d}" for d in self.logs_checked]
                score, feedback, breakdown = grade_episode(
                    conclusion=action.root_cause,
                    ground_truth=self.task_data["ground_truth"],
                    required_evidence=required_evidence,
                    gathered_evidence={"evidence_keys": evidence_keys},
                    step_count=self.step_count,
                    max_steps=self.task_data.get("max_steps", 20),
                )
                reward_value = score
                if breakdown["root_cause_score"] < 0.5:
                    reward_value -= PENALTY_WRONG_CONCLUSION
                if breakdown["missing_evidence"]:
                    reward_value += PENALTY_INSUFFICIENT_EVIDENCE
                    details.append("concluded without sufficient evidence")
                details.append(feedback)
                self.done = True
                self.termination_reason = "concluded"
        else:
            reward_value -= 0.2
            details.append(f"invalid action type {action.action_type}")

        # Step penalty (except on conclusion)
        if action_type != "conclude":
            reward_value += STEP_PENALTY
            details.append("step penalty")

        self.step_count += 1
        self._add_pending_alarms()

        # Timeout
        if self.step_count >= self.task_data.get("max_steps", 20) and not self.done:
            self.done = True
            self.termination_reason = "timeout"
            reward_value += PENALTY_TIMEOUT
            details.append("timeout")

        # Build observation with revealed metrics/logs
        obs_metrics = {d: self.task_data.get("metrics", {}).get(d, {}) for d in self.metrics_queried}
        obs_logs = {d: self.task_data.get("logs", {}).get(d, []) for d in self.logs_checked}

        obs = Observation(
            alarms=self.task_data["alarms"],
            topology_edges=self.task_data.get("topology_edges", []),
            step_count=self.step_count,
            metrics=obs_metrics if obs_metrics else None,
            logs=obs_logs if obs_logs else None
        )
        reward = Reward(value=max(-1.0, min(1.0, reward_value)), details="; ".join(details))
        info = {
            "task_id": self.task_data.get("id", f"{self.difficulty}-0"),
            "termination_reason": self.termination_reason,
            "investigated": list(self.investigated),
            "metrics_queried": list(self.metrics_queried),
            "logs_checked": list(self.logs_checked),
            "actions_taken": self.step_count,
        }
        return obs, reward, self.done, info

    def state(self) -> Dict[str, Any]:
        return {
            "difficulty": self.difficulty,
            "step_count": self.step_count,
            "investigated": list(self.investigated),
            "metrics_queried": list(self.metrics_queried),
            "logs_checked": list(self.logs_checked),
            "done": self.done,
            "alarm_count": len(self.task_data["alarms"])
        }

    def _extract_root_device(self) -> str:
        gt = self.task_data["ground_truth"].lower()
        for dev in ["R1", "R2", "R3", "R4", "R5"]:
            if dev.lower() in gt:
                return dev
        edges = self.task_data.get("topology_edges", [])
        if edges:
            return edges[0][0]
        return "unknown"

    def _remove_expired_alarms(self):
        to_remove = [aid for aid, expire_step in self.expiring_alarms.items() if expire_step <= self.step_count]
        for aid in to_remove:
            self.task_data["alarms"] = [a for a in self.task_data["alarms"] if a.id != aid]
            del self.expiring_alarms[aid]

    def _add_pending_alarms(self):
        new_alarms = []
        remaining_queue = []
        for step, alarm in self.alarm_queue:
            if step <= self.step_count:
                if alarm.id not in self.expiring_alarms or self.expiring_alarms[alarm.id] > step:
                    new_alarms.append(alarm)
            else:
                remaining_queue.append((step, alarm))
        self.alarm_queue = remaining_queue
        if new_alarms:
            self.task_data["alarms"].extend(new_alarms)