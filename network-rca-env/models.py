from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict

class Alarm(BaseModel):
    id: str
    code: str
    name: str
    severity: str
    device: str
    description: str

class Observation(BaseModel):
    alarms: List[Alarm]
    topology_edges: List[Tuple[str, str]]
    step_count: int
    metrics: Optional[Dict[str, Dict[str, float]]] = None
    logs: Optional[Dict[str, List[str]]] = None

class Action(BaseModel):
    action_type: str          # "investigate", "correlate", "conclude", "query_metrics", "check_logs"
    target: Optional[str]
    root_cause: Optional[str]

class Reward(BaseModel):
    value: float
    details: str