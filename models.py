from pydantic import BaseModel 
from typing import Optional

class KYCObservation(BaseModel):
    record_id: int
    name: str     
    age: Optional[int]
    email: Optional[str]
    phone: Optional[str]
    city: Optional[str]
    step_count: int
    episode_id: str

class KYCAction(BaseModel):
    action_id: int

class StepResult(BaseModel):
    observation: Optional[KYCObservation]
    reward: float
    done: bool
    info: dict