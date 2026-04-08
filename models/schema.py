from pydantic import BaseModel 
from typing import Optional

class KYCObservation(BaseModel):
    record_id: int
    name: str     
    age: Optional[int]
    email: Optional[str]
    phone: Optional[str]
    city: Optional[str]

class KYCAction(BaseModel):
    action_id: int