import pandas as pd 
from typing import Optional
from models.schema import KYCAction, KYCObservation

class KYCEnv:
    def __init__(self):
        self.df = pd.read_csv("kyc_data.csv")
        self.cursor = 0
    
    def reset(self) -> KYCObservation:
        self.cursor = 0
        return self.getObservation()

    def getObservation(self) -> Optional[KYCObservation]:
        if self.cursor < 0 or self.cursor >= len(self.df):
            return None
        row = self.df.iloc[self.cursor]

        return KYCObservation(
            record_id = self.cursor,
            name=str(row.get('name', '')) if pd.notna(row.get("name")) else "",
            age=row.get('age') if pd.notna(row.get('age')) else None,
            email=str(row.get('email', '')) if pd.notna(row.get('email')) else None,
            phone=str(row.get('phone', '')) if pd.notna(row.get('phone')) else None,
            city=str(row.get('city', '')) if pd.notna(row.get('city')) else None
        )

    def step(self, action: KYCAction):
        obs = self.getObservation()
        if obs is None:
            return None, 0.0, True, {"correct_was": None}

        correct_action = self.get_correct_action(obs)

        reward = 1.00 if action.action_id == correct_action else -0.50

        self.cursor += 1
        done = self.cursor >= len(self.df)

        if done:
            next_obs = None
        else:
            next_obs = self.getObservation() 

        return next_obs, reward, done, {"correct_was": correct_action}

    def get_correct_action(self, obs):
        fields = [obs.age, obs.email, obs.phone, obs.city]
        
        # Robust count for missing fields
        missing_count = sum(
            1 for f in fields 
            if f is None or str(f).strip() == "" or str(f).lower() == "nan"
        )
        
        # 1. DROP (Priority 1)
        if (obs.age is not None and obs.age < 0) or not obs.name.strip() or obs.name.lower() == "unknown":
            return 3

        # 2. FLAG (Priority 2)
        disposable_domains = ["test.com", "example.com", "tempmail.org"]
        email_str = str(obs.email).lower() if obs.email else ""
        is_test_email = any(domain in email_str for domain in disposable_domains)

        if (obs.age and obs.age > 120) or is_test_email:
            return 4

        # 3. NORMALIZE (Priority 3)
        if (
            (obs.name != obs.name.strip()) or 
            (obs.email and obs.email != obs.email.lower()) or 
            (obs.phone and obs.phone != obs.phone.strip()) or
            (obs.city and obs.city.strip() != "" and obs.city != obs.city.strip())
        ):
            return 2

        # 4. IMPUTE (Priority 4)
        if missing_count == 1:
            return 1
                    
        return 0 # 5. KEEP