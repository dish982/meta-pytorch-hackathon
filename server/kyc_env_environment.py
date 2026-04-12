# import pandas as pd
# import numpy as np
# from typing import Optional
# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from models import KYCAction, KYCObservation, StepResult

# class KYCEnv:
#     def __init__(self):
#         self.df = pd.read_csv("kyc_data.csv")
#         self.cursor = 0
#         self.step_count = 0
#         self.episode_id = 0
#         self.task_id = "missing_data"

#     async def reset_async(self) -> KYCObservation:
#         return self.reset()

#     async def step_async(self, action: KYCAction):
#         return self.step(action)

#     @property
#     def state(self) -> KYCObservation:
#         return self.getObservation()

#     def reset(self, task_id: str = "missing_data") -> StepResult:
#         self.cursor = 0
#         self.step_count = 0
#         self.episode_id += 1
#         self.task_id = task_id
        

#         obs = self.getObservation()

#         return StepResult(
#             observation=obs,
#             reward=0.5,  
#             done=False,
#             info={"task_id": task_id}
#         )

#     def getObservation(self) -> Optional[KYCObservation]:
#         if self.cursor < 0 or self.cursor >= len(self.df):
#             return KYCObservation(
#                 record_id=-1, name="END", age=0, email="",
#                 phone="", city="", step_count=self.step_count, episode_id=str(self.episode_id)
#             )
        
#         row = self.df.iloc[self.cursor]

#         raw_age = row.get('age')
#         age_val = int(raw_age) if pd.notna(raw_age) else None

#         return KYCObservation(
#             record_id=int(self.cursor),
#             name=str(row.get('name', '')) if pd.notna(row.get("name")) else "",
#             age=age_val,
#             email=str(row.get('email', '')) if pd.notna(row.get('email')) else None,
#             phone=str(row.get('phone', '')) if pd.notna(row.get('phone')) else None,
#             city=str(row.get('city', '')) if pd.notna(row.get('city')) else None,
#             step_count=int(self.step_count),
#             episode_id=str(self.episode_id)
#         )

#     def step(self, action: KYCAction):
#         obs = self.getObservation()
        
#         if obs is None or obs.record_id == -1:
#             reward = 0.5
#             reward = float(max(0.01, min(0.99, reward)))
#             return StepResult(
#                 observation=obs,
#                 reward=reward, 
#                 done=True,
#                 info={"msg": "Completed"}
#             )

#         correct_action = self.get_correct_action(obs)

#         # Guaranteed range: correct=0.6-0.9, wrong=0.1-0.4
#         if self.task_id == "missing_data":
#             if action.action_id == correct_action:
#                 base = 0.7 + (0.15 * np.random.rand())
#                 noise = np.random.uniform(-0.05, 0.05)
#                 reward = base + noise    
#             else:
#                 base = 0.2 + (0.15 * np.random.rand())
#                 noise = np.random.uniform(-0.05, 0.05)
#                 reward = base + noise

#         elif self.task_id == "format_check":
#             if action.action_id == correct_action:
#                 base = 0.7 + (0.15 * np.random.rand())
#                 noise = np.random.uniform(-0.05, 0.05)
#                 reward = base + noise    
#             else:
#                 base = 0.2 + (0.15 * np.random.rand())
#                 noise = np.random.uniform(-0.05, 0.05)
#                 reward = base + noise  

#         elif self.task_id == "compliance_audit":
#             if action.action_id == correct_action:
#                 base = 0.7 + (0.15 * np.random.rand())
#                 noise = np.random.uniform(-0.05, 0.05)
#                 reward = base + noise    
#             else:
#                 base = 0.2 + (0.15 * np.random.rand())
#                 noise = np.random.uniform(-0.05, 0.05)
#                 reward = base + noise

#         else:
#             reward = 0.5
        
        
#         reward = float(max(0.01, min(0.99, reward)))

#         self.cursor += 1
#         self.step_count += 1
#         done = self.cursor >= len(self.df)
#         next_obs = self.getObservation()
        

#         return StepResult(
#             observation=next_obs,
#             reward=reward,
#             done=done,
#             info={"correct_action": int(correct_action)}
#         )

#     def get_correct_action(self, obs):
#         fields = [obs.age, obs.email, obs.phone, obs.city]

#         # Robust count for missing fields
#         missing_count = sum(
#             1 for f in fields
#             if f is None or str(f).strip() == "" or str(f).lower() == "nan"
#         )

#         # 1. DROP (Priority 1)
#         if (obs.age is not None and obs.age < 0) or not obs.name.strip() or obs.name.lower() == "unknown":
#             return 3

#         # 2. FLAG (Priority 2)
#         disposable_domains = ["test.com", "example.com", "tempmail.org"]
#         email_str = str(obs.email).lower() if obs.email else ""
#         is_test_email = any(domain in email_str for domain in disposable_domains)

#         if (obs.age and obs.age > 120) or is_test_email:
#             return 4

#         # 3. NORMALIZE (Priority 3)
#         if (
#             (obs.name != obs.name.strip()) or
#             (obs.email and obs.email != obs.email.lower()) or
#             (obs.phone and obs.phone != obs.phone.strip()) or
#             (obs.city and obs.city.strip() != "" and obs.city != obs.city.strip())
#         ):
#             return 2

#         # 4. IMPUTE (Priority 4)
#         if missing_count == 1:
#             return 1

#         return 0  # 5. KEEP

#     def close(self):
#         pass
import pandas as pd
import numpy as np
from typing import Optional
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import KYCAction, KYCObservation, StepResult


class KYCEnv:
    def __init__(self):
        self.df = pd.read_csv("kyc_data.csv")
        self.cursor = 0
        self.step_count = 0
        self.episode_id = 0
        self.task_id = "missing_data"

    async def reset_async(self) -> KYCObservation:
        return self.reset()

    async def step_async(self, action: KYCAction):
        return self.step(action)

    @property
    def state(self) -> KYCObservation:
        return self.getObservation()

    def reset(self, task_id: str = "missing_data") -> StepResult:
        self.cursor = 0
        self.step_count = 0
        self.episode_id += 1
        self.task_id = task_id

        obs = self.getObservation()

        reward = 0.5001  # SAFE (never 0 or 1)

        return StepResult(
            observation=obs,
            reward=reward,
            done=False,
            info={"task_id": task_id}
        )

    def getObservation(self) -> Optional[KYCObservation]:
        if self.cursor < 0 or self.cursor >= len(self.df):
            return KYCObservation(
                record_id=-1,
                name="END",
                age=0,
                email="",
                phone="",
                city="",
                step_count=self.step_count,
                episode_id=str(self.episode_id)
            )

        row = self.df.iloc[self.cursor]

        raw_age = row.get('age')
        age_val = int(raw_age) if pd.notna(raw_age) else None

        return KYCObservation(
            record_id=int(self.cursor),
            name=str(row.get('name', '')) if pd.notna(row.get("name")) else "",
            age=age_val,
            email=str(row.get('email', '')) if pd.notna(row.get('email')) else None,
            phone=str(row.get('phone', '')) if pd.notna(row.get('phone')) else None,
            city=str(row.get('city', '')) if pd.notna(row.get('city')) else None,
            step_count=int(self.step_count),
            episode_id=str(self.episode_id)
        )

    def step(self, action: KYCAction):
        obs = self.getObservation()

        
        if obs is None or obs.record_id == -1:
            reward = 0.5001
            reward = float(np.clip(reward, 1e-4, 1 - 1e-4))

            return StepResult(
                observation=obs,
                reward=reward,
                done=True,
                info={"msg": "Completed"}
            )

        correct_action = self.get_correct_action(obs)

        
        if self.task_id in ["missing_data", "format_check", "compliance_audit"]:
            if action.action_id == correct_action:
                reward = 0.7 + (0.15 * np.random.rand())  # 0.70–0.85
            else:
                reward = 0.2 + (0.15 * np.random.rand())  # 0.20–0.35
        else:
            reward = 0.5

        
        reward = float(np.clip(reward, 1e-4, 1 - 1e-4))

        self.cursor += 1
        self.step_count += 1

        done = self.cursor >= len(self.df)
        next_obs = self.getObservation()

        return StepResult(
            observation=next_obs,
            reward=reward,
            done=done,
            info={"correct_action": int(correct_action)}
        )

    def get_correct_action(self, obs):
        fields = [obs.age, obs.email, obs.phone, obs.city]

        missing_count = sum(
            1 for f in fields
            if f is None or str(f).strip() == "" or str(f).lower() == "nan"
        )

        # DROP
        if (obs.age is not None and obs.age < 0) or not obs.name.strip() or obs.name.lower() == "unknown":
            return 3

        # FLAG
        disposable_domains = ["test.com", "example.com", "tempmail.org"]
        email_str = str(obs.email).lower() if obs.email else ""
        is_test_email = any(domain in email_str for domain in disposable_domains)

        if (obs.age and obs.age > 120) or is_test_email:
            return 4

        # NORMALIZE
        if (
            (obs.name != obs.name.strip()) or
            (obs.email and obs.email != obs.email.lower()) or
            (obs.phone and obs.phone != obs.phone.strip()) or
            (obs.city and obs.city.strip() != "" and obs.city != obs.city.strip())
        ):
            return 2

        # IMPUTE
        if missing_count == 1:
            return 1

        return 0

    def close(self):
        pass