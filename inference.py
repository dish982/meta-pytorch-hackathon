import asyncio
import os
import textwrap
from typing import List, Optional
from openai import OpenAI

from server.kyc_env_environment import KYCEnv
from models import KYCAction

# ---------------- CONFIG ---------------- #
API_KEY = os.environ["API_KEY"]
if not API_KEY:
    raise ValueError("HF_TOKEN environment variable is required")

API_BASE_URL = os.environ.get["API_BASE_URL"]
MODEL_NAME = os.environ("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

if not API_BASE_URL or not API_KEY:
    raise ValueError("API_BASE_URL and API_KEY must be set by environment")

TASK_NAME = "kyc-audit"
BENCHMARK = "disha-kyc-v1"

# ---------------- LOGGING ---------------- #
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    done_val = str(done).lower()
    error_val = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True
    )

def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True
    )

# ---------------- LLM LOGIC ---------------- #
def get_model_action(client: OpenAI, obs) -> int:
    system_prompt = """
You are a strict KYC Data Auditor.

Follow this PRIORITY ORDER EXACTLY: 3 > 4 > 2 > 1 > 0

3 (DROP):
- Name is empty
- Name is 'Unknown'
- Age is negative

4 (FLAG):
- Age > 120
- Email contains test.com, example.com, or tempmail.org

2 (NORMALIZE):
- Name, Phone, or City has leading/trailing spaces
- Email contains uppercase letters

1 (IMPUTE):
- Exactly one field is missing (Age, Email, Phone, or City)

0 (KEEP):
- Record is 100 percent clean

Respond ONLY with one number: 0, 1, 2, 3, or 4.
"""

    user_prompt = f"""
AUDIT THIS RECORD:
Name: {obs.name}
Age: {obs.age}
Email: {obs.email}
Phone: {obs.phone}
City: {obs.city}

What is the Action ID?
"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": textwrap.dedent(system_prompt)},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=5,
        )

        text = (completion.choices[0].message.content or "").strip()

        # Extract digit safely
        digit = ''.join(filter(str.isdigit, text))
        return int(digit[0]) if digit else 4

    except Exception:
        # fallback → FLAG
        return 4


# ---------------- MAIN LOOP ---------------- #
async def main():

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = KYCEnv()

    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        done = False
        step = 1

        while not done:
            # LLM decision
            action_id = get_model_action(client, obs)
            action = KYCAction(action_id=action_id)

            # Env step
            obs, reward, done, info = env.step(action)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=str(action_id),
                reward=reward,
                done=done,
                error=None
            )

            if done:
                break

            step += 1

        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        success = (steps_taken >= len(env.df)) and (avg_reward > 0)

    except Exception as e:
        log_step(step=steps_taken, action="error", reward=0.0, done=True, error=str(e))

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            rewards=rewards
        )

        if hasattr(env, "close"):
            result = env.close()
            if asyncio.iscoroutine(result):
                await result


if __name__ == "__main__":
    asyncio.run(main())