import uvicorn
from fastapi import FastAPI
from openenv_core.env_server import create_env_app
from server.kyc_environment import KYCEnv
from models.schema import KYCAction, KYCObservation

# This tells OpenEnv how to serve your KYC logic over the internet
env = KYCEnv()
app = create_env_app(env, KYCAction, KYCObservation)
def main():
    uvicorn.run(app, host = "0.0.0.0", port= 7860)

if  __name__ == "__main__":
    main()