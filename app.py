import uvicorn
from fastapi import FastAPI
# from openenv.core.env import EnvironmentServer
import openenv_core.env_server as env_server
from server.kyc_env_environment import KYCEnv
from models import KYCAction, KYCObservation

# This tells OpenEnv how to serve your KYC logic over the internet
env = KYCEnv()
app = env_server.create_app(KYCEnv, KYCAction, KYCObservation)
def main():
    uvicorn.run(app, host = "0.0.0.0", port= 8000)

if  __name__ == "__main__":
    main()