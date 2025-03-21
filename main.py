from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import requests
from typing import List, Optional
from langchain.llms.base import LLM
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool

app = FastAPI()

# ---------------------------
# Request Body Model
# ---------------------------
class PromptInput(BaseModel):
    prompt: str

# ---------------------------
# Custom LLM using a Custom API
# ---------------------------
class CustomLLM(LLM):
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"prompt": prompt}
        response = requests.post(self.api_url, json=payload, headers=headers)
        if response.status_code != 200:
            raise Exception(f"LLM API call failed with status code {response.status_code}")
        data = response.json()
        return data.get("result", "")

# ---------------------------
# Example Custom Tool for the Agent
# ---------------------------
class ReverseTool(BaseTool):
    name = "reverse_text"
    description = "Reverses the given input text."

    def _run(self, text: str) -> str:
        return text[::-1]

    async def _arun(self, text: str) -> str:
        return text[::-1]

# ---------------------------
# Dependency to extract API token from request header
# ---------------------------
def get_api_key(x_openapi_token: str = Header(..., alias="X-OpenAPI-Token")):
    return x_openapi_token

# ---------------------------
# FastAPI Endpoint
# ---------------------------
@app.post("/chat")
def chat(input_data: PromptInput, api_key: str = Depends(get_api_key)):
    try:
        # Define your custom LLM API endpoint (update with your URL)
        custom_api_url = "http://your-custom-llm-api/endpoint"
        
        # Initialize our custom LLM with the provided API token
        custom_llm = CustomLLM(api_url=custom_api_url, api_key=api_key)
        
        # Set up a list of tools available to the agent (here a simple text reversal tool)
        tools = [ReverseTool()]
        
        # Initialize an agent with our custom LLM and tools.
        # Here we use the ZERO_SHOT_REACT_DESCRIPTION agent type.
        agent = initialize_agent(tools, custom_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        
        # Run the agent with the input prompt.
        agent_response = agent.run(input_data.prompt)
        
        return {"response": agent_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))