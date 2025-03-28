# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
from typing import Optional, List
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Define a custom LLM class by subclassing LangChain's LLM.
class CustomLLM(LLM):
    # Declare the API endpoint and bearer token as fields.
    url: str = Field(..., description="Custom LLM API endpoint")
    bearer_token: str = Field(..., description="Bearer token for API authorization")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        # Adjust the payload as needed by your API.
        payload = {
            "prompt": prompt,
            "max_tokens": 150
        }
        response = requests.post(self.url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        # Expecting the API to return JSON with a "generated_text" field.
        return result.get("generated_text", "")

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

# Define the request and response models.
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# Create the FastAPI app.
app = FastAPI()

# Instantiate the custom LLM with your API endpoint and bearer token.
custom_llm = CustomLLM(
    url="https://api.yourcustomllm.com/generate",  # Replace with your API endpoint.
    bearer_token="your_custom_llm_bearer_token"      # Replace with your bearer token.
)

# Create a prompt template to format incoming questions.
prompt_template = PromptTemplate.from_template("Question: {question}\nAnswer:")

# Combine the prompt template and custom LLM into a chain.
chain = LLMChain(llm=custom_llm, prompt=prompt_template)

# Define an endpoint that accepts a question and returns the generated answer.
@app.post("/ask", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    try:
        # Call the chain with the provided question.
        result = chain.call({"question": query.question})
        # Depending on the LangChain version, result can be a dict or a string.
        answer = result.get("text") if isinstance(result, dict) else result
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the app:
# uvicorn main:app --reload
