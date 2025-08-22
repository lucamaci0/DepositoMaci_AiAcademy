import os
import json
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, stop_after_delay


load_dotenv("C:/Users/LH668YN/OneDrive - EY/Desktop/AiAcademy/DepositoMaci_AiAcademy/.env")  #C:/Users/LH668YN/OneDrive - EY/Desktop/
subscription_key = os.getenv("SUBSCRIPTION_KEY") or ""
azure_endpoint = os.getenv("AZURE_ENDPOINT") or ""
api_version = os.getenv("API_VERSION") or ""
deployment_name = os.getenv("DEPLOYMENT_NAME") or ""

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=azure_endpoint,
    api_key=subscription_key,
)

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_delay(15))
def call_model():
    return  client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {
                        "role": "user",
                        "content": "I am going to Paris, what should I see?",
                    }
                ],
                max_completion_tokens=300,
                temperature=0.0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                model=deployment_name
            )

response = call_model()
print(response.choices[0].message.content)
