import boto3
import botocore
import json
from typing import Dict, Any
from botocore.exceptions import ClientError
from langchain_community.chat_models import BedrockChat
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric


class BedrockLLMClient:
    def __init__(self, region_name: str):
        """
        Initialize the AWS Bedrock client. Use `aws configure` first to add AWS_ACCESS_KEY_ID and AWS_SECRETE_ACCESS_KEY to the configuration 
        """
        self.client = boto3.client(
            'bedrock',
            region_name=region_name,
        )
    
    def send_prompt(self, model_id: str, prompt: str, parameters: Dict[str, Any] = None) -> str:
        """
        Send a prompt to the specified LLM model and retrieve the output message.
        
        :param model_id: The ID of the LLM model on AWS Bedrock.
        :param prompt: The input text prompt for the LLM.
        :param parameters: Additional parameters to control the LLM’s response.
        :return: The output text message from the LLM.
        """
        if parameters is None:
            parameters = {}
        
        try:
            # Construct the payload for the model inference request
            payload = {
                'modelId': model_id,
                'input': {
                    'prompt': prompt,
                    **parameters  # Optional parameters for tuning the response
                }
            }
            # Call Bedrock's `InvokeModel` API to send the prompt
            response = self.client.invoke_model(
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            # Parse the JSON response from Bedrock
            response_body = json.loads(response['Body'].read().decode('utf-8'))
            return response_body.get('output', 'No output received')
        
        except Exception as e:
            print(f"An error occurred while sending the prompt: {e}")
            return str(e)
        

class AWSBedrock(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"


def test_answer_relevancy():
    answer_relevancy_metric = AnswerRelevancyMetric(model=AWSBedrock, threshold=0.5)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output of your LLM application
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
    )
    assert_test(test_case, [answer_relevancy_metric])


# Example usage:
if __name__ == "__main__":
    # Set up AWS Bedrock client parameters
    region = 'us-west-2'
    model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
    prompt_data = "What is the capital of France?"

    # # Initialize the Bedrock LLM Client
    # bedrock_client = BedrockLLMClient(region_name=region)

    # # Send a prompt and get the output
    # output = bedrock_client.send_prompt(model_id=model_id, prompt=prompt)
    # print("Model Output:", output)
             
    # boto3_bedrock = boto3.client('bedrock')
    # print([models['modelId'] for models in boto3_bedrock.list_foundation_models()['modelSummaries']])

    bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)

    body = json.dumps({
        "prompt": f"\n\nHuman:<${prompt_data}>\n\nAnswer:",
        "max_tokens_to_sample": 300,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman:"]
    })
    
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    print(response_body.get("results")[0].get("outputText"))

    # # Replace these with real values
    # custom_model = BedrockChat(
    #     credentials_profile_name= "default",
    #     region_name="us-west-2",
    #     endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",
    #     model_id="anthropic.claude-3-haiku-20240307-v1:0",
    #     model_kwargs={"temperature": 0.4},
    # )

    # aws_bedrock = AWSBedrock(model=custom_model)
    # print(aws_bedrock.generate("Write me a joke"))
    