import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account
import aiohttp
import json
from typing import Coroutine, Callable, List, Dict, Optional, Union
import requests
from agentx.schema import Message, Content, ToolCall, Function, GenerationConfig


# ================================================================================
# UNDER DEVELOPMENT ==============================================================
# ================================================================================
def add_name(message:Dict, name:Optional[str]=None) -> Dict:
    """
    Adds a 'name' key-value pair to the given message dictionary.

    Args:
        message (Dict): The message dictionary to which the name should be added.
        name (Optional[str], optional): The name to be added. Defaults to None.

    Returns:
        Dict: The updated message dictionary with the 'name' key-value pair added.
    """
    if name is not None:
        message['name'] = name
    return message


def transform_message_vertexai(message:Message) -> Dict:
    """
    Transforms a message for the Vertex AI system.

    Args:
        message (Message): The message to transform.

    Returns:
        Dict: The transformed message dictionary.
    """
    if message.role == 'system' or message.role == 'user':
        content = message.content
        if content.files is None and content.urls is None and content.tool_calls is None and content.tool_response is None:
            return add_name(
                {
                    'role': 'user',
                    'content': content.text,
                },
                message.name
            )
    else:
        # DEVLOPMENT IS HERE
        pass

# ================================================================================
# END UNDER DEVELOPMENT ==========================================================
# ================================================================================


class VertexAIClient():

    def __init__(self, generation_config:GenerationConfig):
        self.credentials = service_account.Credentials.from_service_account_file(
            generation_config.path_to_google_service_account_json,
            scopes=generation_config.google_application_credential_scope
        )
        self.auth_req = google.auth.transport.requests.Request()
        self.credentials.refresh(self.auth_req)

    def generate(
            self,
            messages:List[Message],
            generation_config:GenerationConfig,
            reduce_function:Optional[Callable[[List[Message]], Message]]=None,
            output_model:Optional[str] = None,
        ) -> Message:
        if not self.credentials.valid:
            self.credentials.refresh(self.auth_req)
        
        tools = generation_config.tools
        if tools is not None:
            tools = [
                {
                    'functionDeclaration': tool
                } for tool in generation_config.tools
            ],

        contents = [transform_message_vertexai(message) for message in messages]

        print('here')
        print(contents)

        request_body = {
            "contents": contents,
            "tools": tools,
            "generationConfig": {
                "temperature": generation_config.temperature,
                "topP": generation_config.top_p,
                "topK": generation_config.top_k,
                "candidateCount": generation_config.n_candidates,
                "maxOutputTokens": generation_config.max_tokens,
                "stopSequences": generation_config.stop_sequences
            }
        }

        # Extract project ID from the JSON file
        with open(generation_config.path_to_google_service_account_json) as json_file:
            service_account_data = json.load(json_file)
            project_id = service_account_data['project_id']

        PROJECT_ID = project_id
        REGION = generation_config.region
        MODEL = generation_config.model

        print(REGION)
        print(PROJECT_ID)
        print(MODEL)

        print(f'https://{REGION}-aiplatform.googleapis.com/v1/{MODEL}:streamGenerateContent')

        response = requests.post(
            url = f'https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/{MODEL}:streamGenerateContent',
            json=request_body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": "Bearer " + self.credentials.token
            },
            timeout = generation_config.timeout
        )
        
        print(response)

        return response.json()
        
    async def a_generate(
            self, 
            messages:List[Message],
            generation_config:GenerationConfig,
            reduce_function:Optional[Callable[[List[Message]], Message]]=None
    )-> Union[Message, List[Message]]:
        if not self.credentials.valid:
            self.credentials.refresh(self.auth_req)
        
        tools = generation_config.tools
        if tools != None:
            tools = [
                {
                    'functionDeclaration': tool
                } for tool in generation_config.tools
            ],
        contents = [transform_message_vertexai(message) for message in messages]
        request_body = {
            "contents": contents,
            "tools": tools,
            "generationConfig": {
                "temperature": generation_config.temperature,
                "topP": generation_config.top_p,
                "topK": generation_config.top_k,
                "candidateCount": generation_config.n_candidates,
                "maxOutputTokens": generation_config.max_tokens,
                "stopSequences": generation_config.stop_sequences
            }
        }

        REGION = generation_config.region
        PROJECT_ID = generation_config.project_id
        MODEL = generation_config.model

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=f'https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/{MODEL}:streamGenerateContent',
                json=request_body,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": "Bearer " + self.credentials.token
                }
            ) as response:
                return await response.json()
