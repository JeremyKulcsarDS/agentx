import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account
import aiohttp
import json
from typing import Callable, List, Dict, Optional, Union, Any
import requests
from pydantic import BaseModel
from agentx.schema import Message, ToolCall, FunctionCall, GenerationConfig, Content
from agentx.vertexai_utils import transform_openai_tool_to_vertexai_tool

import vertexai
from vertexai import generative_models

VERTEXAI_API_KW = [
    'model',
    'frequency_penalty',
    'logit_bias',
    'max_tokens',
    'n',
    'presence_penalty',
    'response_format',
    'seed',
    'stop',
    'temperature',
    'tool_choice',
    'tools',
    'top_p'
]
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
    if message.role == 'system':
        content = message.content
        return add_name(
            {
                'role': message.role,
                'parts': [{'text':content.text}],
                },
            None
        )
    
    if message.role == 'user':
        content = message.content
        if content.files is None and content.urls is None and content.tool_calls is None and content.tool_response is None:
            return add_name(
                {
                    'role': message.role,
                    'parts': [{'text':content.text}],
                },
                message.name
            )
    else:
        pass

    if message.role == 'model':
        tool_calls = message.content.tool_calls
        if tool_calls is not None:
            print("tool_content:", content)
            _tool_calls = []
            for tool_call in tool_calls:
                _tool_calls.append({
                    'id': tool_call.id,
                    'type': tool_call.type,
                    'function': {
                        'name': tool_call.function_call.name,
                        'arguments': tool_call.function_call.arguments
                    }
                })
            return add_name(
                {
                    'role': "function",
                    'parts': [{'function_call':_tool_calls}],
                },
                message.name
            )
        
        content = message.content
        if content.files is None and content.urls is None and content.tool_calls is None and content.tool_response is None:
            return add_name(
                {
                    'role': message.role,
                    'parts': [{'text':content.text}],
                },
                None # Keep the same code structure, since Vertex AI API would return a JSON decode error if a name is left
            )
    else:
        pass


""" def validate_function_call(tool_call:ChatCompletionMessageToolCall, config:GenerationConfig) -> bool: # adapt to Vertex AI
    if tool_call.function.name.lower() not in config.tools.keys():
        return False
    try:
        arguments = json.loads(tool_call.function.arguments)
        schema = config.tools.get(tool_call.function.name.lower()).parameters
        validate(arguments, schema)
        return True
    except:
        return False """


def replace_key(dictionary: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    """
    Recursively replaces all occurrences of the old_key with the new_key in a dictionary.

    Args:
        dictionary (Dict[str, Any]): The input dictionary to modify.
        old_key (str): The key to replace.
        new_key (str): The new key to use.

    Returns:
        Dict[str, Any]: The modified dictionary with replaced keys.
    """
    new_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            new_dict[key] = replace_key(value, old_key, new_key)  # Recursively process nested dictionaries
        elif isinstance(value, list):
            new_dict[key] = [replace_key(item, old_key, new_key) if isinstance(item, dict) else item for item in value]
            # Recursively process nested dictionaries within lists
        else:
            new_dict[key] = value
    if old_key in new_dict:
        new_dict[new_key] = new_dict.pop(old_key)  # Replace the old_key with the new_key
    return new_dict


class VertexAIClient():

    def __init__(self, generation_config:GenerationConfig):
        self.credentials = service_account.Credentials.from_service_account_file(
            generation_config.path_to_google_service_account_json,
            scopes=generation_config.google_application_credential_scope
        )
        self.auth_req = google.auth.transport.requests.Request()
        self.credentials.refresh(self.auth_req)

        # Extract project ID from the JSON file
        with open(generation_config.path_to_google_service_account_json) as json_file:
            service_account_data = json.load(json_file)
            project_id = service_account_data['project_id']

        PROJECT_ID = project_id
        REGION = generation_config.region
        MODEL = generation_config.model

        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = MODEL


    def generate(
            self,
            messages:List[Message],
            generation_config:GenerationConfig,
            reduce_function:Optional[Callable[[List[Message]], Message]]=None,
            output_model:BaseModel = None,
        ) -> Union[Message, List[Message], None]:

        # kwargs
        kw_args = {key:value for key, value in generation_config.model_dump().items() if value != None and key in VERTEXAI_API_KW}

        # Check schema constraints
        if output_model != None:
            messages.append(
                Message(
                    role='user',
                    content=Content(
                        text='You must return a JSON object according to this json schema. {schema}'.format(
                            schema = output_model.model_json_schema() 
                        )
                    )
                )
            )
            kw_args['response_format'] = {'type':'json_object'}

        # Creds management
        if not self.credentials.valid:
            self.credentials.refresh(self.auth_req)

        # Model
        if generation_config.api_type == 'vertexai':
            kw_args['model'] = generation_config.model


        test_dict = {
            'name': 'xentropy--geocoding',
            'description': 'Retrieve the latitude and longitude given an address using the highly accurate Google Map API.',
            'parameters': {
                'title': 'Address',
                'type': 'object',
                'properties': {
                'address': {
                    'title': 'Address',
                    'type': 'string'
                }
                },
                'required': [
                'address'
                ]
            }
        }

        generative_models.FunctionDeclaration(**transform_openai_tool_to_vertexai_tool(test_dict))
        print("xentropy--geocoding transformed to vertex AI schema successfully")


        test_dict_2 = {
            'name': 'xentropy--geodesic',
            'description': 'Calculate the earth surface distance between two latitude and longitude coordinate',
            'parameters': {
                'title': 'CoordinatePair',
                'type': 'object',
                'properties': {
                'coordinate_0': {
                    '$ref': '#/definitions/Coordinate'
                },
                'coordinate_1': {
                    '$ref': '#/definitions/Coordinate'
                }
                },
                'required': [
                'coordinate_0',
                'coordinate_1'
                ],
                'definitions': {
                'Coordinate': {
                    'title': 'Coordinate',
                    'type': 'object',
                    'properties': {
                    'latitude': {
                        'title': 'Latitude',
                        'type': 'number'
                    },
                    'longitude': {
                        'title': 'Longitude',
                        'type': 'number'
                    }
                    },
                    'required': [
                    'latitude',
                    'longitude'
                    ]
                }
                }
            }
            }

        generative_models.FunctionDeclaration(**transform_openai_tool_to_vertexai_tool(test_dict_2))
        print("xentropy--geodesic transformed to vertex AI schema successfully")

        """ # Tools
        if generation_config.tools != None:
            kw_args['tools'] = [
                generative_models.FunctionDeclaration(
                    **tool.model_dump()
                ) for tool in generation_config.tools.values()
            ] """
        

        # Tools
        if generation_config.tools != None:
            kw_args['tools'] = [
                generative_models.FunctionDeclaration(
                    **replace_key(tool.model_dump(), "title", "description")
                ) for tool in generation_config.tools.values()
            ]
        

        print(generation_config.tools)
        print(kw_args['tools'])


        # Specify a function declaration and parameters for an API request
        get_product_info_func = generative_models.FunctionDeclaration(
            name="get_product_sku",
            description="Get the SKU for a product",
            # Function parameters are specified in OpenAPI JSON schema format
            parameters={
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Product name"}
                },
            },
        )

        # Specify another function declaration and parameters for an API request
        get_store_location_func = generative_models.FunctionDeclaration(
            name="get_store_location",
            description="Get the location of the closest store",
            # Function parameters are specified in OpenAPI JSON schema format
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string", "description": "Location"}},
            },
        )
        
        print(get_store_location_func)

        # Define a tool that includes the above functions
        retail_tool = generative_models.Tool(
            function_declarations=[
                get_product_info_func,
                get_store_location_func,
            ],
        )

        # Define the user's prompt in a Content object that we can reuse in model calls
        user_prompt_content = generative_models.Content(
            role="user",
            parts=[
                generative_models.Part.from_text("test"),
            ],
        )

        # Initialise Gemini model
        model = generative_models.GenerativeModel(self.model, tools=[retail_tool])

        # Start a chat session
        chat = model.start_chat()
        
        
        
        
        
        _messages = []

        tool_calls = None

        for num_retry in range(generation_config.max_retries):

















            contents = [transform_message_vertexai(message) for message in messages]

            request_body = {
                "contents": contents,
                "tools": tool_calls,
                "generationConfig": {
                    #"temperature": generation_config.temperature,
                    #"topP": generation_config.top_p,
                    #"topK": generation_config.top_k,
                    "candidateCount": generation_config.n_candidates,
                    #"maxOutputTokens": generation_config.max_tokens,
                    #"stopSequences": generation_config.stop_sequences
                }
            }

            response = requests.post(
                url = f'https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/{MODEL}:streamGenerateContent',
                json=request_body,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": "Bearer " + self.credentials.token
                },
                timeout = generation_config.timeout
            )



            # generated_messages = [choice.message for choice in response.choices] adapt to vertex AI

            for message in generated_messages:
                if message.tool_calls != None:
                    tool_calls = [
                        ToolCall(
                            id=tool_call.id,
                            type=tool_call.type,
                            function_call=FunctionCall(
                                name=tool_call.function.name.lower(), 
                                arguments=tool_call.function.arguments
                            )
                        ) for tool_call in message.tool_calls # if validate_function_call(tool_call, generation_config)
                    ]
                    if tool_calls == []:
                        continue

                    content = Content(
                        tool_calls=tool_calls
                    )
                else:
                    content = Content(
                        text=message.content,
                    )

                if output_model and content.text != None:
                    try:
                        output_model.model_validate_json(content.text)
                        _messages.append(
                            Message(
                                role='assistant',
                                content=content,
                            )
                        )
                    except:
                        pass
                else:
                    _messages.append(
                        Message(
                            role='assistant',
                            content=content,
                        )
                    )

            if len(_messages) >= generation_config.n_candidates:
                _messages = _messages[:generation_config.n_candidates]
                break

        if len(_messages) == 0:
            return None
        if reduce_function:
            return reduce_function(_messages)
        else:
            return _messages

        """         tools = generation_config.tools
        if tools is not None:
            tool_content = [
                {
                    "name": tool,
                } for tool in generation_config.tools
            ]
            tools = [
                {
                    'functionCall': 
                        tool_content
                }
            ] """

        print("message:", messages)

        contents = [transform_message_vertexai(message) for message in messages]

        print("contents:", contents)

        request_body = {
            "contents": contents,
            "tools": tools,
            "generationConfig": {
                #"temperature": generation_config.temperature,
                #"topP": generation_config.top_p,
                #"topK": generation_config.top_k,
                "candidateCount": generation_config.n_candidates,
                #"maxOutputTokens": generation_config.max_tokens,
                #"stopSequences": generation_config.stop_sequences
            }
        }

        print("request_body:", request_body)

        # Extract project ID from the JSON file
        with open(generation_config.path_to_google_service_account_json) as json_file:
            service_account_data = json.load(json_file)
            project_id = service_account_data['project_id']

        PROJECT_ID = project_id
        REGION = generation_config.region
        MODEL = generation_config.model

        response = requests.post(
            url = f'https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/{MODEL}:streamGenerateContent',
            json=request_body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": "Bearer " + self.credentials.token
            },
            timeout = generation_config.timeout
        )

        print("response:",response.json())

        # relevant for vertex AI?
        _messages = []

        try:

            if 'error' in response.json()[0]:
                if response.json()[0]['error']['code'] == 429:
                    print(response.json()[0]['error']['message'])
                    return None
            
            if response.json()[0]['candidates'][0]['content']['role'] == "model" and 'functionCall' in response.json()[0]['candidates'][0]['content']['parts'][0]:
            
                role = response.json()[0]['candidates'][0]['content']['role']

                _messages.append(Message(role=role, content=Content(function_call=response.json()[0]['candidates'][0]['content']['parts'][0]['functionCall'])))

            else:

                role = response.json()[0]['candidates'][0]['content']['role']
                text = response.json()[0]['candidates'][0]['content']['parts'][0]['text']

                _messages.append(Message(role=role, content=Content(text=text)))

        except json.JSONDecodeError as e:
            # Handle JSON decoding error
            print("Error decoding JSON from the Vertex AI POST request response:", e)

        except ValueError as e:
            # Handle other value-related errors
            print("ValueError from the Vertex AI POST request response:", e)

        except Exception as e:
            # Handle other exceptions
            print("Exception from the Vertex AI POST request response:", e)
            raise e
        
        print("return:", _messages)
        
        if len(_messages) == 0:
            return None
        elif len(_messages) == 1:
            return _messages[0]
        
        if reduce_function:
            return reduce_function(_messages)
        else:
            return _messages
        

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
