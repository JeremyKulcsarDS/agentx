from google.cloud import aiplatform
import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account
import aiohttp
import json
import os
from typing import Callable, List, Dict, Optional, Union, Any
import requests
from pydantic import BaseModel

from agentx.schema import Message, ToolCall, FunctionCall, GenerationConfig, Content

from agentx.vertexai_utils import transform_openai_tool_to_vertexai_tool

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

    if message.role == 'user':
        content = message.content
        return add_name(
                {
                    'role': message.role,
                    'parts': [
                        generative_models.Part.from_text(content.text)
                    ],
                },
                None
            )
    else:
        pass

    if message.role == 'model':
        return generative_models.Content.from_dict(
            {
                'role': message.role,
                'parts': [
                        {
                            'function_call':{
                                'name':message.content.tool_calls[0].function_call.name,
                                'args':{
                                    'fields': 'fields' # Condition in the Google protocol. Remove the key later.
                                } | json.loads(message.content.tool_calls[0].function_call.arguments)
                        }
                    }            
                ]
            } 
        )
    else:
        pass

    if message.role == 'tool':
        return {
            'role': 'function',
            'parts': [
                generative_models.Part.from_function_response(
                    name = message.content.tool_response.id, response = json.loads(
                        message.content.tool_response.content
                    )
                )
            ],
        }


class VertexAIClient():

    def __init__(self, generation_config:GenerationConfig):
        # Extract project ID from the JSON file
        with open(generation_config.path_to_google_service_account_json) as json_file:
            service_account_data = json.load(json_file)
            project_id = service_account_data['project_id']
            account_type = service_account_data['type']

        PROJECT_ID = project_id
        REGION = generation_config.region

        if account_type == "service_account":
            self.credentials = service_account.Credentials.from_service_account_file(
                generation_config.path_to_google_service_account_json,
                scopes=generation_config.google_application_credential_scope
            )
            self.auth_req = google.auth.transport.requests.Request()
            self.credentials.refresh(self.auth_req)

            aiplatform.init(project=PROJECT_ID, 
                      location=REGION, 
                      credentials=self.credentials ) 
        else:
            # Raise an error
            raise ValueError("This feature only works with Google Service Accounts at the moment")


    def generate(
            self,
            messages:List[Message],
            generation_config:GenerationConfig,
            reduce_function:Optional[Callable[[List[Message]], Message]]=None,
            output_model:BaseModel = None,
        ) -> Union[Message, List[Message], None]:

        # kwargs (to be adapted later)
        kw_args = {key:value for key, value in generation_config.model_dump().items() if value != None and key in VERTEXAI_API_KW}

        # Model
        kw_args['model'] = generation_config.model

        # =========================================================================================================================
        # =================================== WAITING FOR SCHEMA ISSUE TO BE FIXED TO BE USABLE ===================================

        # Creating dummy variables as the current tools are stored with a wrongful schema
        test_dict_geocoding = {
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

        test_dict_geodesic = {
            'name': 'xentropy--geodesic',
            'description': 'Calculate the earth surface distance between two latitude and longitude coordinate',
            'parameters': {
                'title': 'CoordinatePair',
                'type': 'object',
                'properties': {
                    'coordinate_0': {
                        'type': 'object',
                        'description': 'Coordinate',
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
                    },
                    'coordinate_1': {
                        'type': 'object',
                        'description': 'Coordinate',
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
                },
                'required': [
                    'coordinate_0',
                    'coordinate_1'
                ]
            }
        }

        # Once the schema issue is solved, this will be uncommented 
        if generation_config.tools is not None:
            kw_args['function_declaration'] = [
                generative_models.FunctionDeclaration(
                    **transform_openai_tool_to_vertexai_tool(tool)
                ) for tool in [test_dict_geocoding, test_dict_geodesic]
            ]

        if generation_config.tools is not None:
            # Define a tool that includes the above functions
            kw_args['tools'] = [
                generative_models.Tool(
                    #function_declarations=kw_args['tools'], # THIS MUST WORK
                    # These functions are just dummies. After the schema issue is solved, this will turn into what is seen above. =====
                    function_declarations = kw_args['function_declaration']
                )
            ]
        else:
            kw_args['tools'] = None

        # Initialise Gemini model
        model = generative_models.GenerativeModel(kw_args['model'])

        # =================================== WAITING FOR SCHEMA ISSUE TO BE FIXED TO BE USABLE ===================================
        # =========================================================================================================================

        _messages = []
        vertex_messages = [transform_message_vertexai(message) for message in messages]

        vertex_content = [generative_models.Content(**vertex_message) if type(vertex_message) is dict else vertex_message for vertex_message in vertex_messages]

        # Send a prompt for the first conversation turn that should invoke the function
        response = model.generate_content(contents = vertex_content, tools = kw_args['tools'])

        # If the response indicates a function call procedure
        if response.candidates[0].content.parts[0].function_call.args is not None:
            if len(response.candidates[0].content.parts[0].function_call.args) != 1 or 'fields' not in response.candidates[0].content.parts[0].function_call.args.keys():
                # {arg_name: arg_value} dictionary. 
                # Sometimes an argument can have multiple values, so it will become a dict of dict. {arg_name: {arg_1: arg_1_value, arg_2: arg_2_value}}
                args_dict = {
                    key: (
                        response.candidates[0].content.parts[0].function_call.args[key]
                        if isinstance(response.candidates[0].content.parts[0].function_call.args[key], str) # if string, then no nest in the response object
                        else {
                            subkey:
                            response.candidates[0].content.parts[0].function_call.args[key][subkey] for subkey in response.candidates[0].content.parts[0].function_call.args[key]
                        }
                    )
                    for key in response.candidates[0].content.parts[0].function_call.args
                }

                # This is due to a bug in the Google protocol
                # It only accepts customs arguments if the first field
                # is "fields". We add it and then remove it before calling the tool.
                if 'fields' in args_dict:
                    del args_dict['fields']

                # Dump it for the API call
                args_data = json.dumps(args_dict)

                tool_calls = [
                    ToolCall(
                        id=response.candidates[0].content.parts[0].function_call.name.lower(),
                        type="function",
                        function_call=FunctionCall(
                            name=response.candidates[0].content.parts[0].function_call.name.lower(), 
                            arguments=args_data
                        )
                    )
                ]

                content = Content(
                    tool_calls=tool_calls
                )
            else:
                pass
            
        # If no function call, directly return the answer
        else:
            content = Content(
                text=response.candidates[0].content.parts[0].text,
            )

        # Fit the message object
        _messages.append(Message(
                role=response.candidates[0].content.role,
                content=content,
            )
        )
                
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
