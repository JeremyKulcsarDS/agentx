import asyncio
import json
from typing import Dict, List, Callable, Optional, Union
from pydantic import BaseModel
from xentropy.schema import Message, Content, ToolResponse, GenerationConfig, File
import xentropy.oai_client
import xentropy.vertexai_client
import xentropy.bedrock_client
from xentropy.utils import get_coroutine

class Agent():
    """
    Base class for all agents
    
    Attributes:
        name (str): The name of the agent.
        system_prompt (str): The system prompt to be used during generation.
        generation_config (GenerationConfig): The configuration for text generation.
        function_map (Dict[str, Callable]): A dictionary mapping function names to their corresponding callable objects.
        terminate_function (Callable[[List[Message]], bool]): A function that determines the termination criteria for generation.
        reduce_function (Callable[[List[Message]], Message]): A function that reduces a list of messages into a single message. Useful for running self-consistency algorithms to improve performance.

    Methods:
        __init__(self, name:str, system_prompt:str=None, generation_config:GenerationConfig=None,
                 function_map:Dict[str, Callable]=None, terminate_function:Callable[[List[Message]], bool]=None,
                 reduce_function:Callable[[List[Message]], Message]=None)
            Initializes a new instance of the Agent class.
        
        generate_response(self, messages:List[Message], output_model:Optional[BaseModel]=None) -> Message
            Generates a response to the given messages based on the generation config.
        
        a_generate_response(self, messages:List[Message], output_model:Optional[BaseModel]=None)
            Asynchronously generates a response to the given messages based on the generation config.
        
        automatic_prompt_optimization(self, dataset:List[Dict], loss_function:Callable[..., float])
            Fits a system prompt to a given dataset using Automatic Prompt Optimization (APO).
    """

    def __init__(
        self,
        name:str,
        system_prompt:str=None,
        generation_config:GenerationConfig=None,
        function_map:Dict[str, Callable]=None,
        termination_function:Callable[[List[Message]], bool]=lambda x: False,
        reduce_function:Callable[[List[Message]], Message]=lambda x: x[-1],
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.generation_config = generation_config
        self.function_map = function_map
        self.termination_function = termination_function
        self.reduce_function = reduce_function

        if self.generation_config.api_type in ['openai', 'fastchat', 'azure']:
            self.client = xentropy.oai_client.OAIClient(
                generation_config=self.generation_config
            )

        if self.generation_config.api_type == 'vertexai':
            self.client = xentropy.vertexai_client.VertexAIClient(
                generation_config=generation_config
            )
        
        if self.generation_config.api_type == 'bedrock':
            self.client = xentropy.bedrock_client.BedrockClient()

    def generate_response(self, messages:List[Message], output_model:Optional[BaseModel]=None) -> Union[None, Message, List[Message]]:
        """
        Generate a response to the given messages based on the generation config
        """
        # determine termination criteria
        if self.termination_function(messages):
            return None
        
        # tool calls
        tool_calls = messages[-1].content.tool_calls
        if tool_calls != None:
            tool_responses = []
            multimodal_responses = []
            for tool_call in tool_calls:
                if tool_call.type == 'function':
                    function = self.function_map.get(tool_call.function_call.name)
                    if function != None:
                        response = function(**json.loads(tool_call.function_call.arguments))
                        # the response is assumed to be a json string
                        # if the key 'files' or the key 'url' is present, an additional message is generated
                        tool_responses.append(
                            Message(
                                role='tool',
                                content = Content(
                                    tool_response = ToolResponse(
                                        id=tool_call.id,
                                        name=tool_call.function_call.name,
                                        content=response
                                    )
                                ),
                                name=tool_call.function_call.name,
                            ),
                        )

                        deserialised_response = json.loads(response)
                        files = deserialised_response.get('files')
                        if files != None:
                            files = [File(**file) for file in files]
                        urls = deserialised_response.get('url')
                        if files != None or urls != None:
                            multimodal_responses.append(
                                Message(
                                    role='user',
                                    content = Content(
                                        files=files,
                                        url=urls
                                    ),
                                    name=self.name,
                                )
                            )
            return tool_responses + multimodal_responses

        # determine the api_type
        if self.generation_config.api_type in ['openai', 'azure', 'fastchat']:
            message = self.client.generate(
                messages=messages,
                generation_config=self.generation_config,
                reduce_function=self.reduce_function,
                output_model=output_model,
            )
        
        if self.generation_config.api_type == 'vertexai':
            # call the vertex api
            message = self.client.generate(
                messages=messages,
                generation_config=self.generation_config,
                reduce_function=self.reduce_function,
                output_model=output_model,
            )

        if self.generation_config.api_type == 'bedrock':
            # call the bedrock api
            message = self.client.generate(
                messages=messages,
                generation_config=self.generation_config,
                reduce_function=self.reduce_function,
                output_model=output_model,
            )

        message.name = self.name
        return message

    async def a_generate_response(self, messages:List[Message], output_model:Optional[BaseModel]=None) -> Union[None, Message,List[Message]]:
        """
        Async version of generate_response
        """
        # determine termination criteria
        if self.termination_function(messages):
            return None

        # tool calls
        tool_calls = messages[-1].content.tool_calls
        if tool_calls != None:
            tasks = [
                get_coroutine(
                    self.function_map.get(tool_call.function_call.name),
                    **json.loads(tool_call.function_call.arguments)
                 ) for tool_call in tool_calls if tool_call.type == 'function'
            ]

            responses = await asyncio.gather(*tasks)

            tool_responses = []
            multimodal_responses = []
            for response, tool_call in zip(responses, tool_calls):
                # the response is assumed to be a json string
                # if the key 'files' or the key 'url' is present, an additional message is generated
                tool_responses.append(
                    Message(
                        role='tool',
                        content = Content(
                            tool_response = ToolResponse(
                                id=tool_call.id,
                                name=tool_call.function_call.name,
                                content=response
                            )
                        ),
                        name=tool_call.function_call.name,
                    ),
                )

                deserialised_response = json.loads(response)
                files = deserialised_response.get('files')
                if files != None:
                    files = [File(**file) for file in files]
                urls = deserialised_response.get('url')
                if files != None or urls != None:
                    multimodal_responses.append(
                        Message(
                            role='user',
                            content = Content(
                                files=files,
                                url=urls
                            ),
                            name=self.name,
                        )
                    )
            return tool_responses + multimodal_responses
    
        # determine the api_type
        if self.generation_config.api_type in ['openai', 'azure', 'fastchat']:
            message = await self.client.a_generate(
                messages=messages,
                generation_config=self.generation_config,
                reduce_function=self.reduce_function,
                output_model=output_model,
            )
        
        if self.generation_config.api_type == 'vertexai':
            # call the vertex api
            message = await self.client.a_generate(
                messages=messages,
                generation_config=self.generation_config,
                reduce_function=self.reduce_function,
                output_model=output_model,
            )

        if self.generation_config.api_type == 'bedrock':
            # call the bedrock api
            message = await self.client.a_generate(
                messages=messages,
                generation_config=self.generation_config,
                reduce_function=self.reduce_function,
                output_model=output_model,
            )
            
        message.name = self.name
        return message