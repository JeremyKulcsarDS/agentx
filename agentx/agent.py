import asyncio
from copy import deepcopy
import json
from typing import List, Callable, Optional, Union, TypeVar
from pydantic import BaseModel
from agentx.schema import Message, Content, ToolResponse, GenerationConfig, File, Function
from agentx.tool import Tool
import agentx.oai_client
import agentx.vertexai_client
import agentx.bedrock_client

OutputType = TypeVar('OutputType')
class Agent():
    """
    Base class for all agents
    
    Attributes:
        name (str): The name of the agent.
        generation_config (GenerationConfig): The configuration for text generation.
        system_prompt (Optional[str]): The system prompt to be used during generation.
        tools (Optional[List[Tool]]): A dictionary of tools that the agent can use.
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
        generation_config:GenerationConfig,
        system_prompt:Union[str, None]=None,
        tools:Union[List[Tool], None]=None,
        termination_function:Callable[[List[Message]], bool]=lambda x: False,
        reduce_function:Callable[[List[Message]], Message]=lambda x: x[-1],
    ):
        function_map = {tool.name: tool.run for tool in tools} if tools != None else {}
        a_function_map = {tool.name: tool.a_run for tool in tools} if tools != None else {}
        _generation_config = generation_config.model_copy(deep=True) if generation_config != None else None
        if tools != None and _generation_config != None:
            _generation_config.tools = {
                tool.name:Function(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.input_json_schema,
                ) for tool in tools
            }
        self.name = name
        self.system_prompt = system_prompt
        self.generation_config = _generation_config
        self.function_map = function_map
        self.a_function_map = a_function_map
        self.termination_function = termination_function
        self.reduce_function = reduce_function

        if self.generation_config.api_type in ['openai', 'fastchat', 'azure']:
            self.client = agentx.oai_client.OAIClient(
                generation_config=self.generation_config
            )

        if self.generation_config.api_type == 'vertexai':
            self.client = agentx.vertexai_client.VertexAIClient(
                generation_config=generation_config
            )
        
        if self.generation_config.api_type == 'bedrock':
            self.client = agentx.bedrock_client.BedrockClient()


    def generate_response(
        self,
        messages:List[Message],
        output_model:Union[OutputType, None]=None,
    ) -> Union[None, List[Message]]:
        """
        Generate a response to the given messages based on the generation config
        """
        # determine termination criteria
        if self.termination_function(messages):
            return None

        _messages = deepcopy(messages)
        # add system prompt
        if self.system_prompt != None:
            _messages = [Message(
                role='system',
                content=Content(
                    text=self.system_prompt
                ),
                name=self.name,
            )] + _messages

        message:Message = self.client.generate(
            messages=_messages,
            generation_config=self.generation_config,
            reduce_function=self.reduce_function,
            output_model=output_model,
        )
        if message == None:
            return None
        message.name = self.name

        generated_messages:List[Message] = [message]
        
        # tool calls
        tool_calls = generated_messages[-1].content.tool_calls
        while tool_calls != None:
            tool_responses:List[Message] = []
            multimodal_responses:List[Message] = []
            for tool_call in tool_calls:
                if tool_call.type == 'function':
                    function = self.function_map.get(tool_call.function_call.name)
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

            generated_messages += tool_responses + multimodal_responses

            second_message:Message = self.client.generate(
                messages=_messages + generated_messages,
                generation_config=self.generation_config,
                reduce_function=self.reduce_function,
                output_model=output_model,
            )
            second_message.name = self.name
            generated_messages += [second_message]
            tool_calls = generated_messages[-1].content.tool_calls
        
        return generated_messages

    async def a_generate_response(
        self, 
        messages:List[Message],
        output_model:Union[OutputType, None]=None,
    ) -> Union[None, List[Message]]:
        """
        Async version of generate_response
        """
        # determine termination criteria
        if self.termination_function(messages):
            return None
        
        _messages = deepcopy(messages)

        # add system prompt
        if self.system_prompt != None:
            _messages = [Message(
                role='system',
                content=Content(
                    text=self.system_prompt
                ),
                name=self.name,
            )] + _messages

        message:Message = await self.client.a_generate(
            messages=_messages,
            generation_config=self.generation_config,
            reduce_function=self.reduce_function,
            output_model=output_model,
        )
        
        if message == None:
            return None
        message.name = self.name

        generated_messages:List[Message] = [message]
        # tool calls
        tool_calls = generated_messages[-1].content.tool_calls

        while tool_calls != None:
            tasks = [
                self.a_function_map.get(
                    tool_call.function_call.name
                )(
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

            generated_messages += tool_responses + multimodal_responses

            # call language model again
            second_message = await self.client.a_generate(
                messages=messages + generated_messages,
                generation_config=self.generation_config,
                reduce_function=self.reduce_function,
                output_model=output_model,
            )
            if second_message == None:
                return None
            second_message.name = self.name
            generated_messages += [second_message]
            tool_calls = generated_messages[-1].content.tool_calls

        return generated_messages