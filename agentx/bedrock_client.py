from collections import defaultdict
import json
from typing import Any, Coroutine, Dict, List, Union
from defusedxml import ElementTree
import boto3
from uuid import uuid4
from agentx.schema import Message, Content, ToolCall, Function, GenerationConfig

# UNDER DEVELOPMENT

def transform_message_bedrock(message:Message) -> str:

    pass

def format_function_response(tool_name, output):
    return f"""
<function_results>
<result>
<tool_name>{tool_name}</tool_name>
<stdout>
{output}
</stdout>
</result>
</function_results>
"""

def swagger_to_xml(functions:List[Function])->str:

    pass

def etree_to_dict(t) -> dict[str, Any]:
    d = {t.tag: {}}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(("@" + k, v) for k, v in t.attrib.items())
    if t.text and t.text.strip():
        if children or t.attrib:
            d[t.tag]["#text"] = t.text
        else:
            d[t.tag] = t.text
    return d

class BedrockClient():

    def __init__(self):
        self.brt = boto3.client(service_name='bedrock-runtime')
    
    def generate(self, messages: List[Message], generation_config:GenerationConfig):
        # contents needs to be concatenated into a single string
        tools = swagger_to_xml(generation_config.tools)
        prompt = f"""In this environment you have access to a set of tools you can use to answer the user's question.

You may call them like this. Only invoke one function at a time and wait for the results before invoking another function:
<function_calls>
<invoke>
<tool_name>$TOOL_NAME</tool_name>
<parameters>
<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
...
</parameters>
</invoke>
</function_calls>

Here are the tools available:
<tools>
{tools}
</tools>"""
        
        for message in messages:
            role = message.role
            text = message.content.text
            if text == None:
                text = format_function_response(message.name, message.content.tool_response.content)
            prompt += f"{role.capitalize()}: {text}"

        body = json.dumps({
            "prompt": "",
            "max_tokens_to_sample": generation_config.max_tokens,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "top_k": generation_config.top_k,
            "stop_sequences": generation_config.stop_sequences,
        })

        # modelId is the name of the model
        # e.g. anthropic.claude-v2, anthropic.claude-v2:1, anthropic.claude-instant-v1
        modelId = generation_config.model
        accept = 'application/json'
        contentType = 'application/json'

        response = self.brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

        response_body = json.loads(response.get('body').read())

        completion, stop_reason, stop_seq = response_body.get('completion', None), response_body.get('stop_reason', None), response_body.get('stop_seq', None)

        # handle function calls
        if stop_reason == 'stop_sequence' and stop_seq == '</function_calls>':
            start_index = completion.find("<function_calls>")
            if start_index != -1:
                # Extract the XML Claude outputted (invoking the function)
                extracted_text = completion[start_index+16:]

                # Parse the XML find the tool name and the parameters that we need to pass to the tool
                xml = ElementTree.fromstring(extracted_text)
                tool_name_element = xml.find("tool_name")
                tool_name_from_xml = tool_name_element.text.strip()

                parameters_xml = xml.find("parameters")
                param_dict = etree_to_dict(parameters_xml)
                parameters = param_dict["parameters"]

                return Message(
                    role='assistant',
                    content=Content(
                        tool_calls=[
                            ToolCall(
                                id=uuid4(),
                                type='function',
                                function_call=Function(
                                    name=tool_name_from_xml,
                                    arguments=parameters
                                ),
                            )
                        ]
                    )
                )
        
        pass

    async def a_generate(self, messages: List[Message], generation_config: Dict) -> Union[Message, List[Message]]:
        # boto3 is not async so falling back to the sync version
        return self.generate(messages, generation_config)