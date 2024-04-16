import json
from typing import List
import unittest
import os
from pydantic import BaseModel
from dotenv import load_dotenv
from siumai.schema import GenerationConfig, Message, Content, ToolResponse, Function
from siumai.agent import Agent
from siumai.utils import encode_image
from siumai.tool import Tool

load_dotenv()

class TestModel(BaseModel):
    timestamp:int
    name:str

class TestTool(Tool):
    def run(self, **kwargs):
        return json.dumps({'response':'success'})
    async def arun(self, **kwargs):
        return json.dumps({'response':'success'})

class AgentTestCase(unittest.TestCase):
    
    def setUp(self):
        self.generation_config = GenerationConfig(
            api_type='openai',
            api_key=os.environ.get('OPENAI_API_KEY'),
            #base_url=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            model='gpt-3.5-turbo-1106'
        )

    def test_generate_termination_message(self):
        generation_config = self.generation_config
        #generation_config.azure_deployment = 'gpt-35'
        def terminate(message:List[Message]):
            if message[-1].content.text == 'some text':
                return True
            return False
        agent = Agent(
            name='test_agnet',
            generation_config=generation_config,
            termination_function=terminate
        )
        message_should_terminate = agent.generate_response(
            messages=[
                Message(role='assistant', content=Content(text='some text'), name='test')
            ]
        )
        self.assertIsNone(message_should_terminate)

    def test_generate_text_response(self):
        generation_config = self.generation_config
        #generation_config.azure_deployment = 'gpt-35'

        agent = Agent(
            name='test_agent',
            generation_config=generation_config,
        )
        text_message = agent.generate_response(
            messages=[
                Message(role='system', content=Content(text='You are a helpful assistant.'), name='test_agnet'),
                Message(role='user', content=Content(text='Tell me a joke.'), name='user')
            ]
        )

        self.assertEqual(text_message[-1].role, 'assistant')
        print(text_message[-1].content.text)
        self.assertIsNotNone(text_message[-1].content.text)

    def test_generate_output_model_response(self):
        generation_config = self.generation_config
        generation_config.azure_deployment = 'gpt-35'
        agent = Agent(
            name='test_agent',
            generation_config=generation_config,
        )
        json_message = agent.generate_response(
            messages=[
                Message(role='system', content=Content(text='You are a helpful assistant.'), name='test_agnet'),
                Message(role='user', content=Content(text='Fill the following test case with random timestamp and name.'), name='user')
            ],
            output_model=TestModel
        )

        test_output_model = None
        try:
            test_output_model = TestModel.model_validate_json(json_message.content.text)
        except Exception as e:
            print(e)
            pass
        self.assertIsInstance(test_output_model, TestModel)

    def test_generate_response_with_tool_calls_wo_image(self):
        _messages = []
        # Create a test message with tool calls
        tools = [
            TestTool(
                name='test_tool_0',
                description='this function test if LLM agent can make function calls.',
                input_json_schema=TestModel.model_json_schema()
            ),
            TestTool(
                name='test_tool_1',
                description='this function test if LLM agent can make function calls.',
                input_json_schema=TestModel.model_json_schema()
            ),
        ]
        generation_config = self.generation_config
        generation_config.azure_deployment = 'gpt-35'

        agent = Agent(
            name='test_agnet',
            system_prompt='You are a helpful assistant. Use the function you have been provided.',
            generation_config=generation_config,
            tools = tools
        )

        _messages.append(
            Message(
                role='user', 
                content=Content(
                    text='Call test_function_0 and test_function_1 for a test'), 
                    name='user'
                )
        )
        # expect a tool call
        response = agent.generate_response(
            messages=_messages
        )

        self.assertEqual(response.role, 'assistant')
        print(response.content.tool_calls)
        self.assertIsInstance(response.content.tool_calls, List)

        _messages.append(response)
        # expect a response from calling the tool
        response = agent.generate_response(
            messages=_messages
        )

        for message in response:
            print(message.content.tool_response)
            self.assertEqual(message.role, 'tool')
            self.assertIsInstance(message.content.tool_response, ToolResponse)
        
        _messages.extend(response)
        response = agent.generate_response(
            messages=_messages
        )

        self.assertEqual(response.role, 'assistant')
        self.assertIsNotNone(response.content.text)

    def test_generate_response_with_image(self):
        generation_config = self.generation_config
        generation_config.azure_deployment = 'gpt-4v'
        
        agent = Agent(
            name='test_agnet',
            generation_config=generation_config
        )

        # Create a test message with image
        with open('./tests/test_image.jpg', 'rb') as f:
            base64Str = encode_image(f)

        message = Message(
            role="user",
            content=Content(
                text='What is in this picture?',
                files=[
                    {
                        "mime_type": "image/jpeg",
                        "base64Str": base64Str,
                    }
                ]
            ),
        )

        # Call the generate_response method
        response = agent.generate_response(messages=[message])

        # Assert the response
        self.assertEqual(response.role, 'assistant')
        print(response.content.text)
        self.assertIsInstance(response.content.text, str)

    def test_generate_response_with_url(self):
        generation_config = self.generation_config
        generation_config.azure_deployment = 'gpt-4v'
        
        agent = Agent(
            name='test_agent',
            generation_config=generation_config
        )
        # Create a test message with image
        message = Message(
            role="user",
            content=Content(
                text='What is in this picture?',
                urls=['https://upload.wikimedia.org/wikipedia/en/5/5f/Original_Doge_meme.jpg']
            ),
        )

        # Call the generate_response method
        response = agent.generate_response(messages=[message])

        # Assert the response
        self.assertEqual(response.role, 'assistant')
        print(response.content.text)
        self.assertIsInstance(response.content.text, str)


class AsyncAgentTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.generation_config = GenerationConfig(
            api_type='azure',
            api_key=os.environ.get('AZURE_OPENAI_KEY'),
            base_url=os.environ.get('AZURE_OPENAI_ENDPOINT'),
        )

    async def test_a_generate_termination_message(self):
        generation_config = self.generation_config
        generation_config.azure_deployment = 'gpt-35'
        def terminate(message:List[Message]):
            if message[-1].content.text == 'some text':
                return True
            return False
        agent = Agent(
            name='test_agnet',
            generation_config=generation_config,
            termination_function=terminate
        )
        message_should_terminate = await agent.a_generate_response(
            messages=[
                Message(role='assistant', content=Content(text='some text'), name='test')
            ]
        )
        self.assertIsNone(message_should_terminate)

    async def test_a_generate_text_response(self):
        generation_config = self.generation_config
        generation_config.azure_deployment = 'gpt-35'

        agent = Agent(
            name='test_agnet',
            generation_config=generation_config,
        )
        text_message = await agent.a_generate_response(
            messages=[
                Message(role='system', content=Content(text='You are a helpful assistant.'), name='test_agnet'),
                Message(role='user', content=Content(text='Tell me a joke.'), name='user')
            ]
        )

        self.assertEqual(text_message.role, 'assistant')
        print(text_message.content.text)
        self.assertIsNotNone(text_message.content.text)

    async def test_a_generate_output_model_response(self):
        generation_config = self.generation_config
        generation_config.azure_deployment = 'gpt-35'
        agent = Agent(
            name='test_agent',
            generation_config=generation_config,
        )
        json_message = await agent.a_generate_response(
            messages=[
                Message(role='system', content=Content(text='You are a helpful assistant.'), name='test_agnet'),
                Message(role='user', content=Content(text='Fill the following test case with random timestamp and name.'), name='user')
            ],
            output_model=TestModel
        )

        test_output_model = None
        try:
            test_output_model = TestModel.model_validate_json(json_message.content.text)
        except Exception as e:
            print(e)
            pass
        self.assertIsInstance(test_output_model, TestModel)

    async def test_a_generate_response_with_tool_calls_wo_image(self):
        # Create a test message with tool calls
        tools = [
            TestTool(
                name='test_tool_0',
                description='this function test if LLM agent can make function calls.',
                input_json_schema=TestModel.model_json_schema()
            ),
            TestTool(
                name='test_tool_1',
                description='this function test if LLM agent can make function calls.',
                input_json_schema=TestModel.model_json_schema()
            ),
        ]

        generation_config = self.generation_config
        generation_config.azure_deployment = 'gpt-35'

        agent = Agent(
            name='test_agnet',
            system_prompt='You are a helpful assistant. Use the function you have been provided.',
            generation_config=generation_config,
            tools = tools
        )

        # expect a tool call
        response = await agent.a_generate_response(
            messages=[
                Message(role='user', content=Content(text='Call test_function for a test'), name='user')
            ],
        )

        self.assertEqual(response.role, 'assistant')
        print(response.content.tool_calls)
        self.assertIsInstance(response.content.tool_calls, List)

        # expect a response from calling the tool
        response = await agent.a_generate_response(
            messages=[
                Message(role='system', content=Content(text='You are a helpful assistant. Use the function you have been provided.'), name='test_agnet'),
                Message(role='user', content=Content(text='Call test_function for a test'), name='user'),
                response
            ],
        )

        for message in response:
            print(message.content.tool_response)
            self.assertEqual(message.role, 'tool')
            self.assertIsInstance(message.content.tool_response, ToolResponse)

    async def test_a_generate_response_with_image(self):
        generation_config = self.generation_config
        generation_config.azure_deployment = 'gpt-4v'
        
        agent = Agent(
            name='test_agnet',
            generation_config=generation_config
        )

        # Create a test message with image
        with open('./tests/test_image.jpg', 'rb') as f:
            base64Str = encode_image(f)

        message = Message(
            role="user",
            content=Content(
                text='What is in this picture?',
                files=[
                    {
                        "mime_type": "image/jpeg",
                        "base64Str": base64Str,
                    }
                ]
            ),
        )

        # Call the generate_response method
        response = await agent.a_generate_response(messages=[message])

        # Assert the response
        self.assertEqual(response.role, 'assistant')
        print(response.content.text)
        self.assertIsInstance(response.content.text, str)

    async def test_a_generate_response_with_url(self):
        generation_config = self.generation_config
        generation_config.azure_deployment = 'gpt-4v'
        
        agent = Agent(
            name='test_agent',
            generation_config=generation_config
        )
        # Create a test message with image
        message = Message(
            role="user",
            content=Content(
                text='What is in this picture?',
                urls=['https://upload.wikimedia.org/wikipedia/en/5/5f/Original_Doge_meme.jpg']
            ),
        )

        # Call the generate_response method
        response = await agent.a_generate_response(messages=[message])

        # Assert the response
        self.assertEqual(response.role, 'assistant')
        print(response.content.text)
        self.assertIsInstance(response.content.text, str)
    

if __name__ == "__main__":
    unittest.main()