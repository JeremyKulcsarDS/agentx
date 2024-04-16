import unittest
import os
from dotenv import load_dotenv
from siumai.schema import Message, Content, GenerationConfig
from siumai.oai_client import OAIClient
from siumai.vertexai_client import VertexAIClient
from siumai.bedrock_client import BedrockClient

load_dotenv()

class OAIClientTest(unittest.TestCase):
    def setUp(self):
        self.generation_config = GenerationConfig(
            api_type='azure',
            api_key=os.environ.get('AZURE_OPENAI_KEY'),
            base_url=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            azure_deployment='gpt-35',
        )
        self.client = OAIClient(generation_config=self.generation_config)

    def test_generate_response_with_image(self):
        pass

    def test_generate_responses_with_image(self):
        pass

    def test_generate_response_with_url(self):
        pass

    def test_generate_responses_with_url(self):
        pass

    def test_generate_response_with_tool_call(self):
        pass

    def test_generate_responses_with_tool_call(self):
        pass

    def test_generate_tool_call_response(self):
        pass

    def test_generate_tool_call_responses(self):
        pass

    def test_generate_response(self):
        response = self.client.generate(
            messages=[
                Message(role='system', content=Content(text='You are a helpful assistant.'), name='test_agnet'),
                Message(role='user', content=Content(text='Tell me a joke.'), name='user')
            ],
            generation_config=self.generation_config,
            reduce_function=lambda x: x[-1]
        )
        print(response.content.text)
        self.assertIsNotNone(response.content.text)

    def test_generate_responses(self):
        pass

class AsyncOAIClientTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.generation_config = GenerationConfig(
            api_type='azure',
            api_key=os.environ.get('AZURE_OPENAI_KEY'),
            base_url=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            azure_deployment='gpt-35',
        )
        self.client = OAIClient(generation_config=self.generation_config)

    async def test_generate_response(self):
        response = await self.client.a_generate(
            messages=[
                Message(role='system', content=Content(text='You are a helpful assistant.'), name='test_agnet'),
                Message(role='user', content=Content(text='Tell me a joke.'), name='user')
            ],
            generation_config=self.generation_config,
            reduce_function=lambda x: x[-1]
        )
        print(response.content.text)
        self.assertIsNotNone(response.content.text)

class VertexAIClientTest(unittest.TestCase):
    def test_generate_response_with_image(self):
        pass
    def test_generate_response_with_tool_call(self):
        pass
    def test_generate_tool_call_response(self):
        pass
    def test_generate_response_with_image(self):
        pass
    def test_generate_response_with_url(self):
        pass

class BedrockClientTest(unittest.TestCase):
    def test_generate_response_with_image(self):
        pass
    def test_generate_response_with_tool_call(self):
        pass
    def test_generate_tool_call_response(self):
        pass
    def test_generate_response_with_image(self):
        pass
    def test_generate_response_with_url(self):
        pass

if __name__ == '__main__':
    unittest.main()