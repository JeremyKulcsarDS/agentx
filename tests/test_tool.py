import unittest
import os
from dotenv import load_dotenv
from agentx.tool import Tool
from pydantic import BaseModel

load_dotenv()

class ToolTest(unittest.TestCase):
    def test_tool_load(self):
        tool = Tool.load(
            name='xentropy--tool_search',
            api_key=os.environ.get('XENTROPY_API_KEY')
        )
        self.assertIsNotNone(tool)
        self.assertIsNotNone(tool.name)
        self.assertIsNotNone(tool.description)
        self.assertIsNotNone(tool.input_json_schema)
        self.assertIsNotNone(tool.price)
        self.assertIsNotNone(tool.free_quota)

    def test_tool_input_model(self):
        tool = Tool.load(
            name='xentropy--tool_search',
            api_key=os.environ.get('XENTROPY_API_KEY')
        )
        self.assertTrue(issubclass(tool.input_model, BaseModel))

    def test_tool_output_model(self):
        tool = Tool.load(
            name='xentropy--tool-search',
            api_key=os.environ.get('XENTROPY_API_KEY')
        )
        if tool.output_model != None:
            self.assertIsInstance(tool.output_model, BaseModel)

    def test_tool_call(self):
        pass

    def test_tool_call_with_error_handler(self):
        pass

class AsyncToolTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = Tool.load(
            name='xentropy--tool-search',
            api_key=os.environ.get('XENTROPY_API_KEY')
        )
    
    async def test_async_tool_call(self):
        pass

    async def test_async_tool_call_with_error_handler(self):
        pass

if __name__ == '__main__':
    unittest.main()