import unittest
import os
from dotenv import load_dotenv
from agentx.tool import Tool
from pydantic import BaseModel, Field

load_dotenv()

class ToolTest(unittest.TestCase):

    def test_tool_publish(self):
        class SearchQuery(BaseModel):
            query: str
            top_k: int = Field(16, ge=1, le=16)
        
        tool_search = Tool(
            name='tool_search',
            description='Search for tools that accomplish a specific task.',
            markdown="",
            input_model=SearchQuery,
            price=0,
            endpoint='http://localhost:7071/search/tools'
        )

        publishing = tool_search.publish(
            api_key=os.environ.get('XENTROPY_API_KEY'),
            public=True
        )

        tool_search = Tool(
            name='tool_search_paid',
            description='Search for tools that accomplish a specific task.',
            markdown="",
            input_model=SearchQuery,
            price=10, # cost per 1000 calls
            endpoint='http://localhost:7071/search/tools'
        )

        publishing = tool_search.publish(
            api_key=os.environ.get('XENTROPY_API_KEY'),
            webhook_secret=os.environ.get('XENTROPY_WEBHOOK_SECRET'),
            public=True
        )
        
        print(publishing)


    def test_tool_load(self):
        tool = Tool.load(
            name='xentropy:tool_search',
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
            name='xentropy:tool_search',
            api_key=os.environ.get('XENTROPY_API_KEY')
        )
        self.assertTrue(issubclass(tool.input_model, BaseModel))

    def test_tool_output_model(self):
        tool = Tool.load(
            name='xentropy:tool_search',
            api_key=os.environ.get('XENTROPY_API_KEY')
        )
        if tool.output_model != None:
            self.assertTrue(issubclass(tool.output_model, BaseModel))

    def test_tool_call(self):
        tool = Tool.load(
            name='xentropy:tool_search',
            api_key=os.environ.get('XENTROPY_API_KEY')
        )
        tool.run(
            query='search',
            top_k=10
        )
        pass

    def test_tool_call_with_error_handler(self):
        pass

    def test_paid_tool_call(self):

        tool_search_paid = Tool.load(
            name='xentropy:tool_search_paid',
            api_key=os.environ.get('JUSTINTIME_API_KEY')
        )

        result = tool_search_paid.run(
            query='search',
            top_k=10
        )

        print(result)
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