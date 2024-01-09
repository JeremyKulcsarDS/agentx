from hashlib import sha256
import uuid
import agentx.models
import requests
import aiohttp
import os
import json
from os.path import dirname
from pydantic import BaseModel, AnyHttpUrl
from typing import Dict, Optional
from importlib import reload


def to_camel_case(snake: str):
    return ''.join([x.capitalize() for x in snake.split('_')])


class StringOutput(BaseModel):
    output: str


class Tool():

    def __init__(
        self,
        api_key=None,
        **kwargs
    ):
        # api token for server access
        self.api_key: Optional[str] = api_key
        if self.api_key == None:
            self.api_key = os.environ.get('FM_TOOL_API_KEY')

        self.name: Optional[str] = kwargs.get('name')
        self.description: Optional[str] = kwargs.get('description')
        self.input_model: Optional[BaseModel] = kwargs.get('input_model')
        self.input_json_schema: Optional[Dict] = kwargs.get('input_json_schema')
        self.output_model: Optional[BaseModel] = kwargs.get('output_model')
        self.output_json_schema: Optional[Dict] = kwargs.get(
            'output_json_schema')
        self.endpoint: Optional[AnyHttpUrl] = kwargs.get('endpoint')
        self.dynamic_price: Optional[AnyHttpUrl] = kwargs.get('dynamic_price')
        self.price: Optional[int] = kwargs.get('price')
        self.free_quota: Optional[int] = kwargs.get('free_quota')
        self.endpoint: Optional[AnyHttpUrl] = kwargs.get('endpoint')

    @classmethod
    def load(cls, name, api_key):
        tool = Tool(api_key=api_key, name=name)

        response = requests.get(
            f'https://api.xentropy.co/tool/{name}',
            headers={
                'Api-Key': api_key
            }
        )

        if response.status_code != 200:
            raise Exception(response.text)

        response = response.json()

        if 'exception' in response.keys():
            raise Exception(response['exception'])
        
        for key, value in response.items():
            if key in ['input_model', 'output_model']:
                path = dirname(__file__)
                filename = name.replace('--', '_')
                with open(f'{path}/models/{filename}_{key}.py', 'w') as f:
                    f.write(value)
                module = reload(agentx.models)
                submodule = getattr(module, f'{filename}_{key}')
                model = getattr(submodule, to_camel_case(key))
                setattr(tool, key, model)
                continue
            
            if key in ['input_json_schema', 'output_json_schema']:
                setattr(tool, key, json.loads(value))
                continue
            
            setattr(tool, key, value)

        return tool

    def publish(self, webhook_secret: Optional[str] = None, public=True):
        # create a random webhook_secret if not given
        if webhook_secret == None:
            webhook_secret = sha256(uuid.uuid4().bytes).hexdigest()

        assert (self.input_model != None)
        self.input_json_schema = self.input_model.model_json_schema()

        self.output_json_schema = StringOutput.model_json_schema()
        if self.output_model != None:
            self.output_json_schema = self.output_model.model_json_schema()

        response = requests.post(
            f'https://api.xentropy.co/tool',
            json={
                'name': self.name,
                'description': self.description,
                'input_json_schema': json.dumps(self.input_json_schema),
                'output_json_schema': json.dumps(self.output_json_schema),
                'endpoint': self.endpoint,
                'price': self.price,
                'free_quota': self.free_quota,
                'dynamic_price': self.dynamic_price,
                'webhook_secret': webhook_secret,
                'public': public
            },
            headers={'Api-Key': self.api_key}
        )
        if response.status_code == 200:
            res = response.json()
            name = res['name']
            webhook_secret = res['webhook_secret']
            print(
                f'Tool {name} is successfully published, and its webhook_secret is {webhook_secret}. You can access it by Tool.load("{name}").')
            return {'name': name, 'webhook_secret': webhook_secret}
        else:
            return response.json()

    def run(
        self,
        *args,
        **kwargs
    ) -> str:
        url = f'https://api.xentropy.co/tool/{self.name}'

        # build headers
        parent_id = kwargs.pop('parent_id', None)
        headers = {
            'Api-Key': self.api_key,
            'Parent-Id': parent_id
        }

        # use the tool
        model = self.input_model.model_validate(kwargs)

        resp = requests.post(
            url,
            data=bytes(model.json(), 'utf-8'),
            headers=headers,
        )

        if resp.status_code >= 300:
            raise Exception(resp.text)

        response = resp.json()
        status = response.get('status')
        if status >= 300:
            raise Exception(response.get('error_message'))

        return response.get('response')

    async def arun(
        self,
        *args,
        **kwargs,
    ) -> str:
        url = f'https://api.xentropy.co/tool/{self.name}'

        # build headers
        parent_id = kwargs.pop('parent_id', None)
        headers = {
            'Api-Key': self.api_key,
            'Parent-Id': parent_id,
            'Content-Type': 'application/json',
        }

        # use the tool
        model = self.input_model.model_validate(kwargs)

        # use the tool asynchronously
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=bytes(model.json(), 'utf-8'), headers=headers) as resp:

                if resp.status >= 300:
                    raise Exception(await resp.text())

                response = await resp.json()

                status = response.get('status')
                if status >= 300:
                    raise Exception(response.get('error_message'))

                return response.get('response')
