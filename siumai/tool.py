from hashlib import sha256
import uuid
import siumai.models
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
    url = 'https://api.xentropy.co'

    def __init__(
        self,
        api_key=None,
        **kwargs
    ):
        # api token for server access
        self.api_key: Optional[str] = api_key
        if self.api_key == None:
            self.api_key = os.environ.get('XENTROPY_API_KEY')

        self.id: Optional[str] = kwargs.get('id')
        self.name: Optional[str] = kwargs.get('name')
        self.description: Optional[str] = kwargs.get('description')
        self.readme: Optional[str] = kwargs.get('readme')
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

        if self.input_model != None:
            self.input_json_schema = self.input_model.model_json_schema()
        if self.output_model != None:
            self.output_json_schema = self.output_model.model_json_schema()

    @classmethod
    def load(cls, id, api_key):
        tool = Tool(api_key=api_key, id=id)

        response = requests.get(
            f'{cls.url}/tools/{id}',
            headers={
                'Api-Key': api_key
            }
        )

        if response.status_code != 200:
            raise Exception(response.text)

        tool_dict:Dict = response.json()

        if 'detail' in tool_dict.keys():
            raise Exception(tool_dict['detail'])
        
        for key, value in tool_dict.items():
            if key in ['input_model', 'output_model']:
                path = dirname(__file__)
                filename = id
                with open(f'{path}/models/{filename}_{key}.py', 'w') as f:
                    f.write(value)
                module = reload(siumai.models)
                submodule = getattr(module, f'{filename}_{key}')
                model = getattr(submodule, to_camel_case(key))
                setattr(tool, key, model)
                continue
            
            if key in ['input_json_schema', 'output_json_schema']:
                setattr(tool, key, json.loads(value))
                continue
            
            setattr(tool, key, value)

        return tool


    def publish(self, api_key: Optional[str] = None, webhook_secret: Optional[str] = None, public=True):
        self.api_key = api_key
        # create a random webhook_secret if not given
        if webhook_secret == None:
            webhook_secret = uuid.uuid4().hex

        assert (self.input_model != None)
        self.input_json_schema = self.input_model.model_json_schema()

        self.output_json_schema = StringOutput.model_json_schema()
        if self.output_model != None:
            self.output_json_schema = self.output_model.model_json_schema()

        response = requests.post(
            f'{self.url}/tools/',
            json={
                'name': self.name,
                'description': self.description,
                'readme': self.readme,
                'input_json_schema': json.dumps(self.input_json_schema),
                'output_json_schema': json.dumps(self.output_json_schema),
                'endpoint': self.endpoint,
                'price': self.price,
                'free_quota': self.free_quota,
                'dynamic_price': self.dynamic_price,
                'webhook_secret': webhook_secret,
                'public': public
            },
            headers={'Api-Key': self.api_key},
        )
        if response.status_code == 200:
            res = response.json()
            name = res['name']
            webhook_secret = res['webhook_secret']
            print(
                f'Tool {name} is successfully published. You can access it by Tool.load("{name}").')
            return {'name': name, 'webhook_secret': webhook_secret}
        else:
            return response.json()


    def run(
        self,
        *args,
        **kwargs
    ) -> str:
        url = f'{self.url}/tools/{self.id}'

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
            data=bytes(model.model_dump_json(), 'utf-8'),
            headers=headers,
        )

        if resp.status_code >= 300:
            raise Exception(resp.text)

        response = resp.json()
        status = response.get('status')
        if status >= 300:
            raise Exception(response.get('response'))

        return response.get('response')


    async def a_run(
        self,
        *args,
        **kwargs,
    ) -> str:
        url = f'{self.url}/tools/{self.id}'

        # build headers
        headers = {
            'Api-Key': self.api_key,
            'Content-Type': 'application/json',
        }

        parent_id = kwargs.pop('parent_id', None)
        if parent_id != None:
            headers['Parent-Id'] = parent_id

        # use the tool
        model = self.input_model.model_validate(kwargs)

        # use the tool asynchronously
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=bytes(model.model_dump_json(), 'utf-8'), headers=headers) as resp:

                if resp.status >= 300:
                    raise Exception(await resp.text())

                response = await resp.json()

                status = response.get('status')
                if status >= 300:
                    raise Exception(response.get('detail'))

                return response.get('response')
