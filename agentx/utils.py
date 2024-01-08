import base64
from typing import Callable, Coroutine, List, Union
from inspect import iscoroutinefunction
from agentx.schema import Message

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_coroutine(function:Callable, **kwargs) -> Coroutine:
    if iscoroutinefunction(function):
        return function(**kwargs)
    else:
        async def wrapper():
            return function(**kwargs)
        return wrapper()

def append_to_messages(messages:List[Message], message:Union[Message, List[Message]]) -> List[Message]:
    if isinstance(message, list):
        messages.extend(message)
    else:
        messages.append(message)
    return messages