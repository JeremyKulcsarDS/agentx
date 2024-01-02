import base64
from typing import Callable, Coroutine
from inspect import iscoroutinefunction

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_coroutine(function:Callable, **kwargs) -> Coroutine:
    if iscoroutinefunction(function):
        return function(**kwargs)
    else:
        async def wrapper():
            return function(**kwargs)
        return wrapper()