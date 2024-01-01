from pydantic import BaseModel, HttpUrl, Base64Str, FilePath
from typing import Dict, List, Optional, Union, Literal

class Function(BaseModel):
    name:str
    description:str
    parameters: Dict

class FunctionCall(BaseModel):
    name: str
    arguments: str

class File(BaseModel):
    mime_type: str
    base64Str: Base64Str

class ToolCall(BaseModel):
    id: str
    type: str
    function: FunctionCall

class ToolResponse(BaseModel):
    id:str
    content:str

class Content(BaseModel):
    text: Optional[str] = None
    files: Optional[List[File]] = None
    urls: Optional[List[HttpUrl]] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_response: Optional[ToolResponse] = None

class Message(BaseModel):
    role: str
    content: Content
    name:Optional[str] = None

class GenerationConfig(BaseModel):
    """
    Configuration class for generation settings.

    Attributes:
        api_type (Literal['azure', 'bedrock', 'fastchat', 'openai', 'vertexai']): The type of API to use.
        api_key (Optional[str]): The API key to authenticate the requests. Defaults to None.
        api_version (Optional[str]): The version of the API to use. Defaults to "2023-12-01-preview".
        organization (Optional[str]): The organization associated with the API. Defaults to None. openai / azure only
        base_url (Optional[HttpUrl]): The base URL for the API. Defaults to None. openai / azure only
        timeout (float): The timeout duration for API requests in seconds. Defaults to 120.
        max_retries (int): The maximum number of retries for failed API requests. Defaults to 3.
        azure_deployment (Optional[str]): The Azure deployment to use. Defaults to None.
        path_to_google_service_account_json (Optional[FilePath]): The path to the Google service account JSON file. Defaults to None. vertexai only
        google_application_credential_scope (Optional[List[str]]): The scope of the Google application credential. Defaults to None. vertexai only
        region (Optional[str]): The region to use. Defaults to None. vertexai only
        project_id (Optional[str]): The project ID to use. Defaults to None. vertexai only
        model (Optional[str]): The model to use. Defaults to None.
        n_candidates (int): The number of candidates to generate. Defaults to 1. Eqivalent to 'n' in openai and 'candidateCount' in vertexai
        frequency_penalty (Optional[float]): The frequency penalty for generation. Defaults to None.
        logit_bias (Optional[Dict[str, int]]): The logit bias for generation. Defaults to None.
        max_tokens (Optional[int]): The maximum number of tokens to generate. Defaults to None.
        presence_penalty (Optional[float]): The presence penalty for generation. Defaults to None.
        response_format (Optional[Dict[str, str]]): The response format for generation. Defaults to None. Not intended to be used explicitly.
        seed (Optional[int]): The seed value for generation. Defaults to None.
        stop_sequences (Optional[List[str]]): The stop sequences for generation.
        temperature (Optional[float]): The temperature value for generation. Defaults to None.
        tool_choice (Optional[Union[str, Dict[str, Union[str, Dict[str, str]]]]]): The choice of tool for generation. Defaults to None. openai / azure only
        tools (Optional[List[Function]]): The list of tools to use for generation. Defaults to None.
        top_p (Optional[int]): The top-p value for generation. Defaults to None.
        top_k (Optional[int]): The top-k value for generation. Defaults to None.
    """
    api_type: Literal['azure', 'bedrock', 'fastchat', 'openai', 'vertexai']
    api_key: Optional[str] = None
    api_version: Optional[str] = "2023-12-01-preview"
    organization: Optional[str] = None
    base_url: Optional[HttpUrl] = None
    timeout: float = 120
    max_retries: int = 3
    azure_deployment: Optional[str] = None
    path_to_google_service_account_json: Optional[FilePath] = None
    google_application_credential_scope: Optional[List[str]] = None
    region: Optional[str] = None
    project_id: Optional[str] = None
    model: Optional[str] = None
    n_candidates: int = 1
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, int]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    temperature: Optional[float] = None
    tool_choice: Optional[Union[str, Dict[str, Union[str, Dict[str, str]]]]] = None
    tools: Optional[List[Function]] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None