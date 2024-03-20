import copy
from typing import Dict, Optional, Any, List
from vertexai import generative_models

GAPIC_SCHEMA_FIELDS = [
    "type", 
    "format", 
    "description", 
    "nullable", 
    "items", 
    "enum", 
    "properties", 
    "required", 
    "example",
    "$ref"
]


def replace_key(dictionary, old_key, new_key):
    """
    Recursively replaces all occurrences of the old_key with the new_key in a dictionary.

    Args:
        dictionary (Dict[str, Any]): The input dictionary to modify.
        old_key (str): The key to replace.
        new_key (str): The new key to use.

    Returns:
        Dict[str, Any]: The modified dictionary with replaced keys.
    """
    new_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            new_dict[key] = replace_key(value, old_key, new_key)  # Recursively process nested dictionaries
        elif isinstance(value, list):
            new_dict[key] = [replace_key(item, old_key, new_key) if isinstance(item, dict) else item for item in value]
            # Recursively process nested dictionaries within lists
        else:
            new_dict[key] = value
    if old_key in new_dict:
        new_dict[new_key] = new_dict.pop(old_key)  # Replace the old_key with the new_key
    return new_dict


def move_extra_fields_to_properties(dictionary: Dict[str, Any], gapic_schema_fields: list = GAPIC_SCHEMA_FIELDS) -> Dict[str, Any]:
    """
    Moves any field outside the "parameters" key that is in the GAPIC schema inside the "properties" field.
    https://cloud.google.com/vertex-ai/docs/reference/rpc/google.cloud.aiplatform.v1beta1#google.cloud.aiplatform.v1beta1.Schema

    Args:
        dictionary (dict): The input dictionary.

    Returns:
        dict: A modified copy of the input dictionary with extra fields moved to properties.

    Raises:
        None.
    """
    dictionary_copy = copy.deepcopy(dictionary)

    for key in dictionary["parameters"].keys():
        if key not in gapic_schema_fields:
            popped_key = dictionary_copy["parameters"].pop(key)
            dictionary_copy["parameters"]["properties"].update(popped_key)
            del popped_key

    return dictionary_copy


def pop_parameters(dictionary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pops and returns the value of the "parameters" key from the dictionary, if present.
    Args:
        dictionary (dict): The input dictionary.

    Returns:
        dict or None: The value of the "parameters" key, or None if the key is not present.

    Raises:
        None.
    """
    if "parameters" in dictionary:
        return copy.deepcopy(dictionary).pop("parameters")
    else:
        return None


def pop_properties(dictionary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pops and returns the value of the "properties" key from the dictionary, if present.

    Args:
        dictionary (dict): The input dictionary.

    Returns:
        dict or None: The value of the "properties" key, or None if the key is not present.

    Raises:
        None.
    """
    if "properties" in dictionary:
        return copy.deepcopy(dictionary).pop("properties")
    else:
        return None


def change_field_name_to_description(popped_dict: Dict[str, Any], gapic_schema_fields: list = GAPIC_SCHEMA_FIELDS) -> Dict[str, Any]:
    """
    Changes the name of fields that are not in the specified list to "description" within a nested dictionary.

    Args:
        popped_dict (dict): The input dictionary.

    Returns:
        dict: A modified copy of the input dictionary with field names changed to "description".

    Raises:
        None.
    """
    popped_dict_copy = copy.deepcopy(popped_dict)

    for key in popped_dict.keys():
        for subkey in popped_dict[key].keys():
            if subkey not in gapic_schema_fields:
                popped_dict_copy[key]["description"] = popped_dict_copy[key].pop(subkey)
        popped_dict[key] = popped_dict_copy[key]

    return popped_dict


def transform_openai_tool_to_vertexai_tool(dictionary: dict) -> dict:
    """
    Transforms an OpenAI tool dictionary to a Vertex AI tool dictionary by performing the following steps:
    1. Moves extra fields to properties.
    2. Replaces the key "title" with "description".
    3. Changes field names that arent in the GAPIC Schema to "description" within the properties dictionary.

    From the OpenAI schema, it is assume anything not in the GAPIC schema is a description (e.g. title)

    Args:
        dictionary (dict): The input dictionary.

    Returns:
        dict: The transformed dictionary.
    """
    dictionary_copy = copy.deepcopy(dictionary)

    # Move extra fields to properties and replace "title" with "description"
    dictionary_copy_fields_moved_to_prop = move_extra_fields_to_properties(replace_key(dictionary_copy, "title", "description"))

    # Change field names to "description" within properties dictionary
    dictionary_copy_fields_moved_to_prop["parameters"]["properties"] = change_field_name_to_description(pop_properties(pop_parameters(dictionary_copy_fields_moved_to_prop)))

    return dictionary_copy_fields_moved_to_prop


def add_empty_messages(list_messages: List[generative_models.Content]) -> List[generative_models.Content]:
    """
    Adds empty model messages between consecutive messages with the same role.
    
    Vertex AI cannot handle multiturn with the same role.
    If the user role appears twice in a row, put an empty model message in between.
    This can happen for instance if a system prompt is given.
    
    This function isn't used yet in the client since we replaced it with a simple injection of the 
    system prompt content into the first user message content.

    Args:
        list_messages (List[generative_models.Content]): List of messages to process.

    Returns:
        List[generative_models.Content]: List of messages with empty model messages inserted.

    Example:
        list_messages = [
            Content(role="user", parts=[Part.from_text("Hello")]),
            Content(role="user", parts=[Part.from_text("Please wait")]),
            Content(role="model", parts=[Part.from_text("Processing...")]),
            Content(role="user", parts=[Part.from_text("Can you provide more details?")])
        ]

        result = add_empty_messages(list_messages)

        print(result)

        Output:
        [
            Content(role="user", parts=[Part.from_text("Hello")]),
            Content(role="model", parts=[Part.from_text("Add the above message to the next one.")]),
            Content(role="user", parts=[Part.from_text("Please wait")]),
            Content(role="model", parts=[Part.from_text("Processing...")]),
            Content(role="user", parts=[Part.from_text("Can you provide more details?")])
        ]
    """
    # Create the empty model message
    empty_content_model_msg = generative_models.Content(
        role="model",
        parts=[
            generative_models.Part.from_text("Add the above message to the next one.")
        ]
    )

    # Add the empty message between every pair that has the same role
    list_messages_multiturn = []
    for i in range(len(list_messages) - 1):
        list_messages_multiturn.append(list_messages[i])
        if list_messages[i].role == list_messages[i + 1].role:
            list_messages_multiturn.append(empty_content_model_msg)

    list_messages_multiturn.append(list_messages[-1])

    return list_messages_multiturn