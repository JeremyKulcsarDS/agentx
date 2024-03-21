import copy
from typing import Dict, Optional, Any, List, Union
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


def move_title_to_parameters(dictionary: Dict[str, Any], gapic_schema_fields: list = GAPIC_SCHEMA_FIELDS) -> Dict[str, Any]:
    """
    Moves the title field to parameters
    https://cloud.google.com/vertex-ai/docs/reference/rpc/google.cloud.aiplatform.v1beta1#google.cloud.aiplatform.v1beta1.Schema

    Args:
        dictionary (dict): The input dictionary.

    Returns:
        dict: A modified copy of the input dictionary with extra fields moved to properties.

    Raises:
        None.
    """
    dictionary_copy = copy.deepcopy(dictionary)

    for key in dictionary_copy.keys():
        if key == 'title':
            popped_key = dictionary_copy.pop(key)
            dictionary_copy["parameters"].update(popped_key)
            del popped_key

    return dictionary_copy


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


def resolve_json_references(json_data: Dict[str, Any]) -> Union[Dict[str, Any], Any]:
    """
    Resolves JSON references in the provided JSON data.

    Args:
        json_data: The JSON data containing the references.

    Returns:
        The resolved JSON data where the references are replaced with the actual content.
    """
    def resolve_refs(obj: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any], Any]:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref = obj["$ref"]
                reference = ref.split("/")[1:]
                resolved = json_data
                for key in reference:
                    resolved = resolved[key]
                return resolved
            else:
                return {k: resolve_refs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_refs(item) for item in obj]
        else:
            return obj

    resolved_json = resolve_refs(json_data)
    return resolved_json


def move_defs_to_root(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Moves the "$defs" field from the "parameters" field to the root level of the dictionary.

    Args:
        dictionary: The JSON dictionary to process.

    Returns:
        The updated dictionary with the "$defs" field moved to the root level.
    """
    dictionary_copy = copy.deepcopy(dictionary)

    # Check if "$defs" field is present in "parameters"
    if "$defs" in dictionary_copy["parameters"]:
        defs_data = dictionary_copy["parameters"]["$defs"]
        del dictionary_copy["parameters"]["$defs"]
        dictionary_copy["$defs"] = defs_data

    return dictionary_copy


def transform_agentx_tool_to_vertexai_tool(dictionary: dict) -> dict:
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
    
    # Mode the $defs out of the parameters
    dictionary_copy_defs_at_root = move_defs_to_root(dictionary_copy)
    
    # Resolve the refs and remove the $defs field
    dictionary_copy_resolved = resolve_json_references(dictionary_copy_defs_at_root)
    
    if "$defs" in dictionary_copy_resolved.keys():
        del dictionary_copy_resolved['$defs']

    # Move extra fields to properties and replace "title" with "description"
    dictionary_copy_fields_moved_to_prop = move_extra_fields_to_properties(replace_key(dictionary_copy_resolved, "title", "description"))

    # Change field names to "description" within properties dictionary
    dictionary_copy_fields_moved_to_prop["parameters"]["properties"] = change_field_name_to_description(pop_properties(pop_parameters(dictionary_copy_fields_moved_to_prop)))

    return dictionary_copy_fields_moved_to_prop