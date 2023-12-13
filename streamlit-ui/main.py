import streamlit as st
import autogen
import requests
from xentropy.tool import Tool
from dotenv import load_dotenv

load_dotenv()

config_list = autogen.config_list_from_models(
    model_list=["gpt-35"],
)

st.set_page_config(layout='wide')

st.title('XEntropy Playground')

# API Key Input
api_key = st.sidebar.text_input(
    'XEntropy API KEY',
    type="password"
)

# Documentation Link
st.sidebar.link_button('Documentation', 'https://github/')


# Initialize the states
if not st.session_state.get('selected_tools'):
    st.session_state['selected_tools'] = []
if not st.session_state.get('searched_tools'):
    st.session_state['searched_tools'] = []


def add_to_agent(tool):
    if tool not in st.session_state['selected_tools']:
        st.session_state['selected_tools'].append(tool)
        st.session_state['searched_tools'].remove(tool)


def remove_from_agent(tool):
    st.session_state['selected_tools'].remove(tool)


# Tool Search
st.markdown('## Tool Search')
if tool_search_query := st.text_input(
        'Equip LLM agent with external functionalities.'):
    if tool_search_query != st.session_state.get('tool_search_query'):
        response = requests.post(
            "https://api.xentropy.co/search_tool",
            json={'query': tool_search_query},
        )
        tool_search_result = response.json()
        st.session_state['searched_tools'] = tool_search_result
        st.session_state['tool_search_query'] = tool_search_query

for tool in st.session_state['searched_tools']:
    with st.container():
        col1, col2, col3 = st.columns([0.7, 0.1, 0.2])
        with col1:
            st.markdown("##### {}".format(tool.get('tool_id')))
        with col2:
            st.link_button(
                'Details',
                "https://xentropy.co/tool/{tool_id}".format(
                    tool_id=tool.get('tool_id'))
            )
        with col3:
            st.button(
                'Add to agent',
                on_click=add_to_agent,
                key="add_{tool_id}".format(tool_id=tool.get('tool_id')),
                args=[tool]
            )
        st.markdown(
            tool.get('description')
        )
        st.divider()

st.markdown('### Selected Tools')
for tool in st.session_state['selected_tools']:
    with st.container():
        col1, col2, col3 = st.columns([0.7, 0.1, 0.2])
        with col1:
            st.markdown("##### {}".format(tool.get('tool_id')))

        with col2:
            st.link_button(
                'Details',
                "https://xentropy.co/tool/{tool_id}".format(
                    tool_id=tool.get('tool_id'))
            )
        with col3:
            st.button(
                'Remove from agent',
                on_click=remove_from_agent,
                key="remove_{tool_id}".format(tool_id=tool.get('tool_id')),
                args=[tool])
        st.markdown(
            tool.get('description')
        )
        st.divider()

with st.expander('Code Snippet'):
    # Generated Code Snippet
    selected_tools = [tool.get('tool_id')
                      for tool in st.session_state['selected_tools']]
    st.code(f'''from xentropy import Tool
import autogen
import os

api_key = os.environ.get('XENTROPY_API_KEY')
# Only needed if you opt for AZURE OPENAI
os.environ['AZURE_OPENAI_API_KEY'] = 'YOUR AZURE OPENAI API KEY'
os.environ['AZURE_OPENAI_API_BASE'] = 'YOUR AZURE OPENAI API BASE'

tool_name = {selected_tools}

tools = [Tool.load(key, api_key=api_key) for key in tool_name]

llm_config = {{
    "functions": [
        {{
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_model_schema(),
        }} for tool in tools
    ],
    "config_list": autogen.config_list_from_models(
        model_list=["gpt-35"], # or your choice of LLM
    ),
    "request_timeout": 120,
}}

chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message="Use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "") == ""
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    function_map={{
        tool.name:tool.run for tool in tools
    }}
)

user_proxy.initiate_chat(
    chatbot,
    message="YOUR MESSAGE",
)
''', language='python')


st.markdown('## Agent')
selected_tools = [tool.get('tool_id')
                  for tool in st.session_state['selected_tools']]

# tools equipped to agents
if not st.session_state.get('tools'):
    st.session_state['tools'] = []
if [tool.name for tool in st.session_state.get('tools')] != selected_tools:
    st.session_state['tools'] = [
        Tool.load(key, api_key=api_key) for key in selected_tools]

llm_config = {
    "functions": [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_model_schema(),
        } for tool in st.session_state['tools']
    ],
    "config_list": config_list,
    "request_timeout": 120,
}


def termination_msg(x):
    return x.get('role') == 'user' and x.get('content') != None


# Construct / Update the LLM Agent
if not st.session_state.get('user_proxy'):
    st.session_state['user_proxy'] = autogen.UserProxyAgent(
        name="user_proxy",
        llm_config=llm_config,
        is_termination_msg=termination_msg,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        system_message="Reply TERMINATE when the task is done.",
        default_auto_reply="",
        function_map={
            tool.name: tool.run for tool in st.session_state['tools']
        }
    )
    st.session_state['chatbot'] = autogen.AssistantAgent(
        name="chatbot",
        system_message="Use the functions you have been provided with to complete the task. Reply TERMINATE when the task is done.",
        llm_config=llm_config,
        is_termination_msg=lambda x: x.get('content', '') == ''
    )

# update agent tool set
if list(st.session_state.get('user_proxy').function_map.keys()) != selected_tools:
    st.session_state['user_proxy'] = autogen.UserProxyAgent(
        name="user_proxy",
        llm_config=llm_config,
        is_termination_msg=termination_msg,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        system_message="Reply TERMINATE when the task is done.",
        default_auto_reply="",
        function_map={
            tool.name: tool.run for tool in st.session_state['tools']
        }
    )
    st.session_state['chatbot'] = autogen.AssistantAgent(
        name="chatbot",
        system_message="Use the functions you have been provided with to complete the task. Reply TERMINATE when the task is done.",
        llm_config=llm_config,
        is_termination_msg=lambda x: x.get('content', '') == ''
    )

# Message Display
for msg in st.session_state['chatbot'].chat_messages.get(st.session_state['user_proxy'], []):
    if msg['content'] == "" or msg['content'] == "TERMINATE":
        continue
    if msg['role'] == 'assistant' and msg.get('function_call'):
        with st.chat_message(msg['role']):
            st.write("Tool Use: {tool_name}".format(
                tool_name=msg['function_call']['name'])
            )
            st.write("Arguments: {args}".format(
                args=msg['function_call']['arguments'])
            )
    elif msg['role'] == 'function':
        st.chat_message(msg['name']).write(msg['content'])
    else:
        st.chat_message(msg['role']).write(msg['content'])

# Message Input
if prompt := st.chat_input():
    print(st.session_state['user_proxy'].llm_config)

    if not api_key:
        st.info("Please add your XEntropy API KEY to continue.")
        st.stop()

    # current number of messages
    index = len(
        st.session_state['chatbot'].chat_messages.get(st.session_state['user_proxy'], []))

    st.session_state['user_proxy'].initiate_chat(
        st.session_state['chatbot'],
        message=prompt,
        llm_config=llm_config,
        clear_history=False,
    )

    for msg in st.session_state['chatbot'].chat_messages.get(st.session_state['user_proxy'], [])[index:]:
        if msg['content'] == "" or msg['content'] == "TERMINATE":
            continue
        if msg['role'] == 'assistant' and msg.get('function_call'):
            with st.chat_message(msg['role']):
                st.write("Tool Use: {tool_name}".format(
                    tool_name=msg['function_call']['name'])
                )
                st.write("Arguments: {args}".format(
                    args=msg['function_call']['arguments'])
                )
        elif msg['role'] == 'function':
            st.chat_message(msg['name']).write(msg['content'])
        else:
            st.chat_message(msg['role']).write(msg['content'])
