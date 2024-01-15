import asyncio
from copy import deepcopy
import queue
from agentx.agent import Agent, Message
from typing import Callable, List, Union, Tuple, Dict
from pydantic import BaseModel
from tqdm import tqdm

class QueueItem(BaseModel):
    priority:float
    messages:List[List[Message]]

    def __lt__(self, other):
        return self.priority < other.priority
    

def reconstruct_path(
    came_from:Dict[Tuple[int], Union[Tuple[int], None]],
    goal:Tuple[int],
    hash_map:Dict[Union[Tuple[int], Tuple[Message]], Union[Tuple[int], Tuple[Message]]]
) -> List[Message]:
    current = goal
    path = []
    # start is None
    while current != None:
        path.extend(list(hash_map[current]))
        current = came_from[current]
    path.reverse()
    return path


async def astar_chat(
    agents: List[Agent],
    messages: List[Message],
    cost: Callable[[List[Message]], float],
    heuristic: Callable[[List[Message]], Union[float, None]],
    threshold: int=10,
    n_replies: int=1,
    max_iteration:int=10,
) -> Tuple[
        List[Message], 
        Dict[Tuple[int], Union[Tuple[int], None]], 
        Dict[Tuple[int], float],
        Dict[Tuple[int], float],
        Dict[Union[Tuple[int], Tuple[Message]], Union[Tuple[int], Tuple[Message]]]
    ]:
    """
    The agents will concurrently generate a response to the messages.
    The best response will be selected based on the heuristic function.

    :param agents: List of agents participating in the conversation
    :param messages: List of messages to start the conversation
    :param cost: Cost function for the current conversation
    :param heuristic: Heuristic function for estimating how far the current conversation is from the goal
    :param starting_priority: Priority of the first message in the frontier queue
    :param threshold: Threshold for the heuristic function
    :param n_replies: Number of replies to generate for each agent
    :param max_iteration: Terminate the search after max_try iterations
    :param max_queue_size: Maximum size of the frontier priority queue
    """
    # Initialize the frontier queue
    frontier = queue.PriorityQueue()

    # Place first message to the frontier queue
    frontier.put(QueueItem(priority=0, messages=[messages]))

    came_from: Dict[Tuple[int], Union[Tuple[int], None]] = {}
    cost_so_far: Dict[Tuple[int], float] = {}
    heuristic_map: Dict[Tuple[int], float] = {}

    first_hash = tuple([hash(message) for message in messages])
    came_from[first_hash] = None
    cost_so_far[first_hash] = 0
    heuristic_map[first_hash] = heuristic(messages)

    hash_map: Dict[Union[Tuple[int], Tuple[Message]], Union[Tuple[int], Tuple[Message]]] = {
        first_hash: tuple(messages),
        tuple(messages): first_hash,
    }

    for current_iteration in tqdm(range(max_iteration)):
        if frontier.empty():
            break
        # Pick the next list of messages for the conversation
        current_messages: List[List[Message]] = frontier.get().messages
        flatten_current_messages: List[Message] = [message for sublist in current_messages for message in sublist]
        
        # Generate a list of responses from the participating agents
        participating_agents = agents

        tool_calls = flatten_current_messages[-1].content.tool_calls
        if tool_calls != None:
            function_names = [tool_call.function_call.name for tool_call in tool_calls]
            participating_agents = [
                agent for agent in agents if set(function_names).issubset(agent.a_function_map.keys())
            ]
        
        tasks = [
            [
                agent.a_generate_response(flatten_current_messages) for i in range(n_replies)
            ] for agent in participating_agents
        ]
        # Flatten the list of tasks
        tasks = [item for sublist in tasks for item in sublist]

        generated_messages:List[Union[Message, List[Message], None]] = await asyncio.gather(*tasks)
        generated_messages:List[Union[Message, List[Message]]] = [message for message in generated_messages if message != None]
        generated_messages:List[List[Message]] = [message if isinstance(message, List) else [message] for message in generated_messages]

        for next in generated_messages:
            # calculate the cost of the new message
            new_cost:float = cost_so_far[hash_map[tuple(current_messages[-1])]] + cost(flatten_current_messages, next)
            hash_next_message = tuple([hash(message) for message in next])
            previous_cost = cost_so_far.get(hash_next_message, None)
            # the message is never seen before or the new cost is less than the previous cost
            if previous_cost == None or new_cost < previous_cost:
                cost_so_far[hash_next_message] = new_cost
                came_from[hash_next_message] = hash_map[tuple(current_messages[-1])]
                hash_map[hash_next_message] = tuple(next)
                hash_map[tuple(next)] = hash_next_message
                heuristic_score = heuristic(flatten_current_messages + next)
                # if heuristic score is None, use the heuristic score of the previous message
                if heuristic_score == None:
                    heuristic_score = heuristic_map[hash_map[tuple(current_messages[-1])]]
                heuristic_map[hash_next_message] = heuristic_score
                priority = new_cost + heuristic_score

                # Add the new item to the frontier
                new_item = QueueItem(priority=priority, messages=current_messages + [next])
                frontier.put(new_item)

                if heuristic_score < threshold:
                    reconstructed_path:List[Message] = reconstruct_path(
                        came_from=came_from,
                        goal=hash_next_message,
                        hash_map=hash_map
                    )
                    return reconstructed_path, came_from, cost_so_far, heuristic_map, hash_map 

    goal = min(heuristic_map, key=heuristic_map.get)
    reconstructed_path:List[Message] = reconstruct_path(
        came_from=came_from,
        goal=goal,
        hash_map=hash_map,
    )

    return reconstructed_path, came_from, cost_so_far, heuristic_map, hash_map


async def group_chat(
    agents: List[Agent],
    messages: List[Message],
    heuristic: Callable[[List[Message]], Union[float, None]],
    threshold: int=10,
    max_iteration:int=10,
):
    '''
    Start the chat, with the first agent initiating the conversation.
    Each agent in the agents list will take turn in a roundtable to generate a response to the messages.
    '''
    current_messages = deepcopy(messages)
    first_hash = tuple([hash(message) for message in messages])
    heuristic_map: Dict[Tuple[int], float] = {first_hash:heuristic(messages)}
    hash_map: Dict[Union[Tuple[int], Tuple[Message]], Union[Tuple[int], Tuple[Message]]] = {
        first_hash: tuple(messages),
        tuple(messages): first_hash,
    }

    for current_iteration in range(max_iteration):
        # Pick the next agent to generate a response
        agent = agents[current_iteration % len(agents)]
        # Generate a response from the agent
        response:Message = await agent.a_generate_response(current_messages)
        # if the response is a tool_call
        tool_call = response.content.tool_calls
        tool_response:Union[List[Message], None] = None
        if tool_call != None:
            # execute the tool call
            tool_response:List[Message] = await agent.a_generate_response(current_messages)
        
        next = [response] + tool_response if tool_response != None else []
        hash_next_message = tuple([hash(message) for message in next])
        hash_map[hash_next_message] = tuple(next)
        hash_map[tuple(next)] = hash_next_message
        # Check if the conversation is finished
        heuristic_score = heuristic(current_messages + next)
        # if heuristic score is None, use the heuristic score of the previous message
        if heuristic_score == None:
            heuristic_score = heuristic_map[hash_map[tuple(current_messages[-1])]]
            heuristic_map[hash_next_message] = heuristic_score
        
        if heuristic_score < threshold:
            return current_messages.extend(next), heuristic_map, hash_map
    
    return current_messages, heuristic_map, hash_map