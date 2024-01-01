import asyncio
import queue
import json
from xentropy.agent import Agent, Message
from typing import Callable, List, Dict, Optional


async def astarchat(
        agents: List[Agent],
        messages: List[Message],
        cost: Callable,
        heuristic: Callable,
        starting_priority: int=100,
        threshold: int=10,
        n_replies: int=1,
        max_iteration:int=10,
        max_queue_size:int=10,
):
    """
    Start the chat, with the first agent initiating the conversation

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

    # Number of turns in the conversation
    current_iteration = 0

    # Initialize the frontier queue
    frontier = queue.PriorityQueue(maxsize=max_queue_size)

    # Place first message to the frontier queue
    frontier.put((starting_priority, messages))

    came_from: dict[str, Optional[Dict]] = {}
    cost_so_far: dict[str, int] = {}

    came_from[json.dumps(messages)] = None
    cost_so_far[json.dumps(messages)] = 0

    while (not frontier.empty()) and (current_iteration < max_iteration):
        current_iteration += 1
        # Pick the next set of messages for the conversation
        current_messages: List[Dict] = frontier.get()[1]

        # Generate a list of responses from the participating agents
        participating_agents = agents

        function_call = current_messages[-1].get('function_call', None)
        if function_call != None:
            participating_agents = [agent for agent in agents if agent.function_map.get(function_call.get('name'))]
        
        tasks = [
            [agent.a_generate_reply(current_messages) for i in range(n_replies)] for agent in participating_agents
        ]
        # Flatten the list of tasks
        tasks = [item for sublist in tasks for item in sublist]

        generated_messages = await asyncio.gather(*tasks)
        generated_messages = [message for message in generated_messages if message != None]
        
        for message in generated_messages:
            message.pop('tool_calls', None)

        for next in generated_messages:
            if next == None:
                continue
            if isinstance(next, str):
                next = {'content': next, "role": "assistant"}
            # calculate the cost of the new message
            new_cost = cost_so_far[json.dumps(current_messages)] + cost(current_messages, next)
            previous_cost = cost_so_far.get(json.dumps(current_messages + [next]), None)
            # the message is never seen before or the new cost is less than the previous cost
            if previous_cost == None or new_cost < previous_cost:
                cost_so_far[json.dumps(current_messages + [next])] = new_cost
                print(current_messages + [next])
                heuristic_score = heuristic(current_messages + [next])
                priority = new_cost + heuristic_score
                # Add the new message to the frontier
                frontier.put((priority, current_messages + [next]))
                came_from[json.dumps(current_messages + [next])] = current_messages
                if heuristic_score < threshold:
                    return came_from, cost_so_far

    return came_from, cost_so_far