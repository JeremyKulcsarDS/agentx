from autogen import ConversableAgent
from typing import Callable, List, Dict, Optional
import queue

# WORK IN PROGRESS
# Implement the A* algorithm for conversable agents


class AStarChat():
    def __init__(self, agents, cost, heuristic):
        # first agent is assumed to initiate the conversation
        self.agents: List[ConversableAgent] = agents
        self.cost: Callable = cost
        self.heuristic: Callable = heuristic

    def initiate_chat(self, prompt: List[Dict], threshold: float, n_sample: int):
        frontier = queue.PriorityQueue()
        frontier.put(prompt, 0)
        came_from: dict[List[Dict], Optional[List[Dict]]] = {}
        cost_so_far: dict[List[Dict], float] = {}
        came_from[prompt] = None
        cost_so_far[prompt] = 0

        while not frontier.empty():
            current_messages: List[Dict] = frontier.get()

            # check if termination criteria is met
            if self.heuristic(current_messages) < threshold:
                break

            # generate the next message
            # TODO parallelism
            generated_messages = [agent.generate_reply(
                current_messages) for agent in self.agents]  # need type casting

            for next in generated_messages:
                new_cost = cost_so_far[current_messages] + \
                    self.cost(current_messages, next)
                if next not in cost_so_far.keys() or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(next)
                    frontier.put(next, priority)
                    came_from[next] = current_messages

        return came_from, cost_so_far
