from agent import Agent
import ray

@ray.remote
class RayAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)