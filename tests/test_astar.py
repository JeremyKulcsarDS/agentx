import unittest, os
from dotenv import load_dotenv
from agentx.tool import Tool
from agentx.agent import Agent
from agentx.schema import GenerationConfig, Function, Message, Content
from agentx.groupchat import astarchat
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

class HeuristicScore(BaseModel):
    score:float = Field(0, ge=0, le=10)

class AStarChatTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Get a tool that converts address to latitude longitude coordinate
        self.geocoding = Tool.load('xentropy--geocoding', api_key=os.environ.get('XENTROPY_API_KEY'))
        # Get a tool that computes the earth surface distance between two coordinates
        self.geodesic = Tool.load('xentropy--geodesic', api_key=os.environ.get('XENTROPY_API_KEY'))
        
        self.geocoding_agent = Agent(
            name='geocoding_agent',
            system_prompt='Use the functions you have been provided to solve the problem. Reply TERMINATE to end the conversation when the problem is solved.',
            generation_config=GenerationConfig(
                api_type='azure',
                api_key=os.environ.get('AZURE_OPENAI_KEY'),
                base_url=os.environ.get('AZURE_OPENAI_ENDPOINT'),
                azure_deployment='gpt-35',
                tools={
                    self.geocoding.name:Function(
                        name=self.geocoding.name,
                        description=self.geocoding.description,
                        parameters=self.geocoding.input_json_schema,
                    )
                }
            ),
            function_map={
                self.geocoding.name: self.geocoding.run
            },
            reduce_function=lambda x: x[-1],
        )

        self.geodesic_calculation_agent = Agent(
            name = 'geodesic_calculation_agent',
            system_prompt='Use the functions you have been provided to solve the problem. Reply TERMINATE to end the conversation when the problem is solved.',
            generation_config=GenerationConfig(
                api_type='azure',
                api_key=os.environ.get('AZURE_OPENAI_KEY'),
                base_url=os.environ.get('AZURE_OPENAI_ENDPOINT'),
                azure_deployment='gpt-35',
                tools={
                    self.geodesic.name:Function(
                        name=self.geodesic.name,
                        description=self.geodesic.description,
                        parameters=self.geodesic.input_json_schema,
                    )
                }
            ),
            function_map={
                self.geodesic.name: self.geodesic.run
            },
            reduce_function=lambda x: x[-1],
        )

        self.heuristic_agent = Agent(
            name='heuristic',
            generation_config=GenerationConfig(
                api_type='azure',
                api_key=os.environ.get('AZURE_OPENAI_KEY'),
                base_url=os.environ.get('AZURE_OPENAI_ENDPOINT'),
                azure_deployment='gpt-35',
            ),
        )

        # At each timestep, A* minimize heuristic + cost
        # heuristic is an estimation of the distance between the current state and the goal state
        # cost is the distance between the start state and the current state

        # In this example, heuristic is whether the geodesic distance is calculated correctly
        def heuristic(messages:List[Message]) -> float:
            score = self.heuristic_agent.generate_response(
                [
                    Message(
                        role='user',
                        content=Content(
                            text='Base on this chat history: {history}'.format(
                                history=[message.model_dump_json(
                                    exclude_unset=True, 
                                    exclude_none=True
                                ) for message in messages]
                            ),
                        ),
                    )
                ] + [
                    Message(
                        role='user',
                        content=Content(
                            text='Estimate the progress of solving the problem. Give a score of 10 to represent that the geodesic distance is calculated correctly. You must reply an JSON object.',
                        ),
                    )
                ],
                output_model=HeuristicScore # the output model is used to validate the response
            )
            return 10 - HeuristicScore.model_validate_json(score.content.text).score

        self.heuristic = heuristic

        # Cost is the number of LLM calls
        def cost(messages:List[Message], next_message:List[Message]) -> float:
            # tool calls are not counted
            cost = sum([
                1 for message in next_message if message.role != 'tool'
            ])
            return cost
        
        self.cost = cost

    async def test_astar(self):
        reconstructed_path, came_from, cost_so_far, hash_map = await astarchat(
            agents = [self.geocoding_agent, self.geodesic_calculation_agent],
            heuristic = self.heuristic,
            cost = self.cost,
            messages = [
                Message(
                    role='user',
                    content=Content(
                        text='What is the distance between Gare Port La Goulette - Sud in Tunisia and Porto di Napoli in Italy?',
                    ),
                )
            ],
            threshold = 1,
            n_replies = 1,
            max_iteration = 10,
            max_queue_size = 10
        )

        print(reconstructed_path)
        print(came_from)
        print(cost_so_far)
        print(hash_map)