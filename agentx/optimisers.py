import asyncio
import random
import copy
from agentx.agent import Agent
from agentx.schema import Message, Content, GenerationConfig
from agentx.saved_agents import GradientAgent, BackpropAgent
from typing import Awaitable, Dict, List, Tuple, Callable, Union, Any, TypeVar
from math import ceil
from statistics import mean
from pydantic import BaseModel
from tqdm import tqdm

InputType = TypeVar('InputType', bound=BaseModel)
PredictType = TypeVar('PredictType', bound=BaseModel)
TruthType = TypeVar('TruthType', bound=BaseModel)

class PromptSuggestion(BaseModel):
    reason: str
    prompt: str

class PromptSuggestions(BaseModel):
    suggestions: List[PromptSuggestion]

class TextualGradientPromptTrainer():
    def __init__(
        self,
        generation_config: Union[GenerationConfig, None],
        agent: Agent,
        forward: Callable[
            [
                Agent,
                InputType,
            ],
            Awaitable[PredictType]
        ],
        loss: Callable[
            [
                PredictType,
                TruthType
            ],
            Union[float, Awaitable[float]]
        ],
        target: str='agent',
        batch_size: int = 50,
        n_beam: int = 4,
        n_sample: int = 10,
        budget: int = 50,
        concurrency: int = 5,
    ):
        """
        Apply textual gradient descent to optimise the system prompt of an Agent or the description of a Tool
        An implementation of: https://arxiv.org/pdf/2005.00928.pdf
        
        :param agent: The agent to be optimized.
        :type agent: Agent
        :param target: The target to optimize towards. Can be None if optimizing the agent's system prompt. Otherwise, it should be the name of the tool to optimize.
        :type target: Union[str, None]
        :param forward: The forward function that takes in an agent and an input and returns a prediction.
        :type forward: Callable[[Agent, InputType], PredictType]
        :param numeric_loss: The loss function that takes in a prediction and a ground truth and returns a float value representing the loss score.
        :type numeric_loss: Callable[[PredictType, TruthType], float]
        :param generation_config: The generation configuration for the gradient agent and backprop agent.
        :type generation_config: GenerationConfig
        :param n_beam: The number of beams to use in the gradient descent process. Default is 5.
        :type n_beam: int
        :param batch_size: The batch size to use in the gradient descent process. Default is 5.
        :type batch_size: int
        :param max_sample: The maximum number of samples to generate in each iteration of the gradient descent process. Default is 5.
        :type max_sample: int
        :param budget: The maximum number of iterations to perform in the gradient descent process. Default is 25.
        :type budget: int
        :param concurrency: The maximum number of concurrent call for forward(). Default is 5.
        :type concurrency: int
        """
        self.agent = agent
        self.forward = forward
        self.loss = loss
        self.target = target
        self.batch_size = batch_size
        self.n_beam = n_beam
        self.n_sample = n_sample
        self.budget = budget
        self.concurrency = concurrency
        self.gradient_agent = GradientAgent(generation_config=generation_config)
        self.backprop_agent = BackpropAgent(generation_config=generation_config, num_prompts=n_sample)
    

    async def textual_gradient_descent(
        self,
        prompt:str,
        x:List[InputType],
        predict:List[Union[PredictType, None]],
        y:List[TruthType],
    ) -> List[str]:
        # compute the textual gradient
        textual_errors = [
            {
                'input': i.model_dump(),
                'predict': p.model_dump(),
                'truth': t.model_dump(),
            } for i, p, t in zip(x, predict, y) if p != None
        ]

        loss = [self.loss(p, t) for p, t in zip(predict, y) if p != None]
        if asyncio.iscoroutine(loss[-1]):
            loss = asyncio.gather(*loss)

        # include only the largests errors
        largest_errors = sorted(
            zip(textual_errors, loss),
            key=lambda x: x[1],
            reverse=True
        )[:self.n_sample]

        messages = [
            Message(
                role='user',
                content=Content(
                    text='''Current prompt: {prompt}

    Errors: {errors}'''.format(prompt=prompt, errors=largest_errors)
                ),
            )
        ]

        response = await self.gradient_agent.a_generate_response(
            messages=messages
        )

        # compute the textual descent to get new prompts
        new_prompts = await self.backprop_agent.a_generate_response(
            messages=messages+response,
            output_model=PromptSuggestions,
        )

        return [
            suggestion.prompt for suggestion in PromptSuggestions.model_validate_json(
                new_prompts[-1].content.text
            ).suggestions
        ]


    async def _forward(
        self,
        agent:Agent,
        x:List[InputType],
    ) -> List[PredictType]:
        # forward in batches with concurrency
        
        predict:List[PredictType] = []
        # generate responses for each data point
        for index in range(0, self.batch_size, self.concurrency):
            batch = await asyncio.gather(*[
                self.forward(agent, x[i]) for i in range(index, index+self.concurrency)
            ])
            predict.extend(batch)
            # avoid rate limiting error
            await asyncio.sleep(5)
        
        return predict


    async def expand(
        self,
        prompt:str,
        x:List[InputType],
        y:List[TruthType],
    ) -> List[str]:

        _agent:Agent = copy.copy(self.agent)
        if self.target == 'agent':
            _agent.system_prompt = prompt
        else:
            _agent.generation_config.tools[self.target].description = prompt

        predict = await self._forward(_agent, x)

        # calculate the textual gradient, i.e. identify problems which the agent made mistakes on
        new_prompts = await self.textual_gradient_descent(prompt, x, predict, y)

        return new_prompts
    

    async def select(
        self,
        prompts:List[str],
        x: List[InputType],
        y: List[TruthType],
    ) -> Tuple[List[str], Dict[str, float]]:
        # implement successive rejects
        K = len(prompts) - self.n_beam
        log_bar_K = 0.5 + sum([1.0/i for i in range(2, K+1)])

        scores = {
            prompt:0 for prompt in prompts
        }

        prompts_remained = copy.deepcopy(prompts)

        for i in tqdm(range(K), desc='Selecting Prompts'):
            # calculate the number of samples for the current round
            n_k = ((self.budget - K) / (K - i)) / log_bar_K
            n_samples_per_round = ceil(n_k)
            # impose a maximum number of samples per round
            n_samples_per_round = max(self.n_sample, n_samples_per_round)
            # sample data
            sample_indices = random.sample(range(len(x)), n_samples_per_round)
            sampled_x = [x[i] for i in sample_indices]
            sampled_y = [y[i] for i in sample_indices]
            # compute the loss for the expanded prompts
            for prompt in prompts_remained:
                _agent = copy.copy(self.agent)
                if self.target == 'agent':
                    _agent.system_prompt = prompt
                else:
                    _agent.generation_config.tools[self.target].description = prompt

                predict = await self._forward(_agent, sampled_x)

                loss = [self.loss(p, t) for p, t in zip(predict, sampled_y) if p != None]
                if asyncio.iscoroutine(loss[-1]):
                    loss = asyncio.gather(*loss)
                loss = mean(loss)

                scores[prompt] += loss
            # find and remove the lowest scoring prompt
            prompts_remained.sort(key=lambda prompt: scores[prompt], reverse=True)
            prompts_remained.pop()

        return prompts_remained, scores


    async def fit(
        self,
        x:List[InputType],
        y:List[TruthType],
        n_training_steps:int=5, 
        initial_prompts:Union[List[str], None]=None
    ) -> Tuple[List[str], Dict[str, float]]:
        prompts = initial_prompts
        if prompts == None:
            if self.target == 'agent':
                prompts = [self.agent.system_prompt]
            else:
                prompts = [self.agent.generation_config.tools[self.target].description]

        scores_log = []
        for i in tqdm(range(n_training_steps), desc='Training Step'):
            i = i % (len(x) // self.batch_size)
            # sample a mini batch of data
            batch_x = x[i*self.batch_size:(i+1)*self.batch_size]
            batch_y = y[i*self.batch_size:(i+1)*self.batch_size]
            
            # expand
            expanded_prompts = []
            for prompt in tqdm(prompts, desc='Expanding Prompts'):
                expanded_prompts.extend(await self.expand(prompt, x=batch_x, y=batch_y))
            # select
            prompts_remained, scores = await self.select(expanded_prompts, x=x, y=y)
            scores_log.append(scores)
            prompts = prompts_remained

        return prompts, scores_log
    