import random
import copy
from agentx.schema import Message
from agentx.agent import Agent
from typing import Dict, List, Tuple, Callable, Union, Any
from functools import reduce, partial
from math import ceil
from pydantic import BaseModel

class TextualGradientPromptTrainer():
    def __init__(
        self,
        agent: Agent,
        target: str,
        output_model:Union[BaseModel, None],
        textual_gradient: Callable[[str, Union[List[Message], List[List[Message]]], Message], Message],
        step_function: Callable[[str, Message], List[Message]],
        paraphrase_function: Callable[[Message], List[Message]],
        selection_metric: Callable[[Union[List[Message], List[List[Message]]], Any], float],
        n_beam: int = 5,
        batch_size: int = 5,
        max_sample: int = 5,
        budget: int = 25,
    ):
        """
        Apply textual gradient descent to optimise the system prompt of an Agent or the description of a Tool
        An implmentation of: https://arxiv.org/pdf/2005.00928.pdf
        """
        self.agent = agent
        self.target = target
        self.output_model = output_model
        self.textual_gradient = textual_gradient
        self.step_function = step_function
        self.paraphrase_function = paraphrase_function
        self.selection_metric = selection_metric
        self.budget = budget
        self.max_sample = max_sample
        self.batch_size = batch_size
        self.n_beam = n_beam

    def generate(self, agent:Agent, messages:List[Message], output_model:Union[BaseModel, None]):
        response = agent.generate_response(messages=messages, output_model=output_model)
        return output_model.model_validate(response.content.text) if output_model else response
    
    def expand(
        self,
        prompt:str,
        x:List[List[Message]],
        y:List,
    ) -> List[str]:

        _agent = copy.deepcopy(self.agent)
        if self.target == 'agent':
            _agent.system_prompt = prompt
        else:
            _agent.generation_config.tools[self.target].description = prompt

        # generate responses for each data point
        responses = [
            self.generate(
                agent=_agent,
                messages=datum,
                output_model=self.output_model
            ) for datum in x
        ]

        # calculate the textual gradient, i.e. identify problems which the agent made mistakes on
        textual_gradient = self.textual_gradient(prompt, responses, y)

        # generate new prompts based on the textual gradient
        new_prompts = self.step_function(prompt, textual_gradient)

        # paraphrase new prompts
        paraphrased_prompts = [self.paraphrase_function(new_prompt) for new_prompt in new_prompts]

        flattened_expansion = [message for sublist in paraphrased_prompts for message in sublist]

        return  [
            message.content.text for message in flattened_expansion + new_prompts
        ]
    
    def compute_metric(
        self,
        prompt:str,
        x:List[List[Message]],
        y:List,
    ) -> float:
        _agent = copy.deepcopy(self.agent)
        if self.target == 'agent':
            _agent.system_prompt = prompt
        else:
            _agent.generation_config.tools[self.target].description = prompt
        
        # generate responses for each data point
        responses = [
            _agent.generate_response(messages=datum, output_model=self.output_model) for datum in x
        ]

        metric = sum(
            [
                self.selection_metric(response, datum) for response, datum in zip(responses, y)
            ]
        )

        return metric

    def select(
        self,
        prompts:List[str],
        data:List[Tuple[List[Message], Union[Message, List[Message]]]], 
    ) -> Tuple[List[str], Dict[str, float]]:
        # implement successive rejects
        K = len(prompts) - self.n_beam
        log_bar_K = 0.5 + sum([1.0/i for i in range(2, K+1)])

        scores = {
            prompt:0 for prompt in prompts
        }

        prompts_remained = copy.deepcopy(prompts)

        for i in range(K):
            # calculate the number of samples for the current round
            n_k = ((self.budget - K) / (K - i)) / log_bar_K
            n_samples_per_round = ceil(n_k)
            # impose a maximum number of samples per round
            n_samples_per_round = max(self.max_sample, n_samples_per_round)
            # sample data
            sampled_data = random.sample(data, n_samples_per_round)
            # compute the metric for the expanded prompts
            for prompt in prompts_remained:
                scores[prompt] += self.compute_metric(prompt, sampled_data)
            # find and remove the lowest scoring prompt
            prompts_remained.sort(key=lambda prompt: scores[prompt], reverse=True)
            prompts_remained.pop()

        return prompts_remained, scores

    def fit(
        self,
        x:List[List[Message]],
        y:List,
        n_training_steps:int, 
        initial_prompts:Union[List[str], None]
    ) -> Tuple[List[str], Dict[str, float]]:
        prompts = initial_prompts
        # expand
        if prompts == None:
            if self.target == 'agent':
                prompts = [self.agent.system_prompt]
            else:
                prompts = [self.agent.generation_config.tools[self.target].description]

        scores_log = []
        for i in range(n_training_steps):
            expand = partial(self.expand, x=x, y=y)
            prompts = reduce(lambda a, b: a + expand(b), prompts, [])
            # select
            prompts_remained, scores = self.select(prompts, x=x, y=y)
            scores_log.append(scores)
            prompts = prompts_remained
        
        return prompts, scores_log
    