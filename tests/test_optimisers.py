import json
from pandas import read_parquet
from typing import List, Union
import unittest
import os
from random import sample
from pydantic import BaseModel
from dotenv import load_dotenv
from siumai.schema import GenerationConfig, Message, Content
from siumai.agent import Agent
from siumai.optimisers import TextualGradientPromptTrainer

load_dotenv()

class JobPost(BaseModel):
    title: str
    short_description: str
    skill_set: List[str]
    location: Union[str, None]
    formatted_experience_level: Union[str, None]

# model the output
class SalaryPrediction(BaseModel):
    reasons: str
    salary: int

class TrueSalary(BaseModel):
    salary: int

class TextualGradientPromptTrainerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        dataset = read_parquet(
            '~/siumai/docs/data/job_posting_2023.parquet'
        )

        dataset.skill_set = dataset.skill_set.apply(lambda x: json.loads(x))
        x = dataset[
            [
                'title', 
                'short_description',
                'skill_set',
                'location',
                'formatted_experience_level',
            ]
        ].to_dict(orient='records')

        y = dataset['salary'].to_list()

        # Split the dataset into training and testing set

        test_ids = sample(range(len(x)), 50)
        self.x_train = [JobPost(**x[i]) for i in range(len(x)) if i not in test_ids]
        self.x_test = [JobPost(**x[i]) for i in range(len(x)) if i in test_ids]
        self.y_train = [TrueSalary(salary=int(y[i])) for i in range(len(y)) if i not in test_ids]
        self.y_test = [TrueSalary(salary=int(y[i])) for i in range(len(y)) if i in test_ids]

        generation_config = GenerationConfig(
            api_type='azure',
            api_key=os.environ.get('AZURE_OPENAI_KEY'),
            base_url=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            azure_deployment='gpt-35',
        )

        salary_prediction_agent = Agent(
            name='salary_prediction',
            generation_config=generation_config,
            system_prompt='''You will predict the salary of a job posting based on the job title, description, location, and experience level.
        Output JSON format only.''',
        )

        async def forward(agent:Agent, input:JobPost) -> Union[SalaryPrediction, None]:
            response = await agent.a_generate_response(
                messages=[
                    Message(
                        role='user',
                        content=Content(
                            text=input.model_dump_json(),
                        )
                    )
                ],
                output_model=SalaryPrediction,
            )
            return SalaryPrediction.model_validate_json(response[0].content.text) if response else None
        
        def loss(predicted:Union[SalaryPrediction, None], truth:TrueSalary) -> float:
            if not predicted:
                return None

            l1 = abs(predicted.salary - truth.salary)
            return l1
        
        self.trainer = TextualGradientPromptTrainer(
            agent=salary_prediction_agent,
            generation_config=generation_config,
            forward=forward,
            loss=loss,
            batch_size=10,
            n_beam=4,
            n_sample=5,
            budget=50
        )

    async def test_train(self):
        result = await self.trainer.fit(
            x=self.x_train,
            y=self.y_train,
            n_training_steps=2,
        )

        print(result)