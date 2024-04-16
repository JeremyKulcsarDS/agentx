from siumai.agent import Agent


class GradientAgent(Agent):

    def __init__(self, **kwargs):
        super().__init__(
            name='gradient_agent',
            system_prompt='''You are a prompt engineer.
Review the current prompt and error samples, and then analysis why the prompt have gotten these examples wrong.
Think step by step.''',
            **kwargs
        )


class BackpropAgent(Agent):

    def __init__(self, num_prompts:int=5, **kwargs):
        system_prompt = '''You are a prompt engineer.
Review the given prompt, error samples and feedback.

Based on the finding, write {num_prompts} new prompts that address the problems in the current prompt.
'''.format(num_prompts=num_prompts)
        
        super().__init__(
            name='backprop_agent',
            system_prompt=system_prompt,
            **kwargs
        )
