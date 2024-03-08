# AgentX

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AgentX enables developers to build powerful LLM (Large Language Model) agents fast. LLM agents uses LLM as a controlling element to perform a wide range of tasks that require context awareness and reasoning. These agents are typically characterized by 4 components:
1. Large language model
2. Memory (chat and tool use history)
3. Planning (a way to plan and execute actions to reach a goal)
4. Tools (arbitrary sensors, actuators, and other LLM agents)

AgentX comes in as a platform for developers to share tools they built and load existing tools to speed up development. It also provides a set of utilities for automated planning and execution of tools orchestrated with a set of agents.

For more information about LLM agents we highly recommend reading this [blog](https://lilianweng.github.io/posts/2023-06-23-agent/) by Lilian Weng.

## Features

* Load tools to your LLM agents
* Publish your tools on the platform
* A* search for automated planning and execution of tools orchestrated with a set of agents
* Prompt optimisation
* Inference parameter optimisation (on roadmap)
* Supported Models
  * ChatGPT-3.5-Turbo
  * GPT-4
  * GPT-4-Vision
  * ToolLlama-7B
  * Gemini-Pro (currently under development -- need to handle flexible multiturns and tools with multiple arguments / with $refs)
  * Gemini-Pro-Vision (on roadmap)
  * Claude-2 (on roadmap)
  * Claude-2.1 (on roadmap)
* Easy integration with a range of LLM agent libraries
  *  [LangChain](https://github.com/langchain-ai/langchain)
  *  [Autogen](https://github.com/microsoft/autogen)
  *  [XAgent](https://github.com/OpenBMB/XAgent)

## Installation

To install the project, clone this repository and copy to your python site-packages directory:

```bash
git clone https://github.com/xentropy-ai/agentx.git
cd agentx
cp -r agentx /usr/local/lib/python3.8/site-packages/ # replace with your own site-packages directory
```

## Documentation

Refer to the [docs](https://github.com/xentropy-ai/agentx/tree/main/docs) for advanced usage and full documentation of all the features of AgentX.

## Contribution

If you want to contribute, feel free to fork the repository and submit pull requests. If you found any bugs or have suggestions, please create an issue in the [issues](https://github.com/xentropy-ai/agentx/issues) section.

## Contact

For questions or feedback, you can reach out to us at chankahei@xentropy.co or jeremy.kulcsar@diamondhill.io.

---

Cheers, 
XEntropy Team
