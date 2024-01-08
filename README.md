# AgentX

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

XEntropy-SDK enables developers to build powerful LLM (Large Language Model) agents fast. LLM agents uses LLM as a controlling element to perform a wide range of tasks that require context awareness and reasoning. These agents are typically characterized by 4 components:
1. Large language model
2. Memory (chat and tool use history)
3. Planning (a way to plan and execute actions to reach a goal)
4. Tools (arbitrary sensors, actuators, and other LLM agents)

Most LLM agent libraries implement the memory and planning components, but the tool component is often left out. This is where XEntropy comes in as a platform for developers to share tools they built and load existing tools to speed up development.

For more information about LLM agents I highly recommend reading this [blog](https://lilianweng.github.io/posts/2023-06-23-agent/) by Lilian Weng.

## Features

* Load tools to your LLM agents
* Publish your tools on the platform
* A* search for automated planning and execution of tools orchestrated with a set of agents
* Prompt optimisation (on roadmap)
* Inference parameter optimisation (on roadmap)
* Supported Models
  * ChatGPT-3.5-Turbo
  * GPT-4
  * GPT-4-Vision
  * ToolLlama-7B
  * Gemini-Pro (on roadmap)
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

Refer to the [docs](https://github.com/xentropy-ai/agentx/docs) for advanced usage and full documentation of all the features of XEntropy-SDK.

## Contribution

If you want to contribute, feel free to fork the repository and submit pull requests. If you found any bugs or have suggestions, please create an issue in the [issues](https://github.com/xentropy-ai/agentx/issues) section.

## Contact

For questions or feedback, you can reach out to us at chankahei@xentropy.co.

---

Cheers, 
XEntropy Team