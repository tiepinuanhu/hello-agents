import re
from typing import Optional, List, Tuple, Dict
from hello_agents import PlanAndSolveAgent, HelloAgentsLLM, Config, Message, ToolRegistry


# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""

class MyPlanAndSolveAgent(PlanAndSolveAgent):
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
    ):
        # 使用默认提示词模板，或者从配置中获取自定义模板
        
        super().__init__(name, llm, tool_registry)
        self.custom_prompts = {"planner": DEFAULT_PLANNER_PROMPT, "executor": DEFAULT_EXECUTOR_PROMPT} \
            if custom_prompts is None else custom_prompts  