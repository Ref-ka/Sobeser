import operator
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    interview_plan: str
    internal_thoughts: Annotated[List[str], operator.add]
    candidate_profile: dict
    observer_instructions: str
    current_agent_response: str
    last_agent_visible_message: str
    difficulty_level: int
    is_finished: bool
    turn_count: int

