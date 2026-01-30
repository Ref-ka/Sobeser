from langchain_core.messages import HumanMessage
from state import AgentState


def route_before_observer(state: AgentState) -> str:
    if not state.get('interview_plan'):
        return "planner"
    return "observer"


def route_after_observer(state: AgentState) -> str:
    if state.get("is_finished"):
        return "manager"

    if not state.get('messages'):
        return "interviewer"

    last_user_msg = None
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content.lower().strip()
            break

    if last_user_msg:
        stop_keywords = [
            "стоп", "фидбэк", "хватит", "заверши", "закончи",
            "стоп игра", "давай фидбэк", "хочу завершить",
            "завершить интервью", "закончить интервью", "стоп интервью"
        ]

        if any(keyword in last_user_msg.lower() for keyword in stop_keywords):
            return "manager"

    return "interviewer"

