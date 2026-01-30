from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import logging
from state import AgentState


logger = logging.getLogger(__name__)


def interviewer_node(state: AgentState, interviewer_agent) -> dict:
    messages = state.get('messages', [])
    internal_thoughts = state.get('internal_thoughts', [])
    profile = state.get('candidate_profile', {})
    interview_plan = state.get('interview_plan', 'Нет плана интервью')
    difficulty_level = state.get('difficulty_level', 2)

    last_observer_thought = ""
    if internal_thoughts:
        for thought in reversed(internal_thoughts):
            if "[Observer]" in thought:
                last_observer_thought = thought
                break

    context_prompt = f"""ПРОФИЛЬ КАНДИДАТА:
    - Позиция: {profile.get('role', 'Не указано')}
    - Уровень: {profile.get('grade', 'Не указано')}
    - Опыт: {profile.get('exp', 'Не указано')}

    Текущий уровень сложности вопросов (1-5): {difficulty_level}
    
    План проведения интервью:
    {interview_plan}
    
    Мысли наблюдателя интервью:
    {last_observer_thought}"""
    
    try:
        agent_messages = [
            SystemMessage(content=context_prompt)
        ]

        if messages:
            recent_messages = messages[-6:] if len(messages) > 6 else messages
            agent_messages.extend(recent_messages)

        agent_result = interviewer_agent.invoke({"messages": agent_messages})

        agent_messages_result = agent_result.get("messages", [])
        response_content = ""
        response_message = None
        
        if agent_messages_result:
            for msg in reversed(agent_messages_result):
                if isinstance(msg, AIMessage):
                    response_content = msg.content
                    response_message = msg
                    break
        
        if not response_content:
            response_content = "Извините, не удалось сформулировать вопрос."
            response_message = AIMessage(content=response_content)
        
        logger.debug(f"Interviewer response generated: {len(response_content)} characters")
        
        return {
            "messages": [response_message],
            "current_agent_response": response_content,
        }
    except Exception as e:
        logger.exception(f"Interviewer error: {e}")
        error_msg = "Извините, произошла техническая ошибка. Пожалуйста, повторите ваш ответ."
        return {
            "messages": [AIMessage(content=error_msg)],
            "current_agent_response": error_msg
        }

