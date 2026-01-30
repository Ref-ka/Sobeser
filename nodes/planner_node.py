from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging
from state import AgentState


logger = logging.getLogger(__name__)

def planner_node(state: AgentState, planner_agent) -> dict:
    if not state.get('messages'):
        last_user_msg = ""
    else:
        last_user_msg = state['messages'][-1].content
    profile = state.get('candidate_profile', {})

    context_prompt = f"""Профиль кандидата:
    - Имя: {profile.get('name', 'Не указано')}
    - Позиция: {profile.get('role', 'Не указано')}
    - Заявленный уровень: {profile.get('grade', 'Не указано')}
    - Опыт: {profile.get('exp', 'Не указано')}

    Самый первый ответ кандидата:
    "{last_user_msg}"
    """

    try:
        # Создаем сообщения для агента
        messages = [
            SystemMessage(content=context_prompt),
            HumanMessage(content="Составь план интервью на основе предоставленной информации о кандидате."
                                 "Если доступен веб-поиск, используй инструмент `tavily_search`."
                                 "На забывай контролировать размер плана интервью, не увеличивай его излишне.")
        ]

        # Вызываем агента через его граф
        agent_result = planner_agent.invoke({"messages": messages})

        # Извлекаем ответ агента
        # Агент возвращает граф с сообщениями, нужно найти последний ответ
        agent_messages = agent_result.get("messages", [])
        plan = ""
        if agent_messages:
            # Ищем последнее сообщение от агента
            for msg in reversed(agent_messages):
                if isinstance(msg, AIMessage):
                    plan = msg.content
                    break

        if not plan:
            plan = "Не удалось сгенерировать план интервью."

        logger.debug(f"Planner response generated: {len(plan)} characters")

        # Преобразуем план в строку (убираем переносы строк для компактности)
        plan_str = plan.replace('\n', ' ').strip()

        plan_thought = plan.strip()
        if len(plan_thought) > 4000:
            plan_thought = plan_thought[:4000] + "\n\n[... план обрезан ...]"

        return {
            "interview_plan": plan_str,
            "internal_thoughts": [f"[Planner]: {plan_thought}"],
        }
    except Exception as e:
        error_msg = "[Planner]: Ошибка планирования"
        logger.exception(f"Planner error: {e}")
        return {
            "interview_plan": error_msg,
            "internal_thoughts": [error_msg],
        }

