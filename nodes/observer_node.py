from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging
from state import AgentState


logger = logging.getLogger(__name__)


def observer_node(state: AgentState, observer_agent) -> dict:
    if not state.get('messages'):
        return {
            "internal_thoughts": ["[Observer]: Начало интервью. Инициализация анализа."],
            "observer_instructions": "Кандидат ещё ничего не писал. "
                                     "Начни интервью с приветствия и первого технического вопроса.",
            "difficulty_level": 2
        }
    
    # Получаем вопрос и ответ
    if len(state['messages']) > 1:
        question = state['messages'][-2].content
    else:
        question = "Представьтесь."
    last_user_msg = state['messages'][-1].content
    
    # Проверка на стоп
    if "стоп" in last_user_msg.lower():
        return {
            "is_finished": True
        }
    
    profile = state.get('candidate_profile', {})
    current_difficulty = state.get('difficulty_level', 2)
    interview_plan = state.get('interview_plan', "Плана нет")

    context_prompt = f"""Профиль кандидата:
    - Имя: {profile.get('name', 'Не указано')} 
    - Позиция: {profile.get('role', 'Не указано')} 
    - Заявленный уровень: {profile.get('grade', 'Не указано')} 
    - Опыт: {profile.get('exp', 'Не указано')} 
    
    План интервью: 
    {interview_plan} 
    
    Вопрос на который должен был ответить кандидат: 
    "{question}" 
    Ответ кандидата: 
    "{last_user_msg}" """

    try:
        messages = [
            SystemMessage(content=context_prompt),
            HumanMessage(content="Проанализируй ответ кандидата и дай свою оценку.")
        ]

        agent_result = observer_agent.invoke({"messages": messages})

        agent_messages = agent_result.get("messages", [])
        analysis = ""
        if agent_messages:
            for msg in reversed(agent_messages):
                if isinstance(msg, AIMessage):
                    analysis = msg.content
                    break
        
        if not analysis:
            analysis = "Не удалось провести анализ ответа."

        analysis_stripped = analysis.lstrip()
        is_finished = analysis_stripped.lower().startswith("finished")

        analysis_lower = analysis.lower()
        if ("неправ" in analysis_lower) or ("ошиб" in analysis_lower) or ("бред" in analysis_lower):
            next_difficulty = max(1, current_difficulty - 1)
        elif ("правиль" in analysis_lower) and ("частич" not in analysis_lower):
            next_difficulty = min(5, current_difficulty + 1)
        else:
            next_difficulty = current_difficulty
        
        logger.debug(f"Observer response generated: {len(analysis)} characters")
        
        return {
            "internal_thoughts": [f"[Observer]: {analysis}"],
            "is_finished": is_finished,
            "difficulty_level": next_difficulty
        }
    except Exception as e:
        error_msg = "[Observer]: Ошибка анализа"
        logger.exception(f"Observer error: {e}")
        return {
            "internal_thoughts": [error_msg],
            "observer_instructions": "Продолжи интервью с текущим уровнем сложности.",
            "difficulty_level": current_difficulty
        }

