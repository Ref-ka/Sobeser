from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging
from state import AgentState
from summarizer import summarize_observer_thoughts


logger = logging.getLogger(__name__)

def manager_node(state: AgentState, manager_agent, observer_agent) -> dict:
    """Менеджер, формирующий финальный фидбэк по итогам интервью."""
    messages = state.get('messages', [])
    internal_thoughts = state.get('internal_thoughts', [])
    logger.debug("Manager internal thoughts count: %s", len(internal_thoughts))
    profile = state.get('candidate_profile', {})
    
    # Все мысли Observer
    observer_thoughts_list = [
        thought for thought in internal_thoughts
        if "[Observer]" in thought
    ]
    
    # Суммаризируем мысли observer для экономии контекста
    observer_thoughts = summarize_observer_thoughts(observer_thoughts_list, observer_agent)
    
    # Дополнительная проверка: если суммаризация все еще слишком длинная, обрезаем
    # Ограничиваем до 8000 символов для безопасности (оставляем место для промпта и ответа)
    if len(observer_thoughts) > 8000:
        logger.warning(f"Observer thoughts still too long ({len(observer_thoughts)} chars), truncating to 8000")
        observer_thoughts = observer_thoughts[:8000] + "\n\n[... анализ обрезан для экономии контекста ...]"
    
    # Формируем контекст для агента
    context_prompt = f"""Профиль кандидата:
    - Имя: {profile.get('name', 'Не указано')}
    - Позиция: {profile.get('role', 'Не указано')}
    - Заявленный уровень: {profile.get('grade', 'Не указано')}
    - Опыт: {profile.get('exp', 'Не указано')}
    
    Суммаризированный анализ всех ответов кандидата, проведенный наблюдателем:
    {observer_thoughts}"""
    
    logger.debug(f"Manager context length: {len(context_prompt)} characters")
    
    try:
        agent_messages = [
            SystemMessage(content=context_prompt),
            HumanMessage(content="""Сформируй финальный фидбэк по итогам технического интервью на основе предоставленной информации.

            Если в Knowledge Gaps есть темы, подбери 3-5 актуальных источников (документация/статьи/книги/курсы) для улучшения знаний.
            Если доступен веб-поиск, используй инструмент `tavily_search`.
            Верни рекомендации в структурированном виде: название + ссылка + почему полезно.
            """)
        ]

        # Вызываем агента через его граф
        agent_result = manager_agent.invoke({"messages": agent_messages})

        # Извлекаем ответ агента
        agent_messages_result = agent_result.get("messages", [])
        feedback = ""
        
        if agent_messages_result:
            # Ищем последнее сообщение от агента
            for msg in reversed(agent_messages_result):
                if isinstance(msg, AIMessage):
                    feedback = msg.content
                    break
        
        if not feedback:
            feedback = "Не удалось сгенерировать фидбэк."
        
        logger.debug(f"Manager feedback generated: {len(feedback)} characters")
        
        return {
            "is_finished": True,
            "current_agent_response": feedback,
            "internal_thoughts": ["[Manager]: Финальный фидбэк сформирован на основе суммаризированных наблюдений и профиля кандидата.\n"],
        }
    except Exception as e:
        error_feedback = f"""=== ФИНАЛЬНЫЙ ФИДБЭК ===
        Произошла техническая ошибка при генерации фидбэка: {e}
        """
        logger.exception(f"Manager error {e}")
        return {
            "is_finished": True,
            "current_agent_response": error_feedback,
            "internal_thoughts": [f"[Manager]: {error_feedback}\n"],
        }

