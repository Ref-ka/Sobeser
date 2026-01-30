from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging


logger = logging.getLogger(__name__)


# Мысли observer решил суммаризовывать,
# так как очень сильно засорялось контекстное окно, когда кидал сырой текст,
# просто обрывался final_feedback, возможно, просто я чет не то написал
def summarize_observer_thoughts(observer_thoughts: list, observer_agent) -> str:
    if not observer_thoughts:
        return "Наблюдатель не предоставил анализа."

    full_thoughts = "\n".join(observer_thoughts)

    if len(full_thoughts) < 3000:
        return full_thoughts

    logger.debug(f"Summarizing observer thoughts: {len(full_thoughts)} characters, {len(observer_thoughts)} thoughts")
    
    try:
        summary_prompt = f"""Ты — помощник, который суммирует анализ наблюдателя за техническим интервью.

        Ниже представлены все рассуждения наблюдателя по каждому ответу кандидата. 
        Твоя задача — создать краткое, структурированное резюме, которое сохранит всю важную информацию:
        
        1. По каждой теме/вопросу: кратко укажи, что кандидат ответил правильно, а что неправильно
        2. Отметь ключевые ошибки и пробелы в знаниях
        3. Отметь сильные стороны кандидата
        4. Сохрани информацию о том, какие темы были затронуты
        
        ВАЖНО: Будь максимально лаконичным, но сохрани всю критически важную информацию для формирования фидбэка.
        
        Анализ наблюдателя:
        {full_thoughts}
        
        Создай краткое резюме:"""
        
        summary_messages = [
            SystemMessage(content=summary_prompt),
            HumanMessage(content="Суммаризируй анализ наблюдателя, сохранив всю важную информацию.")
        ]
        
        summary_result = observer_agent.invoke({"messages": summary_messages})
        summary_messages_result = summary_result.get("messages", [])
        
        summary = ""
        if summary_messages_result:
            for msg in reversed(summary_messages_result):
                if isinstance(msg, AIMessage):
                    summary = msg.content
                    break
        
        if not summary or len(summary) < 100:
            logger.warning("Summary failed, using last thoughts only")
            return "\n".join(observer_thoughts[-3:])
        
        logger.debug("Summary created: %s characters", len(summary))
        return summary
        
    except Exception as e:
        logger.exception("Error summarizing thoughts, using truncated version")
        return "\n".join(observer_thoughts[-3:])

