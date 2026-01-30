from langchain_core.messages import HumanMessage
import logging
import sys
from interview_engine import InterviewEngine


# logging.basicConfig(
#     level=logging.INFO,
#     format="%(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )


logger = logging.getLogger(__name__)


SEPARATOR = "=" * 60


def _prompt_with_default(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _print_header() -> None:
    print(SEPARATOR)
    print("Multi-Agent Interview Coach")
    print(SEPARATOR)


def _print_system(message: str) -> None:
    print(f"[Система] {message}")


def _print_interviewer(message: str) -> None:
    print(f"\nСобеседователь: {message}\n")


def get_candidate_profile() -> dict:
    _print_header()
    print("\nПрофиль кандидата (можно просто нажимать Enter):\n")

    name = _prompt_with_default("Имя", "Кандидат")
    role = _prompt_with_default("Позиция", "Developer")
    grade = _prompt_with_default("Грейд", "Junior")
    exp = _prompt_with_default("Опыт (кратко)", "Не указано")

    return {
        "name": name,
        "role": role,
        "grade": grade,
        "exp": exp
    }


# Здесь костыль через остановку ключевым словом, ничего другого не придумал)
def get_multiline_input(prompt: str = "Вы: ") -> str:
    print(f"Введите ответ (завершите ввод строкой '--stop--')\n{prompt}")
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == '--stop--':
                _print_system("Формирую следующий вопрос...")
                break
            lines.append(line)
        except EOFError:
            break
    return '\n'.join(lines)


def run_interview():
    try:
        profile = get_candidate_profile()

        _print_system("Инициализация агентов...")
        engine = InterviewEngine()

        _print_system("Интервью готово к началу.")
        _print_system("Начните диалог первым. Для завершения введите 'стоп' или попросите 'фидбэк'.\n")
        state = engine.start_interview(profile)

        while True:
            try:
                user_input = get_multiline_input()

                if not user_input or len(user_input) == 0 or user_input.isspace():
                    _print_system("Введите ответ.")
                    continue

                stop_keywords = [
                    "стоп", "фидбэк", "хватит", "заверши", "закончи", 
                    "стоп игра", "давай фидбэк", "хочу завершить", 
                    "завершить интервью", "закончить интервью", "стоп интервью"
                ]
                user_input_lower = user_input.lower()
                if any(keyword in user_input_lower for keyword in stop_keywords):
                    _print_system("Завершаю интервью...\n")
                    prev_agent_msg = state.get("current_agent_response", "")
                    engine.logger.add_turn(
                        agent_msg=prev_agent_msg if prev_agent_msg else "(начало диалога)",
                        user_msg=user_input,
                        thoughts=""
                    )
                    state["messages"].append(HumanMessage(content=user_input))
                    state["turn_count"] = state.get("turn_count", 0) + 1

                    output = engine.graph.invoke(state)
                    if output.get("is_finished"):
                        feedback = output.get("current_agent_response", "")
                        print(SEPARATOR)
                        print("ФИНАЛЬНЫЙ ФИДБЭК")
                        print(f"{SEPARATOR}\n")
                        print(feedback)
                        print(f"\n{SEPARATOR}\n")
                        engine.logger.set_final_feedback(feedback)
                        engine.logger.save_to_file()
                        engine.logger.save_traces_to_file()
                    break
                
                # Обрабатываем ввод пользователя
                state = engine.process_user_input(state, user_input)
                
                # Проверяем, завершено ли интервью
                if state.get("is_finished"):
                    feedback = state.get("current_agent_response", "")
                    print(f"\n{SEPARATOR}")
                    print("ФИНАЛЬНЫЙ ФИДБЭК")
                    print(f"{SEPARATOR}\n")
                    print(feedback)
                    print(f"\n{SEPARATOR}\n")

                    engine.finish_interview(state)
                    break
                
                # Выводим ответ интервьюера и ждем следующего ввода пользователя
                agent_response = state.get("current_agent_response", "")
                if agent_response:
                    _print_interviewer(agent_response)
                else:
                    # если нет ответа, все равно ждем следующего ввода
                    _print_system("Ожидаю ваш ответ...\n")
                
            except KeyboardInterrupt:
                print("\n")
                _print_system("Интервью прервано.")
                save = input("Сохранить прогресс? (y/n): ").strip().lower()
                if save == 'y':
                    engine.finish_interview(state)
                break
            except Exception as e:
                print(f"\n[Ошибка]: Произошла ошибка: {e}")
                _print_system("Попробуйте продолжить или введите 'стоп' для завершения.")
                continue
        
        _print_system("Интервью завершено. Логи сохранены в папке ./logs/")
        
    except Exception as e:
        logger.exception("\n[Критическая ошибка]")
        logger.info("[Система]: Не удалось запустить систему интервью.")


if __name__ == "__main__":
    run_interview()
