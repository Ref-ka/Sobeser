import json
import os
import logging
from datetime import datetime

LOGS_DIR = "./logs/"


logger = logging.getLogger(__name__)


class InterviewLogger:
    def __init__(self):
        self.log_data = {
            "participant_name": "Черненко Иван Денисович",
            "turns": [],
            "final_feedback": ""
        }
        self.trace_events: list[dict] = []
        self.turn_counter = 1

    def add_turn(
        self, 
        agent_msg: str, 
        user_msg: str, 
        thoughts: str
    ):
        turn_entry = {
            "turn_id": self.turn_counter,
            "agent_visible_message": agent_msg,
            "user_message": user_msg,
            "internal_thoughts": thoughts
        }
            
        self.log_data["turns"].append(turn_entry)
        self.turn_counter += 1

    def set_final_feedback(self, feedback: str):
        self.log_data["final_feedback"] = feedback

    def add_trace_event(
        self,
        *,
        node: str,
        phase: str,
        turn_count: int | None = None,
        duration_ms: int | None = None,
        input_messages: int | None = None,
        output_messages: int | None = None,
        input_internal_thoughts: int | None = None,
        output_internal_thoughts: int | None = None,
        error: str | None = None,
    ):
        self.trace_events.append(
            {
                "ts": datetime.now().isoformat(),
                "node": node,
                "phase": phase,
                "turn_count": turn_count,
                "duration_ms": duration_ms,
                "input_messages": input_messages,
                "output_messages": output_messages,
                "input_internal_thoughts": input_internal_thoughts,
                "output_internal_thoughts": output_internal_thoughts,
                "error": error,
            }
        )

    def save_to_file(self, filename: str | None = None):
        try:
            os.makedirs(LOGS_DIR, exist_ok=True)
            if not filename:
                filename = f"interview_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
            file_path = os.path.join(LOGS_DIR, filename)
            with open(file=file_path, mode='w', encoding='utf-8') as f:
                json.dump(self.log_data, f, ensure_ascii=False, indent=2)
            logger.info("\n[Система]: Лог сохранен в %s", filename)
        except Exception as e:
            logger.exception("\n[Ошибка]: Не удалось сохранить лог")

    def save_traces_to_file(self, filename: str | None = None):
        try:
            os.makedirs(LOGS_DIR, exist_ok=True)
            if not filename:
                filename = "runtime_traces.jsonl"
            file_path = os.path.join(LOGS_DIR, filename)
            with open(file=file_path, mode='w', encoding='utf-8') as f:
                for event in self.trace_events:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            logger.info("\n[Система]: Runtime-трейсы сохранены в %s", filename)
        except Exception as e:
            logger.exception("\n[Ошибка]: Не удалось сохранить runtime-трейсы")

