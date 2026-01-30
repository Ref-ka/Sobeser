from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_tavily._utilities import TavilySearchAPIWrapper
from pydantic import SecretStr
import time

from state import AgentState
from logger import InterviewLogger
from prompts import planner_prompt, observer_prompt, interviewer_prompt, manager_prompt
from routing import route_before_observer, route_after_observer
from nodes import planner_node, observer_node, interviewer_node, manager_node

from config import config


class InterviewEngine:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.model_name,
            openai_api_key=config.vsegpt_api_key,
            base_url="https://api.vsegpt.ru/v1",
            temperature=0.2
        )

        if config.tavily_api_key:
            tavily_tool = TavilySearch(
                api_wrapper=TavilySearchAPIWrapper(tavily_api_key=SecretStr(config.tavily_api_key)),
                max_results=5,
                search_depth="basic",
            )
            manager_tools = [tavily_tool]
            planner_tools = [tavily_tool]
        else:
            manager_tools = None
            planner_tools = None

        self.planner_agent = create_agent(
            model=self.llm,
            tools=planner_tools,
            system_prompt=planner_prompt
        )

        self.observer_agent = create_agent(
            model=self.llm,
            tools=None,
            system_prompt=observer_prompt
        )

        self.interviewer_agent = create_agent(
            model=self.llm,
            tools=None,
            system_prompt=interviewer_prompt
        )

        self.manager_agent = create_agent(
            model=self.llm,
            tools=manager_tools,
            system_prompt=manager_prompt,
        )

        self.logger = InterviewLogger()

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        def wrap_node(node_name: str, fn):
            def _wrapper(state: AgentState) -> dict:
                start = time.perf_counter()
                self.logger.add_trace_event(
                    node=node_name,
                    phase="start",
                    turn_count=state.get("turn_count"),
                    input_messages=len(state.get("messages", [])),
                    input_internal_thoughts=len(state.get("internal_thoughts", [])),
                )
                try:
                    out = fn(state)
                    duration_ms = int((time.perf_counter() - start) * 1000)
                    self.logger.add_trace_event(
                        node=node_name,
                        phase="end",
                        turn_count=state.get("turn_count"),
                        duration_ms=duration_ms,
                        output_messages=len(out.get("messages", [])) if isinstance(out, dict) else None,
                        output_internal_thoughts=len(out.get("internal_thoughts", [])) if isinstance(out, dict) else None,
                    )
                    return out
                except Exception as e:
                    duration_ms = int((time.perf_counter() - start) * 1000)
                    self.logger.add_trace_event(
                        node=node_name,
                        phase="error",
                        turn_count=state.get("turn_count"),
                        duration_ms=duration_ms,
                        error=str(e),
                    )
                    raise

            return _wrapper

        workflow.add_node("planner", wrap_node("planner", lambda s: planner_node(s, self.planner_agent)))
        workflow.add_node("observer", wrap_node("observer", lambda s: observer_node(s, self.observer_agent)))
        workflow.add_node("interviewer", wrap_node("interviewer", lambda s: interviewer_node(s, self.interviewer_agent)))
        workflow.add_node("manager", wrap_node("manager", lambda s: manager_node(s, self.manager_agent, self.observer_agent)))

        workflow.add_conditional_edges(
            START,
            route_before_observer,
            {
                "observer": "observer",
                "planner": "planner",
            }
        )

        workflow.add_edge("planner", "observer")

        workflow.add_conditional_edges(
            "observer",
            route_after_observer,
            {
                "interviewer": "interviewer",
                "manager": "manager"
            }
        )

        # Interviewer всегда ведет к концу (ждем ответа пользователя)
        workflow.add_edge("interviewer", END)

        # Manager завершает интервью
        workflow.add_edge("manager", END)

        return workflow.compile()

    @staticmethod
    def start_interview(candidate_profile: dict) -> dict:
        initial_state = {
            "messages": [],
            "internal_thoughts": [],
            "candidate_profile": candidate_profile,
            "observer_instructions": "",
            "current_agent_response": "",
            "is_finished": False,
            "turn_count": 0
        }

        return initial_state

    def process_user_input(self, state: dict, user_input: str) -> dict:
        old_thoughts = state.get("internal_thoughts", [])
        old_thoughts_len = len(old_thoughts)

        state["messages"].append(HumanMessage(content=user_input))
        state["turn_count"] = state.get("turn_count", 0) + 1

        # Запускаем граф
        output = self.graph.invoke(state)

        # Получаем ответ агента (может быть из output или уже в state)
        agent_response = output.get("current_agent_response", state.get("current_agent_response", ""))

        # Получаем новые мысли только из этого прохода графа (дельта)
        all_thoughts = output.get("internal_thoughts", old_thoughts)
        new_thoughts = all_thoughts[old_thoughts_len:] if len(all_thoughts) >= old_thoughts_len else all_thoughts
        
        # Извлекаем мысли моделей из новых мыслей этого прохода.
        # Для читаемости финальных логов логируем только Planner/Observer (в порядке появления).
        model_thoughts = [
            thought for thought in new_thoughts
            if ("[Planner]" in thought) or ("[Observer]" in thought)
        ]
        thoughts_to_log = "\n".join(model_thoughts) if model_thoughts else ""

        prev_agent_msg = state.get("current_agent_response", "")
        self.logger.add_turn(
            agent_msg=prev_agent_msg if prev_agent_msg else "(начало диалога)",
            user_msg=user_input,
            thoughts=thoughts_to_log
        )

        is_finished = output.get("is_finished", False)
        state["is_finished"] = is_finished

        # Обновляем состояние из результата графа (он уже мерджит списки по reducer'ам)
        if "messages" in output:
            state["messages"] = output["messages"]
        if "internal_thoughts" in output:
            state["internal_thoughts"] = output["internal_thoughts"]
        state["current_agent_response"] = agent_response

        # Обновляем остальные поля из output
        for key, value in output.items():
            if key not in ["messages", "internal_thoughts", "topics_covered", "current_agent_response", "is_finished"]:
                state[key] = value

        return state

    def finish_interview(self, state: dict) -> str:
        feedback = state.get("current_agent_response", "")
        self.logger.set_final_feedback(feedback)
        self.logger.save_to_file()
        self.logger.save_traces_to_file()
        return feedback

