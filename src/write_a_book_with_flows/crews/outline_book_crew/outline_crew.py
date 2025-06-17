from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

from dotenv import load_dotenv
import time

load_dotenv()

from write_a_book_with_flows.typecast import BookOutline
API_BASE = "http://localhost:11434"

@CrewBase
class OutlineCrew:
    """Book Outline Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    llm = LLM(model="ollama/qwen3:14b", base_url=API_BASE, api_key="ollama") # type: ignore

    @agent
    def researcher(self) -> Agent:
        search_tool = SerperDevTool(search_url="https://google.serper.dev/search", n_results=3) # type: ignore
        return Agent(
            config=self.agents_config["researcher"], # type: ignore
            tools=[search_tool],
            llm=self.llm,
            verbose=True,
            ) # type: ignore

    @agent
    def outliner(self) -> Agent:
        return Agent(
            config=self.agents_config["outliner"], # type: ignore
            llm=self.llm,
            verbose=True,
        ) # type: ignore

    @task
    def research_topic(self) -> Task:
        return Task(
            config=self.tasks_config["research_topic"], #type: ignore
        ) # type: ignore

    @task
    def generate_outline(self) -> Task:
        return Task(
            config=self.tasks_config["generate_outline"], #type: ignore
            output_pydantic=BookOutline
        ) # type: ignore

    @crew
    def crew(self) -> Crew:
        """Creates the Book Outline Crew"""
        return Crew(
            agents=self.agents, # type: ignore
            tasks=self.tasks, # type: ignore
            process=Process.sequential,
            verbose=True,
        )

