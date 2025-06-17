from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

from write_a_book_with_flows.typecast import Chapter
from dotenv import load_dotenv
import time

load_dotenv()
API_BASE = "http://localhost:11434"



@CrewBase
class WriteBookChapterCrew:
    """Write Book Chapter Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    llm = LLM(model="ollama/qwen3:14b", base_url=API_BASE, api_key="ollama") # type: ignore

    @agent
    def researcher(self) -> Agent:
        search_tool = SerperDevTool()
        return Agent(
            config=self.agents_config["researcher"], # type: ignore
            tools=[search_tool],
            llm=self.llm,
            verbose=True
        ) # type: ignore

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config["writer"], # type: ignore
            llm=self.llm,
            verbose=True
        ) # type: ignore

    @task
    def research_chapter(self) -> Task:
        return Task(
            config=self.tasks_config["research_chapter"], # type: ignore
        ) # type: ignore

    @task
    def write_chapter(self) -> Task:
        return Task(config=self.tasks_config["write_chapter"], output_pydantic=Chapter) # type: ignore

    @crew
    def crew(self) -> Crew:
        """Creates the Write Book Chapter Crew"""
        return Crew(
            agents=self.agents, # type: ignore
            tasks=self.tasks,   # type: ignore
            process=Process.sequential,
            verbose=True,
        )

