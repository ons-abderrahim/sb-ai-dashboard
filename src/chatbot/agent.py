"""
LangChain NL Chatbot Agent
==========================
Provides a natural-language interface over the Smart Building dashboard.
Connects to:
  - InfluxDB (via custom LangChain tools)
  - The Project 1 RAG pipeline (for research document Q&A)
  - The FastAPI predictions/anomaly endpoints
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from src.chatbot.tools import (
    get_active_anomalies,
    get_occupancy_prediction,
    query_sensor_db,
    rag_search,
    trigger_hvac_adjustment,
)
from src.utils.config import settings

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are the Smart Building AI Assistant for Concordia University's CIISE lab \
research infrastructure. You help building managers and researchers understand sensor data, \
occupancy patterns, energy usage, and anomalies.

You have access to the following tools:
- **query_sensor_db**: Query historical sensor readings (CO₂, temperature, humidity, motion, etc.)
- **get_occupancy_prediction**: Get real-time occupancy predictions for any zone
- **get_active_anomalies**: Retrieve current anomaly alerts
- **trigger_hvac_adjustment**: Send an HVAC override command (ask for confirmation first)
- **rag_search**: Search Concordia CIISE lab publications and ASHRAE research documents

Guidelines:
- Always cite data sources and timestamps when reporting sensor values
- Express probabilities and confidence levels clearly
- Ask for confirmation before triggering any HVAC adjustments
- Be concise and actionable — building managers are busy
- When relevant, link findings to energy-saving recommendations
"""


def create_building_agent(building_id: str = "concordia_ev") -> AgentExecutor:
    """
    Create a LangChain agent executor for a specific building.

    Args:
        building_id: Identifier for the building context

    Returns:
        Configured AgentExecutor ready for .invoke() calls
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=settings.openai_api_key,
        streaming=True,
    )

    tools = [
        query_sensor_db,
        get_occupancy_prediction,
        get_active_anomalies,
        trigger_hvac_adjustment,
        rag_search,
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + f"\n\nCurrent building context: {building_id}"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10,  # Keep last 10 exchanges
    )

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )


class BuildingChatSession:
    """
    Manages a stateful chat session for a single user/building context.

    Usage:
        session = BuildingChatSession(building_id="concordia_ev")
        response = await session.chat("How many people are on floor 3?")
    """

    def __init__(self, building_id: str = "concordia_ev") -> None:
        self.building_id = building_id
        self._executor = create_building_agent(building_id)
        logger.info("Chat session created", building_id=building_id)

    async def chat(self, message: str) -> dict[str, Any]:
        """
        Send a message and get a response.

        Returns:
            dict with keys: answer (str), sources (list), tool_calls (list)
        """
        logger.info("Processing chat message", building=self.building_id)
        try:
            result = self._executor.invoke({"input": message})
            return {
                "answer": result.get("output", ""),
                "sources": result.get("sources", []),
                "tool_calls": result.get("intermediate_steps", []),
            }
        except Exception as exc:
            logger.error("Agent error", error=str(exc))
            return {
                "answer": "I encountered an error processing your request. Please try again.",
                "sources": [],
                "tool_calls": [],
            }
