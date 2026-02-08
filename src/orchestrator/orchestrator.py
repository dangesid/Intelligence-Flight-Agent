from typing import List, Dict, Any
from src.agents.base_agent import BaseAgent


class Orchestrator:
    """
    Autonomous agent orchestrator.

    - No routing logic
    - No hardcoded order dependency
    - Agents decide when to act
    - Runs until system stabilizes
    """

    def __init__(self, agents: List[BaseAgent], max_iterations: int = 10):
        self.agents = agents
        self.max_iterations = max_iterations

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        iteration = 0
        agent_ran = True

        while agent_ran and iteration < self.max_iterations:
            agent_ran = False
            iteration += 1

            for agent in self.agents:
                try:
                    if agent.can_handle(payload):
                        payload = agent.run(payload)
                        agent_ran = True
                except Exception as e:
                    payload.setdefault("errors", []).append({
                        "agent": agent.__class__.__name__,
                        "error": str(e)
                    })
            if payload.get("finalized"):
                break
            
        payload["orchestration"] = {
            "iterations": iteration,
            "completed": not agent_ran
        }

        return payload
