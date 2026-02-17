# src/agents/base_agent.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class BaseAgent(ABC):
    """
    BaseAgent (Agentic Upgrade)
    
    - Supports agentic behavior:
        1️⃣ evaluate() → decide if agent should act
        2️⃣ propose_action() → propose next steps/tools
        3️⃣ execute() → perform processing
    - Legacy can_handle()/run() preserved for backward compatibility
    """

    def __init__(self, name: str):
        self.name = name
        self.local_state: Dict[str, Any] = {}

    # -------------------
    # LEGACY METHODS
    # -------------------
    @abstractmethod
    def can_handle(self, payload: Dict[str, Any]) -> bool:
        """
        Legacy method: Return True if this agent should act on the given payload.
        """
        pass

    @abstractmethod
    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Legacy method: Process the given payload and return updated result.
        """
        pass

    # -------------------
    # AGENTIC METHODS
    # -------------------
    def evaluate(self, payload: Dict[str, Any]) -> bool:
        """
        Decide if this agent should act.
        Default: uses legacy can_handle()
        """
        return self.can_handle(payload)

    def propose_action(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Propose next steps or tools. Return dict with keys:
        {
            'next_agents': List[str],      # Optional
            'tools_to_call': List[str],    # Optional
            'modify_payload': dict,        # Optional changes to payload
        }
        Default: None (no proposal)
        """
        return None

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent logic.
        Default: uses legacy run()
        """
        return self.run(payload)
