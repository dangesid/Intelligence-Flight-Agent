import json


class Orchestrator:
    def __init__(self, agents, max_iterations: int = 10):
        self.agents = agents
        self.max_iterations = max_iterations

    def run(self, payload: dict):
        iterations = 0
        completed = False

        while iterations < self.max_iterations:
            iterations += 1
            handled = False

            for agent in self.agents:
                if agent.can_handle(payload):
                    print(f"⚙ Running agent: {agent.__class__.__name__}")
                    payload = agent.run(payload)
                    handled = True
                    break  # Move to next iteration after one agent runs

            if payload.get("finalized") is True:
                completed = True
                break

            if not handled:
                break

        # Save final output if exists
        if payload.get("final_output"):
            with open("system_state.json", "w") as f:
                json.dump(payload["final_output"], f, indent=4)

        # Attach orchestration info to payload instead of printing here
        payload["orchestration"] = {"iterations": iterations, "completed": completed}

        return payload