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

            # If finalizer marked completion → stop loop
            if payload.get("finalized") is True:
                completed = True
                break

            # If no agent handled → stop
            if not handled:
                break

        # Save final output if exists
        if payload.get("final_output"):
            with open("system_state.json", "w") as f:
                json.dump(payload["final_output"], f, indent=4)

        print("\n🧬 Orchestration Trace (debug only):")
        print({"iterations": iterations, "completed": completed})

        return payload
