# src/orchestrator/orchestrator.py

from langgraph.graph import StateGraph, END

def build_graph(agents):

    graph = StateGraph(dict)

    agent_map = {agent.__class__.__name__: agent for agent in agents}

    # =====================================================
    # MASTER MEDIATOR (AGENTIC READY, LOOP SAFE)
    # =====================================================
    def master_mediator(state: dict) -> str:

        # Temp Debug check - print state flags to trace routing decisions
        print("\n🧭 MASTER MEDIATOR STATE CHECK:")
        print({
            "intent": bool(state.get("intent")),
            "retrieved_docs": bool(state.get("retrieved_docs")),
            "context": bool(state.get("context")),
            "gap_checked": state.get("gap_checked"),
            "gap_signal": state.get("gap_signal") is not None,
            "_feedback_done": state.get("_feedback_done"),
            "answer": bool(state.get("answer")),
            "finalized": state.get("finalized"),
        })

        # Stop condition
        if state.get("finalized") is True:
            return "END"

        # Agent proposals
        proposals = []
        for agent_name, agent in agent_map.items():
            if hasattr(agent, "evaluate") and agent.evaluate(state):
                action = agent.propose_action(state)
                if action:
                    proposals.append((agent_name, action))

        # If no agent proposes, fallback to old deterministic order
        if not proposals:
            if "QueryIntentAgent" in agent_map and not state.get("intent"):
                return "QueryIntentAgent"
            if "VectorAgent" in agent_map and not state.get("retrieved_docs"):
                return "VectorAgent"
            if "ContextBuilderAgent" in agent_map and not state.get("context"):
                return "ContextBuilderAgent"
            if (
                "KnowledgeGapAgent" in agent_map
                and state.get("context")
                and not state.get("gap_checked", False)
            ):
                return "KnowledgeGapAgent"
            if (
                "FeedbackMemoryAgent" in agent_map
                and state.get("gap_signal")
                and not state.get("_feedback_done", False)
            ):
                return "FeedbackMemoryAgent"
            if "ClaraAgent" in agent_map and not state.get("answer"):
                return "ClaraAgent"
            if (
                "FinalizerAgent" in agent_map
                and not state.get("finalized", False)
            ):
                return "FinalizerAgent"

            return "END"

        # Pick the first proposed action (or implement priority logic here)
        next_agent_name = proposals[0][0]
        return next_agent_name

    graph.add_node("MASTER", lambda state: state)

    for agent_name, agent in agent_map.items():

        def create_node(a):
            def node_fn(state: dict):
                # If agent has autonomous execute method, use it
                if hasattr(a, "execute") and a.evaluate(state):
                    print(f"⚙ Running agent: {a.__class__.__name__}")
                    return a.execute(state)
                # fallback to old run/can_handle
                if hasattr(a, "can_handle") and a.can_handle(state):
                    print(f"⚙ Running agent (legacy): {a.__class__.__name__}")
                    return a.run(state)
                return state
            return node_fn

        graph.add_node(agent_name, create_node(agent))
        graph.add_edge(agent_name, "MASTER")

    routing_map = {name: name for name in agent_map.keys()}
    routing_map["END"] = END

    graph.add_conditional_edges("MASTER", master_mediator, routing_map)
    graph.set_entry_point("MASTER")

    return graph.compile()
