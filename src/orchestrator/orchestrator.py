# src/orchestrator/orchestrator.py

from langgraph.graph import StateGraph, END


def build_graph(agents):

    graph = StateGraph(dict)

    agent_map = {agent.__class__.__name__: agent for agent in agents}

    # =====================================================
    # MASTER ROUTER (LOOP SAFE)
    # =====================================================
    def master_router(state: dict) -> str:

        # Temp Debug check - print state flags to trace routing decisions
        print("\n🧭 MASTER ROUTER STATE CHECK:")
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

        # 1️⃣ Intent
        if "QueryIntentAgent" in agent_map and not state.get("intent"):
            return "QueryIntentAgent"

        # 2️⃣ Retrieval
        if "VectorAgent" in agent_map and not state.get("retrieved_docs"):
            return "VectorAgent"

        # 3️⃣ Context
        if "ContextBuilderAgent" in agent_map and not state.get("context"):
            return "ContextBuilderAgent"

        # 4️⃣ Knowledge Gap (RUN ONLY ONCE)
        if (
            "KnowledgeGapAgent" in agent_map
            and state.get("context")
            and not state.get("gap_checked", False)
        ):
            return "KnowledgeGapAgent"

        # 5️⃣ Feedback Logging
        if (
            "FeedbackMemoryAgent" in agent_map
            and state.get("gap_signal")
            and not state.get("_feedback_done", False)
        ):
            return "FeedbackMemoryAgent"

        # 6️⃣ Clara (RUN AFTER GAP CHECK)
        if "ClaraAgent" in agent_map and not state.get("answer"):
            return "ClaraAgent"

        # 7️⃣ Finalize
        if (
            "FinalizerAgent" in agent_map
            and not state.get("finalized", False)
        ):
            return "FinalizerAgent"

        return "END"

    graph.add_node("MASTER", lambda state: state)

    for agent_name, agent in agent_map.items():

        def create_node(a):
            def node_fn(state: dict):
                if a.can_handle(state):
                    print(f"⚙ Running agent: {a.__class__.__name__}")
                    return a.run(state)
                return state
            return node_fn

        graph.add_node(agent_name, create_node(agent))
        graph.add_edge(agent_name, "MASTER")

    routing_map = {name: name for name in agent_map.keys()}
    routing_map["END"] = END

    graph.add_conditional_edges("MASTER", master_router, routing_map)
    graph.set_entry_point("MASTER")

    return graph.compile()
