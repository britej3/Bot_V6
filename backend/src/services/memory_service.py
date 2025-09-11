from backend.src.services.feature_flag_service import feature_flag_service

class MemoryService:
    def route_request(self, request: dict):
        use_graph = feature_flag_service.get_flag("GRAPH_INTEGRATION_ENABLED")

        if use_graph and use_graph.enabled:
            return self._route_to_graph_db(request)
        else:
            return self._route_to_default_db(request)

    def _route_to_graph_db(self, request: dict):
        # Placeholder for graph DB routing logic
        return {"routed_to": "graph_db", "request": request}

    def _route_to_default_db(self, request: dict):
        # Placeholder for default DB routing logic (e.g., Redis, pgvector)
        return {"routed_to": "default_db", "request": request}

memory_service = MemoryService()
