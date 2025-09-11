# Quickstart: Using the Graph Integration Feature Flag

This guide explains how to use the feature flag to control the graph database integration.

## 1. Check the flag status
To see the current status of all feature flags, make a GET request to the `/feature-flags` endpoint.

```bash
cURL http://localhost:8000/feature-flags
```

## 2. Disable the graph integration
To disable the graph integration, send a PUT request to the `/feature-flags/GRAPH_INTEGRATION_ENABLED` endpoint with `{"enabled": false}`.

```bash
curl -X PUT -H "Content-Type: application/json" -d '{"enabled": false}' http://localhost:8000/feature-flags/GRAPH_INTEGRATION_ENABLED
```

## 3. Enable the graph integration
To enable the graph integration, send a PUT request with `{"enabled": true}`.

```bash
curl -X PUT -H "Content-Type: application/json" -d '{"enabled": true}' http://localhost:8000/feature-flags/GRAPH_INTEGRATION_ENABLED
```

## 4. Verify the behavior
- With the flag disabled, execute a code path that would normally use the graph database and verify from the logs or system behavior that the fallback mechanism is used instead.
- With the flag enabled, verify that the graph database is called as expected.