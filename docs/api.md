# API Documentation

## Feature Flags API

### GET /feature-flags
- **Summary**: List all feature flags.
- **Responses**:
  - `200 OK`: A list of feature flag objects.
    ```json
    [
      {
        "name": "GRAPH_INTEGRATION_ENABLED",
        "enabled": true,
        "description": "Enable/disable graph database integration."
      }
    ]
    ```

### PUT /feature-flags/{name}
- **Summary**: Update a feature flag.
- **Parameters**:
  - `name` (path, string, required): The name of the feature flag to update.
- **Request Body**:
  ```json
  {
    "enabled": false
  }
  ```
- **Responses**:
  - `200 OK`: The updated feature flag object.
  - `404 Not Found`: If the feature flag with the given name does not exist.
