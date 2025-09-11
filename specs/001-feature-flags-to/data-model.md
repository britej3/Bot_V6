# Data Model: Feature Flag

## Entity: FeatureFlag
Represents a single feature flag.

### Fields
- `name`: string (e.g., "GRAPH_INTEGRATION_ENABLED") - The unique identifier for the flag.
- `enabled`: boolean (true/false) - The state of the flag.
- `description`: string (optional) - A brief description of what the flag controls.

### Validation Rules
- `name` is required and must be unique.
- `enabled` is required.

### State Transitions
- The `enabled` state can be transitioned from `true` to `false` or vice-versa at runtime.