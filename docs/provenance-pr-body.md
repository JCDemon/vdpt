# Pull Request Body Draft

## Summary
- add structured provenance nodes and edges using a PROV-inspired schema stored alongside run artifacts
- expose provenance pointers and structured logs in preview/execute API responses
- surface per-record operation details in the Streamlit UI with download links

## Example provenance JSON
```json
{
  "run_id": "run-20240101-010203",
  "generated_at": "2024-01-01T01:02:03Z",
  "records": [
    {
      "row_index": 0,
      "inputs": {"title": "Example article"},
      "parameters": [
        {
          "kind": "summarize",
          "params": {"field": "body", "max_tokens": 128}
        }
      ],
      "outputs": {"summary": "Concise abstract."},
      "provenance": {
        "nodes": [
          {"id": "row-0", "type": "Entity", "label": "Row 0 Input"},
          {"id": "op-0-0", "type": "Activity", "label": "summarize"},
          {
            "id": "out-0-0-summary",
            "type": "Entity",
            "label": "summary",
            "metadata": {"value": "Concise abstract."}
          }
        ],
        "edges": [
          {"source": "row-0", "target": "op-0-0", "relation": "used"},
          {"source": "op-0-0", "target": "out-0-0-summary", "relation": "wasGeneratedBy"}
        ]
      },
      "logs": [
        {
          "ts": "2024-01-01T01:02:04Z",
          "level": "INFO",
          "message": "operation 'summarize' completed"
        }
      ]
    }
  ],
  "logs": [
    {
      "ts": "2024-01-01T01:02:04Z",
      "level": "INFO",
      "message": "operation 'summarize' completed",
      "context": {"row_index": 0, "outputs": ["summary"]}
    }
  ]
}
```

The provenance schema references core concepts from the [W3C PROV overview](https://www.w3.org/TR/prov-overview/).
