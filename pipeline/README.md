# pipeline/ — runs the algorithms frame by frame

| File | What it does |
|---|---|
| `traffic_pipeline.py` | `TrafficVisionPipeline`: reads a frame → detection → tracking → rider association → plate OCR → violation rules → duplicate merging → saves the violation (status *waiting for review*) |
| `evidence_recorder.py` | Saves the evidence for each violation: snapshot, plate crop, short clip before/after the event and `metadata.json`, under `data/evidence/<date>/<event id>/` |

Used by `scripts/run_video.py`, the dashboard (Live cameras, Camera setup → Test stream) and the API upload endpoint.
