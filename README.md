# UAV Strategic Deconfliction Web Application

This web application provides a strategic deconfliction system for unmanned aerial vehicles (UAVs) operating in shared airspace. It ensures safe mission planning by generating conflict-free waypoint paths, leveraging AI for conflict zone detection, and presenting results with a classic, user-friendly interface featuring 4D visualization.

## Features
- **Waypoint Input**: Enter primary drone waypoints (x, y, z, t) via a web form.
- **AI-Driven Deconfliction**: Uses DBSCAN clustering to identify high-risk zones and avoid conflicts.
- **Conflict Detection**: Checks spatial and temporal overlaps with predefined drone paths.
- **4D Visualization**: Displays animated paths as MP4 (with `ffmpeg`) or GIF (fallback) in a classic UI.
- **Theme Toggle**: Switch between light (parchment) and dark (vintage) themes.

## Prerequisites
- Python 3.8+
- Dependencies:
  ```bash
  pip install flask numpy matplotlib scikit-learn

## Design Decisions and Architectural Choices
### Backend (Flask)
Flask was chosen for its simplicity and flexibility, ideal for a small-scale web app. The `app.py` file handles routing, form processing, and visualization generation, importing logic from `deconfliction.py` for modularity. This separation allows independent updates to core algorithms without affecting the web layer.

### AI Integration (DBSCAN)
DBSCAN clustering identifies conflict zones from other drones’ waypoints, enhancing path generation by avoiding high-risk areas proactively. I selected DBSCAN because:
- It adapts to variable cluster sizes without needing a predefined number.
- The `eps=5.0` parameter aligns with the safety distance, ensuring relevant zones.
- It filters noise, focusing on significant risks.

The AI reduces path generation attempts, validated by a baseline comparison in `deconfliction.py`, showing improved success rates.

### UI Design
The UI adopts a classic aesthetic with:
- **Typography**: Georgia (serif) for elegance, Courier New (monospace) for inputs, evoking a typewriter feel.
- **Colors**: Light (parchment `#f5f1e9`, brown `#3c2f2f`) and dark (vintage `#2c2526`, gold `#e6d4a3`) themes.
- **Layout**: Flexible two-column design with bordered cards and subtle shadows.
- **Theme Toggle**: Persists via `localStorage`, fixed with `DOMContentLoaded` to ensure DOM readiness.

This design balances usability with a timeless appeal, avoiding modern flat trends.

## Implementation Details
### Spatial and Temporal Checks
- **Spatial**: Euclidean distance (`np.linalg.norm`) with a 5.0-unit safety buffer.
- **Temporal**: Exact time matches (t1 == t2) with a 2.0-unit altitude threshold.
- Conflicts are returned as tuples (other_wp, primary_wp, drone_idx) for detailed reporting.

### Visualization
The 4D visualization (3D space + time) uses Matplotlib animations, saved as MP4 (with `ffmpeg`) or GIF (Pillow fallback). The `Agg` backend ensures threading safety in Flask. Media is styled with gold borders for a classic touch.

### Web Integration
- **Input**: Parsed from a semicolon-separated string into a NumPy array.
- **Output**: JSON response with status, waypoints, conflicts, zones, and visualization URL.
- **Error Handling**: Validates waypoint format, logs exceptions, and returns user-friendly errors.

## Testing Strategy and Edge Cases
### Testing
- Unit tests in `deconfliction.py` check conflict detection and AI zone identification.
- Manual testing via the web UI verifies form submission, visualization, and theme toggle.
- Browser console confirms JavaScript functionality.

### Edge Cases
- **Invalid Input**: Rejects non-4D waypoints with a 400 error.
- **No Conflicts**: Returns “Clear” status and empty conflict list.
- **Dense Airspace**: Limits path attempts to 100, raising an error if unsuccessful.

Future tests could include zero waypoints, out-of-bounds coordinates, and concurrent submissions.

## Scalability Discussion
For tens of thousands of drones:
- **Algorithm**: Replace O(n*m*k) nested loops with a KD-tree (O(n log m)) for conflict checks.
- **Backend**: Use a WSGI server (e.g., Gunicorn) with multiple workers, deployed on a cloud platform (e.g., AWS).
- **Data**: Store paths in a spatiotemporal database (e.g., PostGIS) with Redis caching for frequent queries.
- **Real-Time**: Implement WebSockets (Flask-SocketIO) for live updates and a message queue (Kafka) for path ingestion.
- **AI**: Precompute clusters in a distributed system (e.g., Spark), caching results.

These enhancements would support commercial-scale operations.

## UI Enhancements
The classic UI improves on the initial design by:
- Offering a parchment texture and muted palette for elegance.
- Ensuring responsiveness with flexbox.
- Fixing the theme toggle with proper event handling and persistence.

## Conclusion
This app delivers a functional, AI-enhanced deconfliction system with a polished, classic UI. It meets all requirements while laying a foundation for scalability. Future work could focus on real-time features and advanced AI optimization.
