from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
from deconfliction import (DronePath, generate_conflict_free_primary_waypoints, 
                          query_deconfliction, identify_conflict_zones)
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simulated other drone paths
other_drones_paths = [
    DronePath(np.array([[5, 5, 5, 0], [10, 10, 10, 5], [15, 15, 15, 8]])),
    DronePath(np.array([[2, 2, 2, 2], [8, 8, 8, 6], [12, 12, 12, 9]]))
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deconflict', methods=['POST'])
def deconflict():
    try:
        waypoints_str = request.form.get('waypoints', '')
        waypoints = np.array([list(map(float, wp.split(','))) for wp in waypoints_str.split(';') if wp])
        if waypoints.shape[1] != 4:
            return jsonify({'error': 'Waypoints must have format x,y,z,t'}), 400
        
        primary_path = DronePath(waypoints)
        status, conflicts = query_deconfliction(primary_path, other_drones_paths)
        conflict_zones = identify_conflict_zones(other_drones_paths)
        video_path = generate_visualization(primary_path, other_drones_paths, conflicts, conflict_zones)

        result = {
            'status': status,
            'waypoints': primary_path.waypoints.tolist(),
            'conflicts': [(c[0].tolist(), c[1].tolist(), c[2]) for c in conflicts],
            'conflict_zones': conflict_zones.tolist() if conflict_zones.size > 0 else [],
            'video_url': f'/static/{os.path.basename(video_path)}',
            'is_gif': video_path.endswith('.gif')
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in deconfliction: {e}")
        return jsonify({'error': str(e)}), 500

def generate_visualization(primary_path, other_paths, conflicts, conflict_zones):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_zlim(0, 20)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Altitude')

    min_time = int(min(primary_path.waypoints[:, 3]))
    max_time = int(max(primary_path.waypoints[:, 3]))

    def update(frame):
        ax.clear()
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_zlim(0, 20)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Altitude')
        ax.set_title(f'Time: {frame}')
        ax.view_init(elev=20, azim=45)

        primary_at_t = primary_path.waypoints[primary_path.waypoints[:, 3] == frame]
        if primary_at_t.size > 0:
            ax.scatter(primary_at_t[:, 0], primary_at_t[:, 1], primary_at_t[:, 2],
                       color='blue', label='Primary Drone', s=100)

        for i, path in enumerate(other_paths):
            other_at_t = path.waypoints[path.waypoints[:, 3] == frame]
            if other_at_t.size > 0:
                ax.scatter(other_at_t[:, 0], other_at_t[:, 1], other_at_t[:, 2],
                           color='red', label=f'Other Drone {i+1}' if i == 0 else None, s=100)

        for conflict in conflicts:
            if conflict[0][3] == frame:
                ax.scatter(*conflict[0][:3], color='black', marker='x', label='Conflict', s=150)

        if conflict_zones.size > 0:
            ax.scatter(conflict_zones[:, 0], conflict_zones[:, 1], conflict_zones[:, 2],
                       color='orange', marker='o', label='Conflict Zone', s=200, alpha=0.5)

        if ax.has_data():
            ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=range(min_time, max_time + 1), repeat=False)
    
    video_path = os.path.join('static', 'output.mp4')
    try:
        ani.save(video_path, writer='ffmpeg', fps=2)
        logger.info("Saved animation as MP4 using ffmpeg")
    except ValueError as e:
        logger.warning(f"ffmpeg unavailable: {e}. Falling back to GIF.")
        video_path = os.path.join('static', 'output.gif')
        ani.save(video_path, writer='pillow', fps=2)
    
    plt.close(fig)
    return video_path

@app.route('/static/<path:filename>')
def serve_static(filename):
    # Ensure correct MIME type for GIFs
    if filename.endswith('.gif'):
        return send_file(os.path.join('static', filename), mimetype='image/gif')
    return send_file(os.path.join('static', filename))

if __name__ == '__main__':
    app.run(debug=True)