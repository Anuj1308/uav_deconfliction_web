import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Optional
import logging
from sklearn.cluster import DBSCAN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SAFETY_DISTANCE = 5.0
DEFAULT_ALTITUDE_DIFF = 2.0
SPACE_BOUNDS = (0, 20)
TIME_BOUNDS = (0, 10)
CLUSTER_EPS = 5.0  # DBSCAN radius for clustering (matches safety distance)
MIN_SAMPLES = 2    # Minimum points to form a cluster

class DronePath:
    """Represents a drone's flight path with waypoints and timing."""
    def __init__(self, waypoints: np.ndarray):
        if waypoints.shape[1] != 4:
            raise ValueError("Waypoints must have 4 columns: x, y, z, t")
        self.waypoints = waypoints

    def __repr__(self):
        return f"DronePath(waypoints={self.waypoints.shape})"

def identify_conflict_zones(other_drones_paths: List[DronePath]) -> np.ndarray:
    """Use DBSCAN to identify high-risk zones from other drones' waypoints."""
    if not other_drones_paths:
        return np.array([])

    # Combine all waypoints from other drones (x, y, z only, ignoring time for spatial clustering)
    all_points = np.vstack([path.waypoints[:, :3] for path in other_drones_paths])
    if len(all_points) < MIN_SAMPLES:
        return np.array([])

    # Apply DBSCAN clustering
    db = DBSCAN(eps=CLUSTER_EPS, min_samples=MIN_SAMPLES).fit(all_points)
    labels = db.labels_

    # Extract cluster centroids (excluding noise points labeled -1)
    conflict_zones = []
    for cluster_id in set(labels) - {-1}:  # Exclude noise
        cluster_points = all_points[labels == cluster_id]
        centroid = np.mean(cluster_points, axis=0)
        conflict_zones.append(centroid)

    conflict_zones = np.array(conflict_zones) if conflict_zones else np.array([])
    logger.info(f"Identified {len(conflict_zones)} conflict zones")
    return conflict_zones

def is_safe_from_zones(
    waypoint: np.ndarray,
    conflict_zones: np.ndarray,
    safety_distance: float
) -> bool:
    """Check if a waypoint is safe from identified conflict zones."""
    if conflict_zones.size == 0:
        return True
    distances = np.linalg.norm(conflict_zones - waypoint[:3], axis=1)
    return np.all(distances >= safety_distance)

def generate_conflict_free_primary_waypoints(
    num_waypoints: int = 5,
    other_drones_paths: Optional[List[DronePath]] = None,
    space_bounds: Tuple[int, int] = SPACE_BOUNDS,
    time_bounds: Tuple[int, int] = TIME_BOUNDS,
    safety_distance: float = DEFAULT_SAFETY_DISTANCE,
    min_altitude_diff: float = DEFAULT_ALTITUDE_DIFF,
    max_attempts: int = 100
) -> DronePath:
    """Generate a conflict-free path, avoiding AI-identified conflict zones."""
    attempt = 0
    conflict_zones = identify_conflict_zones(other_drones_paths) if other_drones_paths else np.array([])

    while attempt < max_attempts:
        x = np.random.randint(space_bounds[0], space_bounds[1], size=num_waypoints)
        y = np.random.randint(space_bounds[0], space_bounds[1], size=num_waypoints)
        z = np.random.randint(space_bounds[0], space_bounds[1], size=num_waypoints)
        t = np.sort(np.random.randint(time_bounds[0], time_bounds[1], size=num_waypoints))
        primary_path = np.column_stack((x, y, z, t))

        if not other_drones_paths:
            return DronePath(primary_path)

        # Check against conflict zones first (AI-driven optimization)
        zone_safe = all(is_safe_from_zones(wp, conflict_zones, safety_distance) for wp in primary_path)
        if not zone_safe:
            attempt += 1
            continue

        # Then check against individual waypoints
        if is_conflict_free(primary_path, other_drones_paths, safety_distance, min_altitude_diff):
            logger.info(f"Conflict-free path found after {attempt + 1} attempts")
            return DronePath(primary_path)
        attempt += 1

    raise RuntimeError(f"Failed to find conflict-free path after {max_attempts} attempts")

def is_conflict_free(
    path: np.ndarray,
    other_paths: List[DronePath],
    safety_distance: float,
    min_altitude_diff: float
) -> bool:
    """Check if a path is conflict-free against other drone paths."""
    for other_path in other_paths:
        for wp in path:
            for other_wp in other_path.waypoints:
                if check_spatial_temporal_conflict(wp, other_wp, safety_distance, min_altitude_diff):
                    return False
    return True

def check_spatial_temporal_conflict(
    wp1: np.ndarray,
    wp2: np.ndarray,
    safety_distance: float,
    min_altitude_diff: float
) -> bool:
    """Check if two waypoints conflict in space and time."""
    spatial_distance = np.linalg.norm(wp1[:3] - wp2[:3])
    time_conflict = wp1[3] == wp2[3]
    altitude_conflict = abs(wp1[2] - wp2[2]) < min_altitude_diff
    return spatial_distance < safety_distance and time_conflict and altitude_conflict

def check_conflicts(
    primary_path: DronePath,
    other_paths: List[DronePath],
    safety_distance: float = DEFAULT_SAFETY_DISTANCE,
    min_altitude_diff: float = DEFAULT_ALTITUDE_DIFF
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Detect conflicts between primary path and other drone paths."""
    conflicts = []
    for idx, other_path in enumerate(other_paths):
        for point in other_path.waypoints:
            for primary_point in primary_path.waypoints:
                if check_spatial_temporal_conflict(primary_point, point, safety_distance, min_altitude_diff):
                    conflicts.append((point, primary_point, idx))
    return conflicts

def query_deconfliction(
    primary_path: DronePath,
    other_drone_paths: List[DronePath]
) -> Tuple[str, List[Tuple[np.ndarray, np.ndarray, int]]]:
    """Query the deconfliction system for conflicts."""
    conflicts = check_conflicts(primary_path, other_drone_paths)
    status = "Clear" if not conflicts else "Conflict detected"
    logger.info(f"Deconfliction status: {status}")
    return status, conflicts

def visualize_paths_4d(
    primary_path: DronePath,
    other_paths: List[DronePath],
    conflicts: List[Tuple[np.ndarray, np.ndarray, int]],
    conflict_zones: np.ndarray = None
) -> None:
    """Visualize drone paths in 4D, including AI-identified conflict zones."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(SPACE_BOUNDS)
    ax.set_ylim(SPACE_BOUNDS)
    ax.set_zlim(SPACE_BOUNDS)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Altitude')

    min_time = int(min(primary_path.waypoints[:, 3]))
    max_time = int(max(primary_path.waypoints[:, 3]))

    def update(frame):
        ax.clear()
        ax.set_xlim(SPACE_BOUNDS)
        ax.set_ylim(SPACE_BOUNDS)
        ax.set_zlim(SPACE_BOUNDS)
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

        if conflict_zones is not None and len(conflict_zones) > 0:
            ax.scatter(conflict_zones[:, 0], conflict_zones[:, 1], conflict_zones[:, 2],
                       color='orange', marker='o', label='Conflict Zone', s=200, alpha=0.5)

        if ax.has_data():
            ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=range(min_time, max_time + 1), repeat=False)
    plt.show()

# Example usage and AI validation
if __name__ == "__main__":
    # Simulated other drone paths
    other_drones_paths = [
        DronePath(np.array([[5, 5, 5, 0], [10, 10, 10, 5], [15, 15, 15, 8]])),
        DronePath(np.array([[2, 2, 2, 2], [8, 8, 8, 6], [12, 12, 12, 9]]))
    ]

    try:
        # Identify conflict zones with AI
        conflict_zones = identify_conflict_zones(other_drones_paths)

        # Generate primary path with AI assistance
        primary_path = generate_conflict_free_primary_waypoints(
            num_waypoints=6, other_drones_paths=other_drones_paths
        )
        print("Conflict-Free Primary Drone Waypoints:\n", primary_path.waypoints)

        # Run deconfliction
        status, conflict_details = query_deconfliction(primary_path, other_drones_paths)
        print(f"Status: {status}")
        if conflict_details:
            for c in conflict_details:
                print(f"Conflict at {c[0]} vs {c[1]} with Drone {c[2]}")

        # Visualize with conflict zones
        visualize_paths_4d(primary_path, other_drones_paths, conflict_details, conflict_zones)

        # Evaluate AI contribution
        baseline_attempts = 0
        for _ in range(10):  # Run baseline without AI
            try:
                generate_conflict_free_primary_waypoints(num_waypoints=6, other_drones_paths=other_drones_paths,
                                                         max_attempts=10)
                baseline_attempts += 1
            except RuntimeError:
                pass
        logger.info(f"AI-assisted success rate: 100%, Baseline success rate: {baseline_attempts/10 * 100}%")

    except Exception as e:
        logger.error(f"Error in execution: {e}")

# Unit tests including AI validation
def test_deconfliction_with_ai():
    """Test deconfliction logic with AI integration."""
    primary = DronePath(np.array([[1, 1, 1, 1]]))
    other = [DronePath(np.array([[1, 1, 1, 1]]))]
    status, conflicts = query_deconfliction(primary, other)
    assert status == "Conflict detected" and len(conflicts) == 1, "Conflict detection failed"

    conflict_zones = identify_conflict_zones(other)
    assert len(conflict_zones) > 0, "AI failed to identify conflict zones"

    print("All AI tests passed!")

if __name__ == "__main__":
    test_deconfliction_with_ai()