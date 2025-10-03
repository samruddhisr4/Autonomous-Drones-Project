"""
Topology Utilities

Utility functions for topological data analysis and geometric computations.

Author: Autonomous Drone Detection Team
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import pdist, squareform
import networkx as nx

def compute_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Compute pairwise distance matrix for points."""
    return squareform(pdist(points))

def build_proximity_graph(points: np.ndarray, radius: float) -> nx.Graph:
    """Build proximity graph with edges within given radius."""
    distances = compute_distance_matrix(points)
    graph = nx.Graph()
    
    # Add nodes
    for i in range(len(points)):
        graph.add_node(i, pos=points[i])
    
    # Add edges within radius
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if distances[i, j] <= radius:
                graph.add_edge(i, j, weight=distances[i, j])
    
    return graph

def compute_connected_components(points: np.ndarray, radius: float) -> List[List[int]]:
    """Compute connected components using proximity graph."""
    graph = build_proximity_graph(points, radius)
    components = list(nx.connected_components(graph))
    return [list(comp) for comp in components]

def estimate_cycles(points: np.ndarray, radius: float) -> int:
    """Estimate number of cycles (β₁) in point cloud."""
    graph = build_proximity_graph(points, radius)
    
    # Use fundamental theorem: β₁ = edges - vertices + components
    num_edges = graph.number_of_edges()
    num_vertices = graph.number_of_nodes()
    num_components = nx.number_connected_components(graph)
    
    return max(0, num_edges - num_vertices + num_components)

def compute_convex_hull_area(points: np.ndarray) -> float:
    """Compute area of convex hull of points."""
    if len(points) < 3:
        return 0.0
    
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except:
        return 0.0

def compute_spatial_statistics(points: np.ndarray) -> Dict:
    """Compute various spatial statistics for point cloud."""
    if len(points) == 0:
        return {}
    
    stats = {
        'num_points': len(points),
        'centroid': np.mean(points, axis=0).tolist(),
        'std_dev': np.std(points, axis=0).tolist(),
        'bbox_area': 0,
        'convex_hull_area': 0,
        'avg_nearest_neighbor_dist': 0
    }
    
    if len(points) > 1:
        # Bounding box area
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        stats['bbox_area'] = float(np.prod(max_coords - min_coords))
        
        # Convex hull area
        stats['convex_hull_area'] = compute_convex_hull_area(points)
        
        # Average nearest neighbor distance
        distances = compute_distance_matrix(points)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        nearest_dists = np.min(distances, axis=1)
        stats['avg_nearest_neighbor_dist'] = float(np.mean(nearest_dists))
    
    return stats

def compute_cluster_density(points: np.ndarray, radius: float) -> float:
    """Compute density of point cluster."""
    if len(points) < 2:
        return 0.0
    
    # Count neighbors within radius for each point
    distances = compute_distance_matrix(points)
    neighbor_counts = np.sum(distances <= radius, axis=1) - 1  # Exclude self
    
    return float(np.mean(neighbor_counts))

def compute_spatial_dispersion(points: np.ndarray) -> float:
    """Compute spatial dispersion (coefficient of variation of distances)."""
    if len(points) < 2:
        return 0.0
    
    # Compute all pairwise distances
    distances = pdist(points)
    
    if len(distances) == 0:
        return 0.0
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    return std_dist / mean_dist if mean_dist > 0 else 0.0

def find_spatial_outliers(points: np.ndarray, threshold: float = 2.0) -> List[int]:
    """Find spatial outliers using distance-based method."""
    if len(points) < 3:
        return []
    
    centroid = np.mean(points, axis=0)
    distances_to_centroid = np.linalg.norm(points - centroid, axis=1)
    
    mean_dist = np.mean(distances_to_centroid)
    std_dist = np.std(distances_to_centroid)
    
    outlier_threshold = mean_dist + threshold * std_dist
    outliers = np.where(distances_to_centroid > outlier_threshold)[0]
    
    return outliers.tolist()

def compute_shape_regularity(points: np.ndarray) -> float:
    """Compute shape regularity metric (0=irregular, 1=regular)."""
    if len(points) < 3:
        return 0.0
    
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        
        # Compare convex hull area to bounding box area
        hull_area = hull.volume
        
        # Bounding box area
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        bbox_area = np.prod(max_coords - min_coords)
        
        if bbox_area > 0:
            return float(hull_area / bbox_area)
        else:
            return 0.0
    except:
        return 0.0

def compute_topological_features(points: np.ndarray, radius: float) -> Dict:
    """Compute comprehensive topological features."""
    features = {
        'num_points': len(points),
        'connected_components': 0,
        'estimated_cycles': 0,
        'density': 0,
        'dispersion': 0,
        'regularity': 0,
        'spatial_stats': {}
    }
    
    if len(points) == 0:
        return features
    
    # Basic topological features
    components = compute_connected_components(points, radius)
    features['connected_components'] = len(components)
    features['estimated_cycles'] = estimate_cycles(points, radius)
    
    # Spatial features
    features['density'] = compute_cluster_density(points, radius)
    features['dispersion'] = compute_spatial_dispersion(points)
    features['regularity'] = compute_shape_regularity(points)
    features['spatial_stats'] = compute_spatial_statistics(points)
    
    return features