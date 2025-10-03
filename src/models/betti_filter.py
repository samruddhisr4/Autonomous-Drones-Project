"""
Betti Number Filtering Module

This module implements topological data analysis using Betti numbers 
to filter false positive detections in drone imagery.

Betti numbers capture topological features:
- β₀: Number of connected components
- β₁: Number of holes/loops  
- β₂: Number of voids/cavities

Author: Autonomous Drone Detection Team
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
import logging
from pathlib import Path

# Try to import gudhi for advanced topological analysis
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    logging.warning("gudhi not available, using simplified topological analysis")

logger = logging.getLogger(__name__)

class BettiNumberFilter:
    """
    Betti number-based filtering for drone detection false positives.
    
    The key insight is that drone formations and legitimate objects should have
    specific topological characteristics that differ from noise or false positives.
    """
    
    def __init__(self, 
                 max_distance: float = 100.0,
                 min_cluster_size: int = 2,
                 connectivity_radius: float = 50.0,
                 expected_betti_0: Tuple[int, int] = (1, 10),
                 expected_betti_1: Tuple[int, int] = (0, 3),
                 confidence_weight: float = 0.3,
                 spatial_weight: float = 0.7):
        """
        Initialize Betti number filter.
        
        Args:
            max_distance: Maximum distance for spatial clustering
            min_cluster_size: Minimum cluster size for DBSCAN
            connectivity_radius: Radius for building simplicial complex
            expected_betti_0: Expected range for β₀ (connected components)
            expected_betti_1: Expected range for β₁ (holes/loops)
            confidence_weight: Weight for confidence in filtering decision
            spatial_weight: Weight for spatial distribution in filtering
        """
        self.max_distance = max_distance
        self.min_cluster_size = min_cluster_size
        self.connectivity_radius = connectivity_radius
        self.expected_betti_0 = expected_betti_0
        self.expected_betti_1 = expected_betti_1
        self.confidence_weight = confidence_weight
        self.spatial_weight = spatial_weight
        
        # Statistics tracking
        self.stats = {
            'total_detections': 0,
            'filtered_detections': 0,
            'clusters_analyzed': 0,
            'topology_computed': 0
        }
    
    def extract_detection_features(self, detections: List[Dict]) -> np.ndarray:
        """
        Extract spatial features from detections.
        
        Args:
            detections: List of detection dictionaries with bbox, confidence, class_id
            
        Returns:
            Feature matrix [N, 4] with [x_center, y_center, confidence, area]
        """
        if not detections:
            return np.empty((0, 4))
        
        features = []
        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            confidence = det['confidence']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            features.append([x_center, y_center, confidence, area])
        
        return np.array(features)
    
    def spatial_clustering(self, features: np.ndarray) -> np.ndarray:
        """
        Perform spatial clustering of detections.
        
        Args:
            features: Feature matrix [N, 4]
            
        Returns:
            Cluster labels
        """
        if len(features) < self.min_cluster_size:
            return np.array([-1] * len(features))
        
        # Use only spatial coordinates for clustering
        spatial_coords = features[:, :2]
        
        # DBSCAN clustering
        clustering = DBSCAN(
            eps=self.max_distance, 
            min_samples=self.min_cluster_size
        )
        
        cluster_labels = clustering.fit_predict(spatial_coords)
        
        return cluster_labels
    
    def build_simplicial_complex(self, points: np.ndarray) -> 'gudhi.SimplexTree':
        """
        Build simplicial complex from point cloud.
        
        Args:
            points: Point coordinates [N, 2]
            
        Returns:
            Simplicial complex
        """
        if not GUDHI_AVAILABLE:
            raise RuntimeError("gudhi is required for simplicial complex construction")
        
        # Create Rips complex
        rips_complex = gudhi.RipsComplex(
            points=points,
            max_edge_length=self.connectivity_radius
        )
        
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        
        return simplex_tree
    
    def compute_betti_numbers_gudhi(self, points: np.ndarray) -> Dict[int, int]:
        """
        Compute Betti numbers using gudhi library.
        
        Args:
            points: Point coordinates [N, 2]
            
        Returns:
            Dictionary mapping dimension to Betti number
        """
        try:
            simplex_tree = self.build_simplicial_complex(points)
            persistence = simplex_tree.persistence()
            
            # Count Betti numbers
            betti_numbers = {0: 0, 1: 0, 2: 0}
            
            for dimension, (birth, death) in persistence:
                if death == float('inf'):  # Infinite persistence
                    betti_numbers[dimension] += 1
            
            return betti_numbers
            
        except Exception as e:
            logger.warning(f"Failed to compute Betti numbers with gudhi: {e}")
            return self.compute_betti_numbers_graph(points)
    
    def compute_betti_numbers_graph(self, points: np.ndarray) -> Dict[int, int]:
        """
        Compute Betti numbers using graph-based approach (fallback).
        
        Args:
            points: Point coordinates [N, 2]
            
        Returns:
            Dictionary mapping dimension to Betti number
        """
        if len(points) < 2:
            return {0: len(points), 1: 0, 2: 0}
        
        # Build proximity graph
        distances = squareform(pdist(points))
        graph = nx.Graph()
        
        # Add nodes
        for i in range(len(points)):
            graph.add_node(i)
        
        # Add edges within connectivity radius
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if distances[i, j] <= self.connectivity_radius:
                    graph.add_edge(i, j)
        
        # Compute β₀ (connected components)
        betti_0 = nx.number_connected_components(graph)
        
        # Estimate β₁ (cycles) using fundamental theorem of graph theory
        # For planar graphs: β₁ = edges - vertices + components
        num_edges = graph.number_of_edges()
        num_vertices = graph.number_of_nodes()
        betti_1 = max(0, num_edges - num_vertices + betti_0)
        
        return {0: betti_0, 1: betti_1, 2: 0}
    
    def compute_betti_numbers(self, points: np.ndarray) -> Dict[int, int]:
        """
        Compute Betti numbers for point cloud.
        
        Args:
            points: Point coordinates [N, 2]
            
        Returns:
            Dictionary mapping dimension to Betti number
        """
        if GUDHI_AVAILABLE:
            return self.compute_betti_numbers_gudhi(points)
        else:
            return self.compute_betti_numbers_graph(points)
    
    def evaluate_cluster_topology(self, cluster_points: np.ndarray, 
                                 cluster_confidences: np.ndarray) -> Dict:
        """
        Evaluate topological properties of a detection cluster.
        
        Args:
            cluster_points: Spatial coordinates [N, 2]
            cluster_confidences: Detection confidences [N]
            
        Returns:
            Topology evaluation results
        """
        if len(cluster_points) < 2:
            return {
                'betti_numbers': {0: len(cluster_points), 1: 0, 2: 0},
                'is_valid': len(cluster_points) > 0,
                'confidence_score': np.mean(cluster_confidences) if len(cluster_confidences) > 0 else 0,
                'topology_score': 0.5
            }
        
        # Compute Betti numbers
        betti_numbers = self.compute_betti_numbers(cluster_points)
        self.stats['topology_computed'] += 1
        
        # Evaluate topology against expected values
        betti_0_valid = self.expected_betti_0[0] <= betti_numbers[0] <= self.expected_betti_0[1]
        betti_1_valid = self.expected_betti_1[0] <= betti_numbers[1] <= self.expected_betti_1[1]
        
        # Compute topology score
        topology_score = 0.0
        if betti_0_valid:
            topology_score += 0.6
        if betti_1_valid:
            topology_score += 0.4
        
        # Adjust score based on confidence
        confidence_score = np.mean(cluster_confidences)
        
        # Combined validity
        is_valid = (topology_score > 0.5) and (confidence_score > 0.3)
        
        return {
            'betti_numbers': betti_numbers,
            'is_valid': is_valid,
            'confidence_score': confidence_score,
            'topology_score': topology_score,
            'betti_0_valid': betti_0_valid,
            'betti_1_valid': betti_1_valid
        }
    
    def filter_detections(self, detections: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Filter detections using Betti number analysis.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            (filtered_detections, filtering_info)
        """
        self.stats['total_detections'] += len(detections)
        
        if not detections:
            return [], {'clusters': [], 'filtered_count': 0}
        
        # Extract features
        features = self.extract_detection_features(detections)
        
        # Spatial clustering
        cluster_labels = self.spatial_clustering(features)
        unique_labels = set(cluster_labels)
        
        filtered_detections = []
        cluster_info = []
        
        for label in unique_labels:
            if label == -1:  # Noise points (not in any cluster)
                # Keep isolated detections if confidence is high enough
                noise_indices = np.where(cluster_labels == label)[0]
                for idx in noise_indices:
                    if detections[idx]['confidence'] > 0.7:  # High confidence threshold for isolated detections
                        filtered_detections.append(detections[idx])
                continue
            
            # Get cluster detections
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_detections = [detections[i] for i in cluster_indices]
            cluster_points = features[cluster_indices, :2]
            cluster_confidences = features[cluster_indices, 2]
            
            # Evaluate cluster topology
            topo_eval = self.evaluate_cluster_topology(cluster_points, cluster_confidences)
            self.stats['clusters_analyzed'] += 1
            
            cluster_info.append({
                'label': int(label),
                'size': len(cluster_detections),
                'topology_eval': topo_eval,
                'detections': cluster_detections
            })
            
            # Keep detections from valid clusters
            if topo_eval['is_valid']:
                filtered_detections.extend(cluster_detections)
        
        self.stats['filtered_detections'] += len(filtered_detections)
        
        filtering_info = {
            'clusters': cluster_info,
            'filtered_count': len(detections) - len(filtered_detections),
            'original_count': len(detections),
            'final_count': len(filtered_detections)
        }
        
        return filtered_detections, filtering_info
    
    def visualize_clusters(self, detections: List[Dict], 
                          filtering_info: Dict,
                          image: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize clustering and filtering results.
        
        Args:
            detections: Original detections
            filtering_info: Results from filter_detections
            image: Background image (optional)
            save_path: Path to save visualization
            
        Returns:
            Visualization image
        """
        if image is None:
            # Create blank image based on detection coordinates
            if not detections:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            
            all_boxes = [det['bbox'] for det in detections]
            max_x = max(box[2] for box in all_boxes)
            max_y = max(box[3] for box in all_boxes)
            image = np.zeros((int(max_y) + 50, int(max_x) + 50, 3), dtype=np.uint8)
        
        vis_image = image.copy()
        
        # Color palette for clusters
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
            (128, 0, 255), (255, 0, 128)
        ]
        
        # Draw clusters
        for i, cluster in enumerate(filtering_info['clusters']):
            color = colors[i % len(colors)]
            label = cluster['label']
            is_valid = cluster['topology_eval']['is_valid']
            
            # Draw cluster detections
            for det in cluster['detections']:
                bbox = det['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box
                thickness = 3 if is_valid else 1
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label
                label_text = f"C{label} ({'✓' if is_valid else '✗'})"
                cv2.putText(vis_image, label_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add legend
        legend_y = 30
        cv2.putText(vis_image, "Betti Number Filtering Results", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        legend_y += 30
        cv2.putText(vis_image, f"Original: {filtering_info['original_count']}", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        legend_y += 20
        cv2.putText(vis_image, f"Filtered: {filtering_info['final_count']}", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        legend_y += 20
        cv2.putText(vis_image, f"Removed: {filtering_info['filtered_count']}", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            logger.info(f"Visualization saved to {save_path}")
        
        return vis_image
    
    def get_statistics(self) -> Dict:
        """Get filtering statistics."""
        total = self.stats['total_detections']
        filtered = self.stats['filtered_detections']
        
        stats = self.stats.copy()
        stats['filter_rate'] = (total - filtered) / total if total > 0 else 0
        stats['retention_rate'] = filtered / total if total > 0 else 0
        
        return stats
    
    def reset_statistics(self):
        """Reset filtering statistics."""
        self.stats = {
            'total_detections': 0,
            'filtered_detections': 0,
            'clusters_analyzed': 0,
            'topology_computed': 0
        }


class AdvancedBettiFilter(BettiNumberFilter):
    """
    Advanced Betti number filter with additional topological features.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.persistent_homology_analysis = True
    
    def compute_persistent_homology(self, points: np.ndarray) -> Dict:
        """
        Compute persistent homology features.
        
        Args:
            points: Point coordinates [N, 2]
            
        Returns:
            Persistent homology analysis results
        """
        if not GUDHI_AVAILABLE:
            logger.warning("gudhi required for persistent homology")
            return {}
        
        try:
            # Build Rips complex with increasing filtration
            rips_complex = gudhi.RipsComplex(points=points)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
            persistence = simplex_tree.persistence()
            
            # Analyze persistence intervals
            ph_features = {
                'total_intervals': len(persistence),
                'dimension_counts': {},
                'persistence_lengths': [],
                'birth_times': [],
                'death_times': []
            }
            
            for dimension, (birth, death) in persistence:
                if dimension not in ph_features['dimension_counts']:
                    ph_features['dimension_counts'][dimension] = 0
                ph_features['dimension_counts'][dimension] += 1
                
                ph_features['birth_times'].append(birth)
                if death != float('inf'):
                    ph_features['death_times'].append(death)
                    ph_features['persistence_lengths'].append(death - birth)
            
            return ph_features
            
        except Exception as e:
            logger.warning(f"Persistent homology computation failed: {e}")
            return {}
    
    def evaluate_cluster_topology(self, cluster_points: np.ndarray, 
                                 cluster_confidences: np.ndarray) -> Dict:
        """Enhanced topology evaluation with persistent homology."""
        # Get base evaluation
        result = super().evaluate_cluster_topology(cluster_points, cluster_confidences)
        
        # Add persistent homology analysis
        if self.persistent_homology_analysis and len(cluster_points) > 3:
            ph_features = self.compute_persistent_homology(cluster_points)
            result['persistent_homology'] = ph_features
            
            # Adjust topology score based on persistence
            if ph_features and 'persistence_lengths' in ph_features:
                avg_persistence = np.mean(ph_features['persistence_lengths']) if ph_features['persistence_lengths'] else 0
                # Longer persistence intervals suggest more stable topological features
                persistence_bonus = min(0.2, avg_persistence / 100.0)
                result['topology_score'] += persistence_bonus
        
        return result


# Utility functions
def create_betti_filter_config(filter_type: str = 'standard') -> Dict:
    """Create configuration for Betti number filter."""
    
    configs = {
        'standard': {
            'max_distance': 100.0,
            'min_cluster_size': 2,
            'connectivity_radius': 50.0,
            'expected_betti_0': (1, 10),
            'expected_betti_1': (0, 3),
            'confidence_weight': 0.3,
            'spatial_weight': 0.7
        },
        'strict': {
            'max_distance': 75.0,
            'min_cluster_size': 3,
            'connectivity_radius': 40.0,
            'expected_betti_0': (1, 5),
            'expected_betti_1': (0, 2),
            'confidence_weight': 0.4,
            'spatial_weight': 0.6
        },
        'permissive': {
            'max_distance': 150.0,
            'min_cluster_size': 1,
            'connectivity_radius': 75.0,
            'expected_betti_0': (1, 15),
            'expected_betti_1': (0, 5),
            'confidence_weight': 0.2,
            'spatial_weight': 0.8
        }
    }
    
    return configs.get(filter_type, configs['standard'])


def load_betti_filter(config_path: str) -> BettiNumberFilter:
    """Load Betti filter from configuration file."""
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    filter_class = AdvancedBettiFilter if config.get('advanced', False) else BettiNumberFilter
    
    return filter_class(**config)