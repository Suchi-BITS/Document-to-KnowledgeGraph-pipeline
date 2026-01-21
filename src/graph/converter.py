"""
Graph data converter for transforming NetworkX graphs to Cytoscape format.
Handles node and edge data conversion with styling information.
"""

import networkx as nx
from typing import List, Dict, Any
from config.settings import settings


class CytoscapeConverter:
    """Converts NetworkX graphs to Cytoscape-compatible format."""
    
    def __init__(self):
        """Initialize the converter."""
        self.node_min_size = settings.NODE_MIN_SIZE
        self.node_max_size_factor = settings.NODE_MAX_SIZE_FACTOR
    
    def convert_graph(self, graph: nx.DiGraph) -> Dict[str, List[Dict]]:
        """
        Convert a NetworkX graph to Cytoscape format.
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            Dictionary with 'nodes' and 'edges' lists in Cytoscape format
        """
        if graph is None or graph.number_of_nodes() == 0:
            return {'nodes': [], 'edges': []}
        
        nodes = self._convert_nodes(graph)
        edges = self._convert_edges(graph)
        
        return {'nodes': nodes, 'edges': edges}
    
    def _convert_nodes(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Convert NetworkX nodes to Cytoscape format.
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            List of node dictionaries in Cytoscape format
        """
        nodes = []
        
        # Calculate degrees for sizing
        node_degrees = dict(graph.degree())
        max_degree = max(node_degrees.values()) if node_degrees else 1
        
        for node_id in graph.nodes():
            degree = node_degrees.get(node_id, 0)
            
            # Calculate node size based on degree
            if max_degree > 0:
                node_size = self.node_min_size + (degree / max_degree) * self.node_max_size_factor
            else:
                node_size = self.node_min_size
            
            # Create display label (wrap spaces with newlines)
            display_label = str(node_id).replace(' ', '\n')
            
            nodes.append({
                'data': {
                    'id': str(node_id),
                    'label': display_label,
                    'degree': degree,
                    'size': node_size,
                    'tooltip_text': f"Entity: {str(node_id)}\nDegree: {degree}"
                }
            })
        
        return nodes
    
    def _convert_edges(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Convert NetworkX edges to Cytoscape format.
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            List of edge dictionaries in Cytoscape format
        """
        edges = []
        edge_count = 0
        
        for u, v, data in graph.edges(data=True):
            edge_id = f"edge_{edge_count}"
            predicate_label = data.get('label', '')
            
            edges.append({
                'data': {
                    'id': edge_id,
                    'source': str(u),
                    'target': str(v),
                    'label': predicate_label,
                    'tooltip_text': f"Relationship: {predicate_label}"
                }
            })
            
            edge_count += 1
        
        return edges
    
    def get_conversion_statistics(
        self,
        cytoscape_data: Dict[str, List[Dict]]
    ) -> Dict[str, int]:
        """
        Get statistics about converted data.
        
        Args:
            cytoscape_data: Converted Cytoscape data
            
        Returns:
            Dictionary with conversion statistics
        """
        return {
            'total_nodes': len(cytoscape_data.get('nodes', [])),
            'total_edges': len(cytoscape_data.get('edges', [])),
        }