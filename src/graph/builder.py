"""
Graph building module for creating NetworkX graphs from SPO triples.
Constructs directed graphs with nodes and labeled edges.
"""

import networkx as nx
from typing import List, Dict, Any


class GraphBuilder:
    """Builds NetworkX directed graphs from SPO triples."""
    
    def __init__(self):
        """Initialize the graph builder."""
        self.graph = None
    
    def build_graph(self, triples: List[Dict[str, str]]) -> nx.DiGraph:
        """
        Build a directed graph from a list of SPO triples.
        
        Args:
            triples: List of triple dictionaries with 'subject', 'predicate', 'object'
            
        Returns:
            NetworkX DiGraph with nodes and labeled edges
        """
        # Create empty directed graph
        self.graph = nx.DiGraph()
        
        if not triples:
            return self.graph
        
        # Add edges (nodes are added automatically)
        for triple in triples:
            subject = triple['subject']
            predicate = triple['predicate']
            obj = triple['object']
            
            # Add edge with predicate as label
            self.graph.add_edge(subject, obj, label=predicate)
        
        return self.graph
    
    def get_graph_statistics(self, graph: nx.DiGraph = None) -> Dict[str, Any]:
        """
        Get statistics about the graph.
        
        Args:
            graph: NetworkX graph (uses internal graph if not provided)
            
        Returns:
            Dictionary with graph statistics
        """
        if graph is None:
            graph = self.graph
        
        if graph is None or graph.number_of_nodes() == 0:
            return {
                'num_nodes': 0,
                'num_edges': 0,
                'density': 0,
                'is_connected': False,
                'num_components': 0,
            }
        
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        stats = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': nx.density(graph),
            'is_weakly_connected': nx.is_weakly_connected(graph),
        }
        
        if not stats['is_weakly_connected']:
            stats['num_weakly_connected_components'] = nx.number_weakly_connected_components(graph)
        
        # Add degree statistics
        degrees = dict(graph.degree())
        if degrees:
            stats['avg_degree'] = sum(degrees.values()) / len(degrees)
            stats['max_degree'] = max(degrees.values())
            stats['min_degree'] = min(degrees.values())
        
        return stats
    
    def get_top_nodes(
        self,
        graph: nx.DiGraph = None,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get the top N nodes by degree (most connected).
        
        Args:
            graph: NetworkX graph (uses internal graph if not provided)
            n: Number of top nodes to return
            
        Returns:
            List of dictionaries with node and degree information
        """
        if graph is None:
            graph = self.graph
        
        if graph is None or graph.number_of_nodes() == 0:
            return []
        
        # Get degrees
        degrees = dict(graph.degree())
        
        # Sort by degree and get top N
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:n]
        
        return [
            {'node': node, 'degree': degree}
            for node, degree in sorted_nodes
        ]
    
    def get_node_info(self, node_id: str, graph: nx.DiGraph = None) -> Dict[str, Any]:
        """
        Get detailed information about a specific node.
        
        Args:
            node_id: ID of the node
            graph: NetworkX graph (uses internal graph if not provided)
            
        Returns:
            Dictionary with node information
        """
        if graph is None:
            graph = self.graph
        
        if graph is None or node_id not in graph:
            return {'error': 'Node not found'}
        
        # Get neighbors
        predecessors = list(graph.predecessors(node_id))
        successors = list(graph.successors(node_id))
        
        # Get edges
        incoming_edges = [
            {'from': pred, 'relation': graph[pred][node_id]['label']}
            for pred in predecessors
        ]
        outgoing_edges = [
            {'to': succ, 'relation': graph[node_id][succ]['label']}
            for succ in successors
        ]
        
        return {
            'node_id': node_id,
            'degree': graph.degree(node_id),
            'in_degree': graph.in_degree(node_id),
            'out_degree': graph.out_degree(node_id),
            'num_predecessors': len(predecessors),
            'num_successors': len(successors),
            'incoming_edges': incoming_edges,
            'outgoing_edges': outgoing_edges,
        }