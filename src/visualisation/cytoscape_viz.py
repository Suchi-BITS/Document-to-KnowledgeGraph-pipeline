"""
Interactive graph visualization using ipycytoscape.
Provides colorful, animated visualization with hover tooltips and selection.
"""

import ipycytoscape
import networkx as nx
from typing import Optional, Dict, List, Any
from config.settings import settings
from src.graph.converter import CytoscapeConverter


class CytoscapeVisualizer:
    """Creates interactive graph visualizations using ipycytoscape."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.converter = CytoscapeConverter()
        self.widget = None
    
    def create_widget(
        self,
        graph: nx.DiGraph,
        apply_style: bool = True
    ) -> ipycytoscape.CytoscapeWidget:
        """
        Create an interactive Cytoscape widget from a NetworkX graph.
        
        Args:
            graph: NetworkX directed graph
            apply_style: Whether to apply the default visual style
            
        Returns:
            Configured CytoscapeWidget
        """
        # Convert graph to Cytoscape format
        cytoscape_data = self.converter.convert_graph(graph)
        
        # Create widget
        self.widget = ipycytoscape.CytoscapeWidget()
        
        # Load graph data
        self.widget.graph.add_graph_from_json(cytoscape_data, directed=True)
        
        # Apply visual style if requested
        if apply_style:
            self._apply_visual_style()
        
        # Set layout
        self._apply_layout()
        
        # Highlight center nodes
        self._highlight_center_nodes()
        
        return self.widget
    
    def _apply_visual_style(self):
        """Apply enhanced colorful and interactive visual style."""
        visual_style = [
            # Default node style
            {
                'selector': 'node',
                'style': {
                    'label': 'data(label)',
                    'width': 'data(size)',
                    'height': 'data(size)',
                    'background-color': '#3498db',  # Bright blue
                    'background-opacity': 0.9,
                    'color': '#ffffff',
                    'font-size': '12px',
                    'font-weight': 'bold',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'text-wrap': 'wrap',
                    'text-max-width': '100px',
                    'text-outline-width': 2,
                    'text-outline-color': '#2980b9',
                    'text-outline-opacity': 0.7,
                    'border-width': 3,
                    'border-color': '#1abc9c',  # Turquoise
                    'border-opacity': 0.9,
                    'shape': 'ellipse',
                    'transition-property': 'background-color, border-color, border-width, width, height',
                    'transition-duration': '0.3s',
                    'tooltip-text': 'data(tooltip_text)'
                }
            },
            # Selected node style
            {
                'selector': 'node:selected',
                'style': {
                    'background-color': '#e74c3c',  # Red
                    'border-width': 4,
                    'border-color': '#c0392b',
                    'text-outline-color': '#e74c3c',
                    'width': 'data(size) * 1.2',
                    'height': 'data(size) * 1.2'
                }
            },
            # Hover node style
            {
                'selector': 'node:hover',
                'style': {
                    'background-color': '#9b59b6',  # Purple
                    'border-width': 4,
                    'border-color': '#8e44ad',
                    'cursor': 'pointer',
                    'z-index': 999
                }
            },
            # Default edge style
            {
                'selector': 'edge',
                'style': {
                    'label': 'data(label)',
                    'width': 2.5,
                    'curve-style': 'bezier',
                    'line-color': '#2ecc71',  # Green
                    'line-opacity': 0.8,
                    'target-arrow-color': '#27ae60',
                    'target-arrow-shape': 'triangle',
                    'arrow-scale': 1.5,
                    'font-size': '10px',
                    'font-weight': 'normal',
                    'color': '#2c3e50',
                    'text-background-opacity': 0.9,
                    'text-background-color': '#ecf0f1',
                    'text-background-shape': 'roundrectangle',
                    'text-background-padding': '3px',
                    'text-rotation': 'autorotate',
                    'edge-text-rotation': 'autorotate',
                    'transition-property': 'line-color, width, target-arrow-color',
                    'transition-duration': '0.3s',
                    'tooltip-text': 'data(tooltip_text)'
                }
            },
            # Selected edge style
            {
                'selector': 'edge:selected',
                'style': {
                    'line-color': '#f39c12',  # Orange
                    'target-arrow-color': '#d35400',
                    'width': 4,
                    'text-background-color': '#f1c40f',
                    'color': '#ffffff',
                    'z-index': 998
                }
            },
            # Hover edge style
            {
                'selector': 'edge:hover',
                'style': {
                    'line-color': '#e67e22',  # Orange
                    'width': 3.5,
                    'cursor': 'pointer',
                    'target-arrow-color': '#d35400',
                    'z-index': 997
                }
            }
        ]
        
        self.widget.set_style(visual_style)
    
    def _apply_layout(self):
        """Apply graph layout algorithm."""
        layout_config = settings.get_layout_config()
        self.widget.set_layout(**layout_config)
    
    def _highlight_center_nodes(self):
        """Add special styling for high-degree center nodes."""
        if not self.widget or len(self.widget.graph.nodes) == 0:
            return
        
        # Find nodes with high degree (>10)
        main_nodes = [
            node.data['id'] for node in self.widget.graph.nodes
            if node.data.get('degree', 0) > 10
        ]
        
        if not main_nodes:
            return
        
        # Get current style
        current_style = self.widget.get_style()
        
        # Add center node styles
        for node_id in main_nodes:
            center_style = {
                'selector': f'node[id = "{node_id}"]',
                'style': {
                    'background-color': '#9b59b6',  # Purple
                    'background-opacity': 0.95,
                    'border-width': 4,
                    'border-color': '#8e44ad',
                    'border-opacity': 1,
                    'text-outline-width': 3,
                    'text-outline-color': '#8e44ad',
                    'font-size': '14px'
                }
            }
            current_style.append(center_style)
        
        # Update style
        self.widget.set_style(current_style)
    
    def get_widget(self) -> Optional[ipycytoscape.CytoscapeWidget]:
        """
        Get the current widget instance.
        
        Returns:
            CytoscapeWidget or None
        """
        return self.widget