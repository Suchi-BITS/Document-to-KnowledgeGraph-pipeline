"""
Demo Notebook - Knowledge Graph Builder
This is a simplified example showing how to use the modular components.
Save this as a .ipynb file and run in Jupyter.
"""

# Cell 1: Setup and Imports
import sys
sys.path.append('..')  # Add parent directory to path

from src.llm.client import LLMClient
from src.text_processing.chunker import TextChunker
from src.text_processing.normalizer import TripleNormalizer
from src.extraction.extractor import TripleExtractor
from src.graph.builder import GraphBuilder
from src.visualization.cytoscape_viz import CytoscapeVisualizer
from config.settings import settings
import pandas as pd

print("Libraries imported successfully!")

# Cell 2: Validate Configuration
try:
    settings.validate()
    print("✓ Configuration validated")
    print(f"Using model: {settings.LLM_MODEL_NAME}")
except Exception as e:
    print(f"Configuration error: {e}")

# Cell 3: Define Input Text
unstructured_text = """
Marie Curie, born Maria Skłodowska in Warsaw, Poland, was a pioneering physicist and chemist.
She conducted groundbreaking research on radioactivity. Together with her husband, Pierre Curie,
she discovered the elements polonium and radium. Marie Curie was the first woman to win a Nobel Prize,
the first person and only woman to win the Nobel Prize twice, and the only person to win the Nobel Prize
in two different scientific fields. She won the Nobel Prize in Physics in 1903 with Pierre Curie
and Henri Becquerel. Later, she won the Nobel Prize in Chemistry in 1911 for her work on radium and
polonium.
"""

print("Input text loaded")
print(f"Characters: {len(unstructured_text)}")
print(f"Words: {len(unstructured_text.split())}")

# Cell 4: Chunk the Text
chunker = TextChunker()
chunks = chunker.chunk_text(unstructured_text)

print(f"\nCreated {len(chunks)} chunks")
chunks_df = pd.DataFrame(chunks)
display(chunks_df[['chunk_number', 'word_count', 'text']])

# Cell 5: Extract Triples
llm_client = LLMClient()
extractor = TripleExtractor(llm_client)

print("Extracting triples from chunks...")
all_triples = extractor.extract_from_chunks(chunks)

print(f"\nExtracted {len(all_triples)} triples")
if all_triples:
    triples_df = pd.DataFrame(all_triples)
    display(triples_df.head(10))

# Cell 6: Normalize and Deduplicate
normalizer = TripleNormalizer()
normalized_triples = normalizer.normalize_and_deduplicate(all_triples)

stats = normalizer.get_statistics(len(all_triples))
print(f"\nNormalization Statistics:")
print(f"  Original: {stats['original_count']}")
print(f"  Removed (empty): {stats['empty_removed']}")
print(f"  Removed (duplicates): {stats['duplicates_removed']}")
print(f"  Final: {stats['final_count']}")

display(pd.DataFrame(normalized_triples))

# Cell 7: Build Graph
builder = GraphBuilder()
graph = builder.build_graph(normalized_triples)

graph_stats = builder.get_graph_statistics(graph)
print("\nGraph Statistics:")
for key, value in graph_stats.items():
    print(f"  {key}: {value}")

# Cell 8: Visualize
visualizer = CytoscapeVisualizer()
widget = visualizer.create_widget(graph)

print("\nInteractive graph created!")
print("Interact: Zoom (scroll), Pan (drag), Move nodes, Hover for details")
display(widget)

# Cell 9: Explore Top Nodes
top_nodes = builder.get_top_nodes(graph, n=5)
print("\nTop 5 Most Connected Nodes:")
display(pd.DataFrame(top_nodes))

# Cell 10: Node Details
if top_nodes:
    node_id = top_nodes[0]['node']
    node_info = builder.get_node_info(node_id, graph)
    
    print(f"\nDetails for '{node_id}':")
    print(f"  Total degree: {node_info['degree']}")
    print(f"  Incoming: {node_info['in_degree']}")
    print(f"  Outgoing: {node_info['out_degree']}")
    
    print("\n  Outgoing relationships:")
    for edge in node_info['outgoing_edges'][:5]:
        print(f"    -{edge['relation']}-> {edge['to']}")