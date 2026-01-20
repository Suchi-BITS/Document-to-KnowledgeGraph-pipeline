# Document to Knowledge Graph Pipeline

Transform unstructured text documents into interactive, structured knowledge graphs using Large Language Models (LLMs). This end-to-end pipeline extracts Subject-Predicate-Object (SPO) triples from raw text and visualizes them as an interactive graph.

## Overview

This project demonstrates a comprehensive approach to knowledge extraction and graph construction. It breaks down complex text into meaningful relationships between entities, enabling better understanding of information structure, connections, and patterns.

### What is a Knowledge Graph?

A Knowledge Graph represents information as a network where:
- **Nodes (Entities)**: Real-world objects, concepts, people, places (e.g., 'Marie Curie', 'Physics', 'Paris')
- **Edges (Relationships)**: Directed connections with labels describing relationships (e.g., 'Marie Curie' --`discovered`--> 'Radium')

### SPO Triples

The fundamental building blocks are Subject-Predicate-Object triples:
- **Subject**: The entity the statement is about
- **Predicate**: The relationship connecting subject and object
- **Object**: The entity related to the subject

Example: "Marie Curie discovered Radium" → (`Marie Curie`, `discovered`, `Radium`)

## Features

- **Intelligent Text Chunking**: Handles long documents by splitting text with configurable overlap
- **LLM-Powered Extraction**: Uses advanced language models to identify factual relationships
- **Multi-Model Support**: Compatible with OpenAI, Ollama, Nebius AI, and other OpenAI-compatible APIs
- **Robust Parsing**: Multiple fallback strategies for JSON extraction
- **Data Normalization**: Automatic deduplication and cleaning of extracted triples
- **Interactive Visualization**: Beautiful, interactive graph visualization with ipycytoscape
- **Step-by-Step Processing**: Granular visibility into each stage of the pipeline
- **Error Handling**: Comprehensive error tracking and recovery mechanisms

## Requirements

```python
openai>=1.0.0
networkx>=2.8
ipycytoscape>=1.3.1
ipywidgets>=8.0.0
pandas>=1.5.0
```

## Installation

```bash
# Install required packages
pip install openai networkx ipycytoscape ipywidgets pandas

# For Jupyter Notebook (classic), enable widgets extension
jupyter nbextension enable --py widgetsnbextension
```

## Configuration

### API Setup

Set up your LLM API credentials using environment variables (recommended):

```bash
# For OpenAI
export OPENAI_API_KEY='your_openai_api_key'

# For Ollama (local)
export OPENAI_API_KEY='ollama'
export OPENAI_API_BASE='http://localhost:11434/v1'

# For Nebius AI
export OPENAI_API_KEY='your_nebius_api_key'
export OPENAI_API_BASE='https://api.studio.nebius.com/v1/'
```

### Model Configuration

Choose your preferred model:

```python
llm_model_name = "gpt-4o"  # OpenAI
# llm_model_name = "llama3"  # Ollama
# llm_model_name = "deepseek-ai/DeepSeek-V3"  # Nebius AI
```

### Processing Parameters

Customize the pipeline behavior:

```python
# Text chunking
chunk_size = 150  # Words per chunk
overlap = 30      # Overlapping words between chunks

# LLM parameters
llm_temperature = 0.0      # Lower = more deterministic
llm_max_tokens = 4096      # Maximum response length
```

## Usage

### Basic Workflow

```python
# 1. Import libraries
import openai
import json
import networkx as nx
import ipycytoscape
import pandas as pd

# 2. Initialize LLM client
client = openai.OpenAI(
    base_url=base_url,
    api_key=api_key
)

# 3. Define your input text
unstructured_text = """
Your document text here...
"""

# 4. Process text into chunks
chunks = create_chunks(unstructured_text, chunk_size=150, overlap=30)

# 5. Extract triples using LLM
all_extracted_triples = []
for chunk in chunks:
    triples = extract_triples(chunk, llm_client)
    all_extracted_triples.extend(triples)

# 6. Normalize and deduplicate
normalized_triples = normalize_and_deduplicate(all_extracted_triples)

# 7. Build knowledge graph
knowledge_graph = nx.DiGraph()
for triple in normalized_triples:
    knowledge_graph.add_edge(
        triple['subject'], 
        triple['object'], 
        label=triple['predicate']
    )

# 8. Visualize interactively
cyto_widget = create_visualization(knowledge_graph)
display(cyto_widget)
```

### Prompt Engineering

The extraction prompt is carefully designed to ensure consistent, structured output:

```python
extraction_system_prompt = """
You are an AI expert specialized in knowledge graph extraction. 
Your task is to identify and extract factual Subject-Predicate-Object (SPO) triples.
Focus on accuracy and adhere strictly to the JSON output format.
"""

extraction_user_prompt_template = """
Extract Subject-Predicate-Object triples from the text.

IMPORTANT RULES:
1. Output ONLY valid JSON array
2. Each element: {"subject": "", "predicate": "", "object": ""}
3. Concise predicates (1-3 words)
4. All values in lowercase
5. Resolve pronouns to entity names
6. No markdown formatting or extra text

Text: {text_chunk}

Output:
"""
```

## Pipeline Stages

### Stage 1: Text Preprocessing
- Load and validate input text
- Calculate basic statistics
- Split into manageable chunks with overlap

### Stage 2: Triple Extraction
- Send chunks to LLM with structured prompts
- Parse JSON responses with multiple fallback strategies
- Track successful extractions and failures
- Validate triple structure

### Stage 3: Data Normalization
- Trim whitespace and convert to lowercase
- Remove empty or invalid entries
- Deduplicate exact matches
- Preserve source chunk information

### Stage 4: Graph Construction
- Create directed graph structure
- Add nodes for unique entities
- Create edges with relationship labels
- Calculate graph metrics

### Stage 5: Visualization
- Convert to Cytoscape format
- Apply visual styling (colors, sizes, labels)
- Configure interactive layout algorithm
- Enable zoom, pan, and node manipulation

## Visualization Features

The interactive graph provides:
- **Node Sizing**: Proportional to connection degree
- **Color Coding**: Highlights central nodes and relationships
- **Hover Tooltips**: Shows entity/relationship details
- **Interactive Layout**: Drag nodes, zoom, pan
- **Selection**: Click to highlight nodes and edges
- **Animations**: Smooth transitions and layout adjustments

## Example Output

For a biography of Marie Curie, the pipeline extracts relationships such as:
- (`marie curie`, `discovered`, `radium`)
- (`marie curie`, `won`, `nobel prize in physics`)
- (`marie curie`, `born in`, `warsaw, poland`)
- (`marie curie`, `established`, `curie institute`)

These are visualized as an interconnected graph showing her scientific achievements, personal history, and legacy.

## Advanced Features

### Error Handling
- API call retries with exponential backoff
- JSON parsing fallbacks (regex extraction)
- Chunk-level failure tracking
- Graceful degradation

### Graph Analytics
```python
# Calculate centrality measures
degree_centrality = nx.degree_centrality(knowledge_graph)
betweenness = nx.betweenness_centrality(knowledge_graph)

# Find shortest paths
path = nx.shortest_path(knowledge_graph, source, target)

# Community detection
communities = nx.community.greedy_modularity_communities(knowledge_graph)
```

### Export Options
```python
# Save as GraphML
nx.write_graphml(knowledge_graph, "knowledge_graph.graphml")

# Export triples to CSV
pd.DataFrame(normalized_triples).to_csv("triples.csv", index=False)

# Save as JSON
with open("graph_data.json", "w") as f:
    json.dump(cytoscape_graph_data, f, indent=2)
```

## Performance Considerations

- **Chunk Size**: Balance between context and API limits (150-200 words recommended)
- **Overlap**: Maintains context across chunks (20-30% of chunk size)
- **Temperature**: Lower values (0.0-0.2) for factual extraction
- **Rate Limiting**: Implement delays for API rate limits
- **Caching**: Store intermediate results to avoid reprocessing

## Troubleshooting

### Common Issues

**LLM returns non-JSON output**:
- Enable `response_format={"type": "json_object"}` in API call
- Use regex fallback to extract JSON arrays
- Adjust prompt to emphasize JSON-only output

**Graph visualization not appearing**:
- Restart kernel and reload ipywidgets extension
- Verify ipycytoscape installation
- Check for JavaScript console errors

**Empty or incomplete extractions**:
- Review LLM prompt clarity
- Increase temperature slightly (0.1-0.2)
- Verify input text quality
- Check chunk size appropriateness

## Future Enhancements

- **Entity Linking**: Connect to external knowledge bases (Wikidata, DBpedia)
- **Relation Clustering**: Group similar predicates
- **Multi-document Support**: Merge graphs from multiple sources
- **Graph Database Integration**: Export to Neo4j, Neptune
- **Real-time Processing**: Stream processing for large document collections
- **Quality Metrics**: Automated evaluation of extraction accuracy
- **Custom Ontologies**: Support domain-specific entity types

## Contributing

Contributions are welcome! Areas for improvement:
- Support for additional LLM providers
- Enhanced visualization options
- Better entity resolution algorithms
- Performance optimizations
- Additional export formats

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with NetworkX for graph operations
- Powered by OpenAI-compatible LLM APIs
- Visualized using ipycytoscape
- Inspired by modern knowledge graph extraction techniques

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{kg_pipeline_2024,
  title={Document to Knowledge Graph Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Document-to-KnowledgeGraph-pipeline}
}
```

## Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the maintainers.

---

**Ready to transform your documents into knowledge graphs? Get started now!**
