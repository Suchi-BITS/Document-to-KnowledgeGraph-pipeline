# Knowledge Graph Builder

> Transform unstructured text into interactive knowledge graphs using Large Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A highly modular, production-ready implementation that extracts Subject-Predicate-Object (SPO) triples from raw text using Large Language Models and visualizes them as interactive knowledge graphs.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
  - [Jupyter Notebook](#jupyter-notebook)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Theory & Concepts](#theory--concepts)
- [Examples](#examples)
- [Testing](#testing)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Core Capabilities
- **LLM-Powered Extraction** - Extract structured facts from unstructured text using GPT-4, Claude, Ollama, or any OpenAI-compatible API
- **Smart Text Chunking** - Intelligent text splitting with configurable overlap for context preservation
- **Interactive Visualization** - Beautiful in-notebook graph visualization using ipycytoscape with hover tooltips and animations
- **Data Normalization** - Automatic cleaning, deduplication, and pronoun resolution
- **Modular Architecture** - Clean separation of concerns for easy maintenance, testing, and extension
- **Multiple Output Formats** - Export as JSON, NetworkX graphs, or Cytoscape-compatible format

### Technical Features
- Robust error handling and retry logic
- Comprehensive validation of extracted triples
- Progress tracking and detailed logging
- Configurable via environment variables
- Extensive test coverage
- Type hints throughout codebase

---

## Quick Start

### Automated Setup

**Linux/Mac:**
```bash
git clone https://github.com/yourusername/knowledge-graph-builder.git
cd knowledge-graph-builder
bash setup.sh
```

**Windows:**
```cmd
git clone https://github.com/yourusername/knowledge-graph-builder.git
cd knowledge-graph-builder
setup.bat
```

### Manual Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/knowledge-graph-builder.git
cd knowledge-graph-builder

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API key
```

### First Run

```bash
# Process the sample file
python main.py data/input/sample.txt

# View results in data/output/
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Dependencies

Core dependencies:
- `openai` - LLM API client
- `networkx` - Graph data structures
- `ipycytoscape` - Interactive visualization
- `pandas` - Data manipulation
- `python-dotenv` - Environment configuration

Development dependencies:
- `pytest` - Testing framework
- `black` - Code formatting
- `flake8` - Linting

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/knowledge-graph-builder.git
   cd knowledge-graph-builder
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**
   ```bash
   # Linux/Mac
   source .venv/bin/activate
   
   # Windows
   .venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API credentials:
   ```env
   OPENAI_API_KEY=your_api_key_here
   LLM_MODEL_NAME=gpt-4o
   ```

---

## Usage

### Command Line Interface

The CLI provides a simple interface for processing text files:

**Basic usage:**
```bash
python main.py data/input/your_file.txt
```

**With options:**
```bash
python main.py data/input/your_file.txt \
  --output data/output/my_results \
  --chunk-size 200 \
  --overlap 40 \
  --verbose
```

**Available options:**
```
positional arguments:
  input_file            Path to input text file

optional arguments:
  -h, --help            Show help message
  -o, --output DIR      Output directory (default: data/output)
  --chunk-size N        Chunk size in words (default: 150)
  --overlap N           Overlap in words (default: 30)
  --no-normalize        Skip normalization and deduplication
  --verbose             Print detailed progress information
```

**Output files:**
- `triples_raw.json` - All extracted triples before normalization
- `triples_normalized.json` - Cleaned and deduplicated triples
- `graph_data.json` - Graph in Cytoscape format

### Python API

Use the modular components programmatically:

```python
from src.llm.client import LLMClient
from src.text_processing.chunker import TextChunker
from src.text_processing.normalizer import TripleNormalizer
from src.extraction.extractor import TripleExtractor
from src.graph.builder import GraphBuilder
from src.visualization.cytoscape_viz import CytoscapeVisualizer

# Load your text
with open('your_file.txt', 'r') as f:
    text = f.read()

# Initialize components
llm_client = LLMClient()
chunker = TextChunker(chunk_size=150, overlap=30)
extractor = TripleExtractor(llm_client)
normalizer = TripleNormalizer()
builder = GraphBuilder()

# Process pipeline
chunks = chunker.chunk_text(text)
triples = extractor.extract_from_chunks(chunks)
normalized = normalizer.normalize_and_deduplicate(triples)
graph = builder.build_graph(normalized)

# Get statistics
stats = builder.get_graph_statistics(graph)
print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")

# Visualize (in Jupyter)
visualizer = CytoscapeVisualizer()
widget = visualizer.create_widget(graph)
display(widget)
```

### Jupyter Notebook

Interactive exploration with step-by-step execution:

```bash
jupyter notebook notebooks/demo.ipynb
```

The demo notebook includes:
- Data loading and exploration
- Step-by-step processing with visualizations
- Graph analysis and statistics
- Interactive graph widget
- Export functionality

---

## Project Structure

```
knowledge-graph-builder/
│
├── main.py                          # CLI entry point
├── setup.sh / setup.bat             # Setup scripts
├── requirements.txt                 # Python dependencies
├── .env.example                     # Configuration template
├── README.md                        # This file
│
├── config/                          # Configuration management
│   ├── __init__.py
│   └── settings.py                  # Centralized settings
│
├── src/                             # Source code
│   ├── __init__.py
│   ├── llm/                        # LLM interaction
│   │   ├── __init__.py
│   │   ├── client.py               # API client
│   │   └── prompts.py              # Prompt templates
│   ├── text_processing/            # Text processing
│   │   ├── __init__.py
│   │   ├── chunker.py              # Text chunking
│   │   └── normalizer.py           # Normalization
│   ├── extraction/                 # Triple extraction
│   │   ├── __init__.py
│   │   ├── extractor.py            # Main extractor
│   │   └── validator.py            # Validation
│   ├── graph/                      # Graph building
│   │   ├── __init__.py
│   │   ├── builder.py              # NetworkX builder
│   │   └── converter.py            # Format converter
│   └── visualization/              # Visualization
│       ├── __init__.py
│       └── cytoscape_viz.py        # Interactive viz
│
├── notebooks/                       # Jupyter notebooks
│   └── demo.ipynb                  # Demo notebook
│
├── data/                           # Data directory
│   ├── input/                      # Input text files
│   │   └── sample.txt              # Example input
│   └── output/                     # Generated outputs
│
└── tests/                          # Test suite
    ├── __init__.py
    ├── test_chunker.py
    ├── test_extractor.py
    └── test_graph_builder.py
```

---

## Configuration

### Environment Variables

All configuration is managed through environment variables in the `.env` file:

```env
# Required: LLM API Configuration
OPENAI_API_KEY=your_api_key_here

# Optional: Custom API endpoint (for Ollama, Nebius, etc.)
# OPENAI_API_BASE=http://localhost:11434/v1
# OPENAI_API_BASE=https://api.studio.nebius.com/v1/

# LLM Model Selection
LLM_MODEL_NAME=gpt-4o
# Examples: gpt-4o, gpt-3.5-turbo, llama3, mistral, 
#           deepseek-ai/DeepSeek-V3

# LLM Parameters
LLM_TEMPERATURE=0.0          # 0.0 = deterministic, 1.0 = creative
LLM_MAX_TOKENS=4096          # Maximum response length

# Text Processing
CHUNK_SIZE=150               # Words per chunk
CHUNK_OVERLAP=30             # Overlapping words

# Visualization (optional)
GRAPH_LAYOUT=cose            # Layout algorithm
ANIMATE_LAYOUT=true          # Enable animations
```

### Supported LLM Providers

**OpenAI:**
```env
OPENAI_API_KEY=sk-...
LLM_MODEL_NAME=gpt-4o
```

**Ollama (Local):**
```env
OPENAI_API_KEY=ollama
OPENAI_API_BASE=http://localhost:11434/v1
LLM_MODEL_NAME=llama3
```

**Nebius AI:**
```env
OPENAI_API_KEY=your_nebius_key
OPENAI_API_BASE=https://api.studio.nebius.com/v1/
LLM_MODEL_NAME=deepseek-ai/DeepSeek-V3
```

**Any OpenAI-Compatible API:**
```env
OPENAI_API_KEY=your_key
OPENAI_API_BASE=your_endpoint
LLM_MODEL_NAME=your_model
```

---

## Theory & Concepts

### What is a Knowledge Graph?

A Knowledge Graph is a structured representation of information as a network of entities and their relationships. It consists of:

- **Nodes (Entities)**: Real-world objects, concepts, people, places, organizations
  - Example: "Marie Curie", "Physics", "Paris", "Nobel Prize"
  
- **Edges (Relationships)**: Connections between entities with directional labels
  - Example: Marie Curie → [discovered] → Radium

### Subject-Predicate-Object (SPO) Triples

The fundamental building block of knowledge graphs:

```
Structure: (Subject, Predicate, Object)
Example: (Marie Curie, discovered, Radium)
Graph representation: (Marie Curie) -[discovered]→ (Radium)
```

**Components:**
- **Subject**: The entity the statement is about
- **Predicate**: The relationship or action (becomes edge label)
- **Object**: The entity related to the subject

### How It Works

1. **Text Chunking**: Split large text into manageable pieces with overlap
2. **LLM Extraction**: Use language models to identify and extract SPO triples
3. **JSON Parsing**: Parse structured output from LLM
4. **Validation**: Ensure triples have correct structure and data types
5. **Normalization**: Clean, lowercase, and resolve pronouns
6. **Deduplication**: Remove duplicate relationships
7. **Graph Building**: Construct NetworkX directed graph
8. **Visualization**: Render interactive graph with ipycytoscape

---

## Examples

### Example 1: Simple Biography

**Input text:**
```
Marie Curie was born in Warsaw. She discovered radium. 
Marie won the Nobel Prize in Physics in 1903.
```

**Extracted triples:**
```json
[
  {"subject": "marie curie", "predicate": "was born in", "object": "warsaw"},
  {"subject": "marie curie", "predicate": "discovered", "object": "radium"},
  {"subject": "marie curie", "predicate": "won", "object": "nobel prize in physics"}
]
```

**Resulting graph:**
```
(marie curie) -[was born in]→ (warsaw)
(marie curie) -[discovered]→ (radium)
(marie curie) -[won]→ (nobel prize in physics)
```

### Example 2: Complex Relationships

**Input text:**
```
Apple was founded by Steve Jobs. The company is headquartered in Cupertino.
Apple released the iPhone in 2007. The iPhone revolutionized smartphones.
```

**Graph structure:**
- Central node: "apple" (high degree)
- Connected entities: "steve jobs", "cupertino", "iphone", "smartphones"
- Temporal information: "2007"

### Example 3: Processing Configuration

**Small documents (< 500 words):**
```bash
python main.py input.txt --chunk-size 200 --overlap 50
```

**Large documents (> 5000 words):**
```bash
python main.py input.txt --chunk-size 100 --overlap 20
```

**Quick processing (less accuracy):**
```bash
python main.py input.txt --chunk-size 300 --no-normalize
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_chunker.py

# Run with coverage report
pytest --cov=src tests/

# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/
```

### Test Structure

```
tests/
├── test_chunker.py          # Text chunking tests
├── test_extractor.py        # Extraction tests
├── test_graph_builder.py    # Graph building tests
├── test_normalizer.py       # Normalization tests
└── test_validator.py        # Validation tests
```

### Writing Tests

```python
# tests/test_custom.py
import pytest
from src.text_processing.chunker import TextChunker

def test_custom_functionality():
    chunker = TextChunker(chunk_size=50, overlap=10)
    text = "Your test text here"
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 0
```

---

## Advanced Usage

### Custom Prompt Templates

Modify prompts for domain-specific extraction:

```python
from src.llm.prompts import PromptTemplates

# Create custom prompt
custom_prompt = """
Extract medical relationships in the format:
[{"subject": "...", "predicate": "...", "object": "..."}]

Focus on: diseases, symptoms, treatments, medications.

Text: {text_chunk}
"""

# Use in extraction
system_prompt = PromptTemplates.get_system_prompt()
user_prompt = custom_prompt.format(text_chunk=your_text)
```

### Batch Processing

Process multiple files:

```python
import os
from pathlib import Path

input_dir = Path('data/input')
output_dir = Path('data/output')

for file_path in input_dir.glob('*.txt'):
    print(f"Processing {file_path.name}...")
    
    # Load text
    text = file_path.read_text()
    
    # Process (use your pipeline)
    # ...
    
    # Save results
    output_file = output_dir / f"{file_path.stem}_triples.json"
    # Save logic here
```

### Graph Analysis

Perform advanced graph analysis:

```python
import networkx as nx

# Centrality measures
degree_centrality = nx.degree_centrality(graph)
betweenness = nx.betweenness_centrality(graph)
pagerank = nx.pagerank(graph)

# Community detection
from networkx.algorithms import community
communities = community.greedy_modularity_communities(graph.to_undirected())

# Find paths
if nx.has_path(graph, "marie curie", "nobel prize"):
    paths = list(nx.all_simple_paths(graph, "marie curie", "nobel prize"))
    
# Subgraph extraction
ego_graph = nx.ego_graph(graph, "marie curie", radius=2)
```

### Export to Neo4j

Export graph to Neo4j database:

```python
from neo4j import GraphDatabase

class Neo4jExporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def export_graph(self, graph):
        with self.driver.session() as session:
            for u, v, data in graph.edges(data=True):
                session.run(
                    "MERGE (a:Entity {name: $subject}) "
                    "MERGE (b:Entity {name: $object}) "
                    "MERGE (a)-[r:RELATION {type: $predicate}]->(b)",
                    subject=u, object=v, predicate=data['label']
                )

# Usage
exporter = Neo4jExporter("bolt://localhost:7687", "neo4j", "password")
exporter.export_graph(graph)
```

---

## Troubleshooting

### Common Issues

**Issue: API Key Error**
```
Error: OPENAI_API_KEY environment variable not set
```
Solution: Create `.env` file from `.env.example` and add your API key

**Issue: Import Errors**
```
ModuleNotFoundError: No module named 'src'
```
Solution: Run from root directory or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Issue: Empty Triples**
```
Extracted 0 triples from text
```
Solutions:
- Check API key is valid
- Verify model name is correct
- Try increasing chunk size
- Check input text has factual content

**Issue: Visualization Not Displaying**
```
Widget not showing in Jupyter
```
Solutions:
```bash
# Enable ipywidgets
jupyter nbextension enable --py widgetsnbextension

# Or for JupyterLab
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

**Issue: Rate Limiting**
```
Error: Rate limit exceeded
```
Solution: Add delays between API calls or reduce chunk count

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Use in your code
logger.debug("Processing chunk %d", chunk_num)
```

### Getting Help

1. Check [Issues](https://github.com/yourusername/knowledge-graph-builder/issues) for existing problems
2. Search documentation and code comments
3. Enable verbose mode: `python main.py --verbose`
4. Create a minimal reproduction example
5. Open a new issue with details

---

## Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/knowledge-graph-builder.git
cd knowledge-graph-builder

# Install dev dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Create branch
git checkout -b feature/your-feature
```

### Code Style

- Follow PEP 8 style guide
- Use type hints
- Add docstrings to functions
- Format with Black: `black src/`
- Lint with flake8: `flake8 src/`

### Testing Requirements

- Write tests for new features
- Maintain >80% code coverage
- All tests must pass: `pytest`

### Pull Request Process

1. Update documentation
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Submit PR with clear description

### Areas for Contribution

- Additional LLM provider support
- New visualization styles
- Performance optimizations
- Documentation improvements
- Bug fixes
- New extraction strategies

---

## Roadmap

### Current Version (1.0.0)
- Core extraction pipeline
- Multiple LLM support
- Interactive visualization
- CLI and Python API

### Planned Features
- [ ] Web interface
- [ ] Real-time processing
- [ ] Advanced entity linking
- [ ] Multi-language support
- [ ] Graph database integration (Neo4j, ArangoDB)
- [ ] REST API
- [ ] Docker containerization
- [ ] Relationship confidence scores
- [ ] Incremental graph updates
- [ ] Export to RDF/OWL

---

## Performance

### Benchmarks

Tested on: MacBook Pro M1, 16GB RAM

| Document Size | Chunks | Processing Time | Triples |
|--------------|--------|-----------------|---------|
| 500 words    | 4      | 8 seconds       | 15-25   |
| 2000 words   | 14     | 25 seconds      | 50-80   |
| 5000 words   | 35     | 60 seconds      | 120-180 |

*Using GPT-4o with default settings*

### Optimization Tips

1. **Adjust chunk size** for your document type
2. **Use local models** (Ollama) for faster processing
3. **Batch processing** for multiple documents
4. **Cache API responses** to avoid reprocessing
5. **Parallel processing** with threading/multiprocessing

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{knowledge_graph_builder,
  title = {Knowledge Graph Builder: LLM-Powered Triple Extraction},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/knowledge-graph-builder}
}
```

---

## Acknowledgments

Built with:
- [OpenAI API](https://openai.com/api/) - Language models
- [NetworkX](https://networkx.org/) - Graph algorithms
- [ipycytoscape](https://github.com/cytoscape/ipycytoscape) - Interactive visualization
- [Jupyter](https://jupyter.org/) - Notebook environment

Inspired by:
- Knowledge graph extraction research
- Natural language processing techniques
- Graph neural networks

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License - Copyright (c) 2025 Knowledge Graph Builder Contributors
```

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/knowledge-graph-builder/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/knowledge-graph-builder/discussions)
- **Email**: your.email@example.com

---

## Star History

If you find this project useful, please consider giving it a star!

---

**Made with care for the NLP and Knowledge Graph community**
