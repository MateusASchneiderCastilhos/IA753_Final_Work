# IA753 Final Work

A Python project for IA753 final work assignment.

## Setup

### Prerequisites
- Miniconda/Anaconda with conda-forge channel
- Python 3.13+

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd IA753_Final_Work
```

2. Activate the conda environment:
```bash
conda activate ia753
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Tests

```bash
pytest tests/
```

### Code Formatting

Format code with black:
```bash
black src/ tests/
```

Check formatting without changes:
```bash
black --check src/ tests/
```

## Project Structure

```
src/ia753_project/     - Main package code
tests/                 - Unit tests
```

## License

MIT
