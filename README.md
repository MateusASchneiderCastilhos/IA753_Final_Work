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

## Using in Google Colab

You can run this project directly in Google Colab without installing anything locally. Follow these steps:

1. Open [Google Colab](https://colab.research.google.com)

2. In a new cell, clone the repository:
```python
!git clone https://github.com/<your-username>/IA753_Final_Work.git
%cd IA753_Final_Work
```

3. Install dependencies:
```python
!pip install -r requirements.txt
```

4. Import and use the package:
```python
import sys
sys.path.insert(0, '/content/IA753_Final_Work/src')

from ia753_project import your_module
# Use the modules from the project
```

5. If you want to save work, mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
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
