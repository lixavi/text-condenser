# TextCondenser 

TextCondenser is a Python-based text summarization engine that uses Transformer models to distill key information from extensive text sources.

## Usage

1. Ensure you have Python 3.x installed on your system.
2. Install the required dependencies using the following command:
   ```pip install -r requirements.txt```
3. Place your input text in the `data/input_text.txt` file.
4. Run the `main.py` script to generate a summary:
   ```python main.py```
5. The generated summary will be saved in `data/summaries/summary.txt`.

## Dependencies

- `transformers>=4.11.3`: For Transformer model implementation.
- `torch>=1.10.0`: PyTorch deep learning library.
- `numpy>=1.21.2`: Fundamental package for numerical computing.
