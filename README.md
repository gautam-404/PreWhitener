This module is designed for pre-whitening a light curve by iteratively fitting sinusoids to the peaks in the amplitude spectrum. Starting with the highest peak, the function fits and subsequently removes it from the spectrum. This iterative process continues until a specified Signal-to-Noise Ratio (SNR) threshold is reached.

Install with `pip install git+https://github.com/gautam-404/PreWhitener.git`.  
Read the docs [here](https://gautam-404.github.io/PreWhitener).

## Usage

```python
from PreWhitener import PreWhitener

# Create a PreWhitener object by specifying the KIC/TIC/HD ID
pw = PreWhitener(name='HDxxxxx')
pw.auto()
```

