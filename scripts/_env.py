"""Common environment setup for all experiment scripts.

Import this FIRST, before any other imports, to prevent
the OpenBLAS/MKL BLAS deadlock that occurs when numpy and
PyTorch CUDA coexist in the same process.
"""

import os
import sys
import warnings
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Add project root and scripts dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
