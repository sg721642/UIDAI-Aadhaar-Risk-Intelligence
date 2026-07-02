import pytest
import pandas as pd
from src.data.data_loader import canonicalize_state

def test_canonicalize_state():
    assert canonicalize_state("delhi") == "Delhi"
    assert canonicalize_state("andhra pradesh") == "Andhra Pradesh"
    assert canonicalize_state("dadra & nagar haveli") == "Dadra & Nagar Haveli and Daman & Diu"
    assert canonicalize_state("dadra and nagar haveli and daman and diu") == "Dadra & Nagar Haveli and Daman & Diu"
    assert canonicalize_state("west bengal") == "West Bengal"
    assert canonicalize_state("west bangal") == "West Bengal"
    assert canonicalize_state("pondicherry") == "Puducherry"
    assert canonicalize_state("orissa") == "Odisha"

def test_canonicalize_state_none():
    assert pd.isna(canonicalize_state(None))
