import pickle
import numpy as np
from Levenshtein import distance as levenshtein_distance  # Install via `pip install python-Levenshtein`

def read_pickle(file_path):
    """
    Reads a pickle file and returns the stored object.
    
    :param file_path: str, path to the pickle file
    :return: object stored in the pickle file
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

    
# DNA Nucleotide Mapping (including 'N' for unknown bases)
DNA_MAP = {0: 'N', 1: 'A', 2: 'C', 3: 'G', 4: 'T'}  # Mapping from bytes to DNA

# Function: Convert a byte sequence to a DNA string
def bytes_to_dna(byte_seq):
    return ''.join(DNA_MAP.get(b, 'N') for b in byte_seq)  # 'N' for unknown bases

# Function: Convert DNA sequence back to bytes
def dna_to_bytes(dna_seq):
    reverse_map = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}  # Reverse mapping
    return bytes([reverse_map.get(nuc, 0) for nuc in dna_seq])  # Default 0 ('N') for unknown bases

def enc_to_int_array(estr) -> np.ndarray:
    """
    Convert a byte-encoded DNA sequence into an array of integers.
    If input is a string, encode it to bytes before processing.
    """
    if isinstance(estr, str):  # Convert string to bytes
        estr = estr.encode('utf-8')
    return np.frombuffer(estr, dtype=np.uint8)

def pad_sequences(sequences, max_length):
    """
    Pad sequences with zeros to ensure uniform length.
    """
    return np.array([np.pad(seq, (0, max_length - len(seq)), mode='constant') for seq in sequences])


# Function: Convert DNA sequence to numerical array
def dna_to_int_array(dna_seq, max_length=20):
    dna_seq = dna_seq[2:-1]
    byte_seq = dna_to_bytes(dna_seq)
    arr = list(byte_seq)[:max_length]
    #arr += [0] * (max_length - len(arr))  # Pad to max_length
    return np.array(arr, dtype=np.float32)

# DNA Sequence Metrics
def dna_similarity(pred_seq, true_seq):
    """Compute DNA sequence similarity metrics."""
    lev_dist = levenshtein_distance(pred_seq, true_seq)
    hamming_dist = sum(1 for p, t in zip(pred_seq, true_seq) if p != t) if len(pred_seq) == len(true_seq) else None
    seq_identity = (1 - (lev_dist / max(len(pred_seq), len(true_seq)))) * 100
    return lev_dist, hamming_dist, seq_identity