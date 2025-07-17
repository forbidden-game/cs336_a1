import regex as re
import os
import io
import collections
from collections import Counter
import multiprocessing
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_tokens: list[bytes]
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = -1
            for special_token in split_special_tokens:
                found_at = mini_chunk.find(special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize_worker(
    corpus_path: str,
    start_pos: int,
    end_pos: int,
    special_tokens: list[str]
) -> tuple[dict[tuple[bytes, ...], int], dict[tuple[bytes, bytes], int]]:
    
    with open(corpus_path, 'rb') as file:
        file.seek(start_pos)
        chunk = file.read(end_pos-start_pos)
        
    # It's safer to decode with an error handler
    chunk_str = chunk.decode('utf-8', errors='ignore')
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    remove_tokens_pat = '|'.join(re.escape(token) for token in special_tokens)
    
    word_counts = Counter()
    text_chunks = re.split(remove_tokens_pat, chunk_str)
    for chunk in text_chunks:
        if chunk:
            words = re.findall(PAT, chunk)
            word_counts.update(words)
    
    # Convert words to byte-tuples and count initial pairs
    pretokens = {}
    pair_counts = collections.defaultdict(int)
    
    for word, freq in word_counts.items():
        # Correctly convert a word like "low" into (b'l', b'o', b'w')
        byte_tuple = tuple(bytes([b]) for b in word.encode('utf-8'))
        pretokens[byte_tuple] = freq
        
        # Count initial pairs in this word
        for i in range(len(byte_tuple) - 1):
            pair_counts[(byte_tuple[i], byte_tuple[i+1])] += freq
        
    return pretokens, dict(pair_counts)
    

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    assert vocab_size > 256 + len(special_tokens), "TOO SMALL VOCAB_SIZE!"
    
    # Step 1: Pre-tokenization and Initial Counts (Optimized)
    num_workers = multiprocessing.cpu_count()
    with open(input_path, 'rb') as file:
        chunks = find_chunk_boundaries(file, num_workers, [token.encode('utf-8') for token in special_tokens])
        
    print("----------start multiprocess----------")
    args = [(input_path, start_pos, end_pos, special_tokens) for start_pos, end_pos in zip(chunks[:-1], chunks[1:])]
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Each worker returns a tuple of (word_counts, pair_counts)
        results = pool.starmap(pretokenize_worker, args)
    print("----------end multiprocess----------")
    
    # Aggregate results from all workers
    word_counts = collections.defaultdict(int)
    pair_counts = collections.defaultdict(int)
    for wc, pc in results:
        for word, freq in wc.items():
            word_counts[word] += freq
        for pair, freq in pc.items():
            pair_counts[pair] += freq
    
    # Step 2: Initialize vocabulary
    merges = []
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode('utf-8')
    
    # Step 3: Iterative Merging (The "Fast BPE" Algorithm)
    num_merges = vocab_size - len(vocab)
    for merge_step in range(num_merges):
        if not pair_counts:
            break  # No more pairs to merge
        
        # Find the best pair to merge
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        best_count = pair_counts[best_pair]
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        
        # Update word_counts and pair_counts incrementally
        new_word_counts = {}
        # Remove the count for the pair we're merging
        del pair_counts[best_pair]
        
        for word_tuple, freq in word_counts.items():
            # Check if this word contains the best_pair
            if len(word_tuple) < 2:
                new_word_counts[word_tuple] = freq
                continue
                
            # Merge the best_pair in this word
            i = 0
            new_word = []
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == best_pair[0] and word_tuple[i+1] == best_pair[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            
            new_word_tuple = tuple(new_word)
            new_word_counts[new_word_tuple] = freq
            
            # If the word changed, update pair counts
            if new_word_tuple != word_tuple:
                # Remove old pairs
                for j in range(len(word_tuple) - 1):
                    old_pair = (word_tuple[j], word_tuple[j+1])
                    pair_counts[old_pair] -= freq
                    if pair_counts[old_pair] <= 0:
                        del pair_counts[old_pair]
                
                # Add new pairs
                for j in range(len(new_word_tuple) - 1):
                    new_pair = (new_word_tuple[j], new_word_tuple[j+1])
                    if new_pair not in pair_counts:
                        pair_counts[new_pair] = 0
                    pair_counts[new_pair] += freq
        
        word_counts = new_word_counts
        
    return vocab, merges