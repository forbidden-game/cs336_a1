import regex as re
import os
import multiprocessing
from collections import Counter
from typing import BinaryIO

vocab = {}
for i in range(256):
    vocab[i] = bytes([i])
    

def find_chunk_boundaries(
    file_str: str, 
    desired_num_chunks: int, 
    split_special_token="<|endoftext|>".encode("utf-8")
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    with open(file_str, 'rb') as file:
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
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenization_worker(
    corpus_path: str, start_offset: int, end_offset: int
) -> Counter:
    
    with open(corpus_path, 'rb') as f:
        f.seek(start_offset)
        chunk = f.read(end_offset-start_offset)
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    splited_chunk = re.findall(PAT, chunk.decode('utf-8', errors='ignore'))
    
    return Counter(splited_chunk)

def parallelized_pretokenize(
    num_workers: int,
    corpus_path: str
) -> Counter:
    
    chunk_offsets = find_chunk_boundaries(corpus_path, num_workers)
    worker_args = [(corpus_path, start, end) for start, end in zip(chunk_offsets[:-1], chunk_offsets[1:])]
    
    print("--------Start Parallel Processing--------")
    with multiprocessing.Pool(processes=num_workers) as pool:
        list_of_counters = pool.starmap(pretokenization_worker, worker_args)
    print("--------Parallel Processing Finished--------")
    
    total_counts = Counter()
    for counter in list_of_counters:
        total_counts.update(counter)
        
    return total_counts

corpus_path = "/home/lucain/workspace/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"

tc = parallelized_pretokenize(6, corpus_path)

print(tc)