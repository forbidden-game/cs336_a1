import regex as re
import os
import multiprocessing
import collections
from collections import Counter
from typing import BinaryIO, List, Dict, Tuple

# This is a constant, fine as a global
VOCAB_BYTES = {i: bytes([i]) for i in range(256)}

def find_chunk_boundaries(
    file_path: str, 
    desired_num_chunks: int, 
    split_special_token="<|endoftext|>".encode("utf-8")
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return [0]

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks)]
    chunk_boundaries.append(file_size)

    # Adjust boundaries to not split in the middle of a special token
    with open(file_path, 'rb') as f:
        for i in range(1, len(chunk_boundaries) - 1):
            # Seek to a position slightly before the guessed boundary
            # to handle cases where the token crosses the boundary.
            seek_pos = max(0, chunk_boundaries[i] - len(split_special_token))
            f.seek(seek_pos)
            
            # Read a buffer that is large enough to likely contain the token
            buffer = f.read(chunk_size // 10) 
            if not buffer:
                continue

            # Find the first occurrence of the special token in the buffer
            split_pos = buffer.find(split_special_token)
            
            if split_pos != -1:
                # Adjust boundary to the start of the found token
                chunk_boundaries[i] = seek_pos + split_pos
            # If not found, we leave the boundary as is. It's an approximation.

    # Remove duplicates and ensure sorted order
    return sorted(list(set(chunk_boundaries)))

def get_word_vocab_from_chunk(
    corpus_path: str, start_offset: int, end_offset: int, special_tokens: list[str]
) -> Counter:
    """Worker to read a chunk and return word counts."""
    with open(corpus_path, 'rb') as f:
        f.seek(start_offset)
        chunk_bytes = f.read(end_offset - start_offset)
    
    text = chunk_bytes.decode('utf-8', errors='ignore')

    # Remove special tokens from the text before tokenization
    for token in special_tokens:
        text = text.replace(token, "")

    # GPT-2 pre-tokenization pattern
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    pre_tokens = re.findall(PAT, text)
    return Counter(pre_tokens)

def get_stats(vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
    """Calculates frequency of adjacent pairs in the vocabulary."""
    pairs = collections.defaultdict(int)
    for word_tuple, freq in vocab.items():
        for i in range(len(word_tuple) - 1):
            pairs[word_tuple[i], word_tuple[i+1]] += freq
    return pairs

def merge_pair(
    pair: Tuple[str, str], 
    vocab_in: Dict[Tuple[str, ...], int]
) -> Dict[Tuple[str, ...], int]:
    """Merges a pair of tokens in the vocabulary."""
    vocab_out = {}
    p1, p2 = pair
    merged_token = p1 + p2
    
    for word_tuple, freq in vocab_in.items():
        i = 0
        new_word_tuple = []
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and word_tuple[i] == p1 and word_tuple[i+1] == p2:
                new_word_tuple.append(merged_token)
                i += 2
            else:
                new_word_tuple.append(word_tuple[i])
                i += 1
        vocab_out[tuple(new_word_tuple)] = freq
    return vocab_out


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    assert vocab_size > 256 + len(special_tokens), "Vocab size is too small"
    
    # 1. Parallel Pre-tokenization to get word counts
    num_workers = multiprocessing.cpu_count()
    chunk_offsets = find_chunk_boundaries(input_path, num_workers, split_special_token=b"<|endoftext|>")
    worker_args = [(input_path, start, end, special_tokens) for start, end in zip(chunk_offsets[:-1], chunk_offsets[1:])]
    
    print(f"--------Starting Parallel Pre-tokenization with {num_workers} workers--------")
    with multiprocessing.Pool(processes=num_workers) as pool:
        list_of_counters = pool.starmap(get_word_vocab_from_chunk, worker_args)
    print("--------Finished Parallel Pre-tokenization--------")
    
    word_counts = Counter()
    for counter in list_of_counters:
        word_counts.update(counter)

    # 2. Initialize BPE vocabulary from character splits
    # e.g., "hello": 5 -> ('h', 'e', 'l', 'l', 'o'): 5
    bpe_vocab = {tuple(word): freq for word, freq in word_counts.items()}
    
    # 3. Add special tokens and build initial vocab
    merges = []
    final_vocab = dict(VOCAB_BYTES) # Start with base 256 bytes
    next_token_id = 256
    
    for token in special_tokens:
        final_vocab[next_token_id] = token.encode('utf-8')
        next_token_id += 1

    # 4. Iteratively merge the most frequent pairs
    num_merges = vocab_size - len(final_vocab)
    print(f"Starting {num_merges} merges...")

    for i in range(num_merges):
        stats = get_stats(bpe_vocab)
        if not stats:
            print("No more pairs to merge. Stopping early.")
            break
            
        # Find the best pair to merge
        best_pair = max(stats, key=stats.get)
        
        # Perform the merge in our BPE vocabulary
        bpe_vocab = merge_pair(best_pair, bpe_vocab)
        
        # Record the merge
        p1_bytes, p2_bytes = best_pair[0].encode('utf-8'), best_pair[1].encode('utf-8')
        merges.append((p1_bytes, p2_bytes))
        
        # Add the new merged token to our final vocabulary
        final_vocab[next_token_id] = p1_bytes + p2_bytes
        next_token_id += 1

        if (i + 1) % 100 == 0:
            print(f"Merge {i+1}/{num_merges}: Merged {best_pair} into '{best_pair[0]}{best_pair[1]}'")
        
    return final_vocab, merges

if __name__ == '__main__':
    # Example usage:
    # Create a dummy corpus file for demonstration
    dummy_corpus_path = "dummy_corpus.txt"
    with open(dummy_corpus_path, "w", encoding="utf-8") as f:
        f.write("hello world<|endoftext|>\n")
        f.write("hello there, this is a low-resource test.\n")
        f.write("The lowest of the low.\n")
        
    special_tokens = ["<|endoftext|>", "<|pad|>"]
    vocab_size = 10000 # 256 bytes + 2 special + 42 merges
    
    final_vocab, merges = train_bpe("/home/lucain/workspace/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", vocab_size, special_tokens)
    
    print("\n--- Training Complete ---")
    print(f"Final Vocab Size: {len(final_vocab)}")
    print("First 10 merges:")
    for p1, p2 in merges[:10]:
        print(f"  {p1.decode('utf-8', 'ignore')} + {p2.decode('utf-8', 'ignore')} -> {(p1+p2).decode('utf-8', 'ignore')}")

    os.remove(dummy_corpus_path)