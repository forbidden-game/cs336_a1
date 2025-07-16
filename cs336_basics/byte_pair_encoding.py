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
) -> dict[tuple[bytes, ...], int]:
    
    with open(corpus_path, 'rb') as file:
        file.seek(start_pos)
        chunk = file.read(end_pos-start_pos)
        
    # It's safer to decode with an error handler
    chunk_str = chunk.decode('utf-8', errors='ignore')

    remove_tokens_pat = ''.join(re.escape(token) for token in special_tokens)
    removed_tokens = "".join(re.split(remove_tokens_pat, chunk_str))
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    words = re.findall(PAT, removed_tokens)
    
    word_counts = Counter(words)
    
    # return {tuple(token.encode('utf-8')): freq for token, freq in word_counts.items()}
    pretokens = {}
    for word, freq in word_counts.items():
        # Correctly convert a word like "low" into (b'l', b'o', b'w')
        # This is the crucial change.
        byte_tuple = tuple(bytes([b]) for b in word.encode('utf-8'))
        pretokens[byte_tuple] = freq
        
    return pretokens
    
def pair(
    tokens: dict[tuple[bytes, ...], int]
) -> dict[tuple[bytes, ...], int]:
    
    paired_tokens = collections.defaultdict(int)
    
    for token, freq in tokens.items():
        for i in range(len(token)-1):
            paired_tokens[(token[i], token[i+1])] += freq
    
    return paired_tokens

def merge(
    tokens: dict[tuple[bytes, ...], int],
    pair_to_merge: tuple[bytes, bytes]
) -> dict[tuple[bytes, ...], int]:
    
    merged_out = {}
    p1, p2 = pair_to_merge
    merged_pair = p1 + p2
    
    for token, freq in tokens.items():
        i = 0
        merged_token = []
        while i < len(token):
            if i < len(token) - 1 and token[i] == p1 and token[i+1] == p2:
                merged_token.append(merged_pair)
                i += 2
            else:
                merged_token.append(token[i])
                i += 1
        merged_out[tuple(merged_token)] = freq
    
    return merged_out   

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    assert vocab_size > 256 + len(special_tokens), "TOO SMALL VOCAB_SIZE!"
    
    num_workers = multiprocessing.cpu_count()
    with open(input_path, 'rb') as file:
        chunks = find_chunk_boundaries(file, num_workers, [token.encode('utf-8') for token in special_tokens])
        
        print("----------start multiprocess----------")
        args = [(input_path, start_pos, end_pos, special_tokens) for start_pos, end_pos in zip(chunks[:-1], chunks[1:])]
        with multiprocessing.Pool(processes=num_workers) as pool:
            multi_pretokens = pool.starmap(pretokenize_worker, args)
        print("----------end multiprocess----------")
        pretokens = Counter()
        for pretoken in multi_pretokens:
            pretokens.update(pretoken)
    
    pretoken_dict = dict(pretokens)
    merges = []
    vocab = {i: bytes([i]) for i in range(256)}
    for i in range(len(special_tokens)):
        vocab[256+i] = special_tokens[i].encode('utf-8')

    base_vocab_size = 256 + len(special_tokens)
    for i in range(base_vocab_size, vocab_size):
        
        pairs = pair(pretoken_dict)
        pair_to_merge = max(pairs, key=lambda p: (pairs[p], p))
        merges.append(pair_to_merge)
        vocab[i] = pair_to_merge[0] + pair_to_merge[1]

        pretoken_dict = merge(pretoken_dict, pair_to_merge)
        
    return vocab, merges

# if __name__ == '__main__':
    # Example usage:
    # Create a dummy corpus file for demonstration
    # dummy_corpus_path = "dummy_corpus.txt"
    # with open(dummy_corpus_path, "w", encoding="utf-8") as f:
    #     f.write("hello world<|endoftext|>\n")
    #     f.write("hello there, this is a low-resource test.\n")
    #     f.write("The lowest of the low.\n")
        
    # special_tokens = ["<|endoftext|>", "<|pad|>"]
    # vocab_size = 500 # 256 bytes + 2 special + 42 merges
    
    # final_vocab, merges = train_bpe("/home/lucain/workspace/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", vocab_size, special_tokens)
    
    # print("\n--- Training Complete ---")
    # print(f"Final Vocab Size: {len(final_vocab)}")
    # print("First 10 merges:")
    # for p1, p2 in merges[:10]:
    # print(f"  {p1.decode('utf-8', 'ignore')} + {p2.decode('utf-8', 'ignore')} -> {(p1+p2).decode('utf-8', 'ignore')}")