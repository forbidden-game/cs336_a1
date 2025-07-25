from byte_pair_encoding import train_bpe
import json
import os


def save_vocab(
    path: str,
    vocab: dict[int, bytes]
):
    
    # inverted_vocab = {}
    # for vocab_id, vocab_word in vocab.items():
    #     if vocab_id > 255:
    #         inverted_vocab[vocab_word.decode('utf-8', 'ignore')] = vocab_id
    
    inverted_vocab = {v.decode('utf-8', 'ignore'): k for k, v in vocab.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(inverted_vocab, f, indent=2, ensure_ascii=False)
        
    print(f"vocab have saved to {path}!")


def save_merges(
    path: str,
    merges: list[tuple[bytes, bytes]]
):
    
    with open(path, 'w', encoding='utf-8') as f:
        for p1,p2 in merges:
            s1 = p1.decode('utf-8', errors='backslashreplace')
            s2 = p2.decode('utf-8', errors='backslashreplace')
            f.write(f"({s1}, {s2}) -> ({s1}{s2})\n")
    print(f"Merges have saved to {path}")


def train_bpe_tinystories():
    
    vocab, merges = train_bpe("/home/lucain/workspace/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    
    output_dir = "./train_bpe_tinystories/"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting BPE training on TinyStories...")
    vocab_save_path = os.path.join(output_dir, "vocab.json")
    merges_save_path = os.path.join(output_dir, "merges.txt")
    print("Training complete.")
    
    save_vocab(vocab_save_path, vocab)
    save_merges(merges_save_path, merges)
    
    
if __name__ == "__main__":
    train_bpe_tinystories()