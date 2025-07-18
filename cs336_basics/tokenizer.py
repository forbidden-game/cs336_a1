import os
import multiprocessing
import json


class tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens=list[str] | None = None
        ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        
        
    def encode(
        self,
        text: str
    ) -> list[int]:
        
        
    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int]:
        
        
    def decode(
        self,
        ids: list[int]
    ) -> str:
        
        return ""
        

def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> tokenizer:


        with open(vocab_filepath, 'r') as f:
            vocab = {index: token.encode('utf-8') for index, token in json.load(f).items()}
        merges = []
        with open(merges_filepath, 'r') as f:
            for p1, p2 in json.load(f).items():
                merges.append((p1, p2))
        
        return tokenizer(vocab, merges, special_tokens)
                
            