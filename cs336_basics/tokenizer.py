import os
import multiprocessing
import json
import regex as re


class tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        
        
    def encode(
        self,
        text: str
    ) -> list[int]:
        
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        words = re.findall(PAT, text)
        
        text_token = []
        text_token_encode = []
        
        for word in words:
            word_bytes = [bytes([c]) for c in word.encode('utf-8')]
            
            word_token = []
            word_length = len(word_bytes)
            if word_length == 1:
                word_token = word_bytes
            else:
                i = 0
                while i < word_length - 1:
                    pair = (word_bytes[i], word_bytes[i+1])
                    if pair in self.merges:                    
                        # replace with new token
                        new_token = pair[0] + pair[1]
                        word_bytes[i+1] = new_token
                        word_token.append(new_token)
                    else:
                        word_token.append(word_bytes[i])
                    
                    i += 1
            text_token.extend(word_token)
        
        for wt in text_token:
            for t in wt:
                for integer, vocab in self.vocab.items():
                    if t == vocab:
                       text_token_encode.append(integer)
        
        return text_token_encode          
            
        
        
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

        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab = {index: token.encode('utf-8') for index, token in json.load(f).items()}
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for p1, p2 in json.load(f):
                merges.append((p1.encode('utf-8'), p2.encode('utf-8')))
        
        return tokenizer(vocab, merges, special_tokens)
                
            