import json
import regex as re
from _collections_abc import Iterable, Iterator

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
        
        if special_tokens:
            bytes_special_tokens = [st.encode('utf-8') for st in special_tokens]
            next_id = next(reversed(vocab.keys())) + 1
            
            for bst in bytes_special_tokens:
                if bst not in self.vocab.values():
                    self.vocab[next_id] = bst
                    next_id += 1
                    # for debug
                    # print(f"current id is {next_id - 1}, bst is {bst}")
                    
        
    def encode(
        self,
        text: str
    ) -> list[int]:
        
        words = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        if self.special_tokens:
            remove_tokens_pat = '|'.join(re.escape(token) for token in self.special_tokens)
            all_special_tokens = re.findall(remove_tokens_pat, text)
            # print(f"all special tokens: {all_special_tokens}")
            # print(f"len_of_special_tokens_in_this_text_is: {len(all_special_tokens)}")
            index_special_tokens = 0
            # print(f"splited chunks: {re.split(remove_tokens_pat, text)}")
            for chunk in re.split(remove_tokens_pat, text):
                if chunk:
                    words.extend(re.findall(PAT, chunk))
                # print(f"index_is: {index_special_tokens}")
                words.append(all_special_tokens[index_special_tokens])
                # print(f"current words is {words}")
                index_special_tokens += 1
                if index_special_tokens == len(all_special_tokens):
                    break
                    
        else:
            words = re.findall(PAT, text)
        
        # for bebug
        # print(words)
        
        text_token = []
        text_token_encode = []
        
        for word in words:
            if word in self.special_tokens:
                text_token.append(word.encode('utf-8'))
                continue
            word_token = [bytes([c]) for c in word.encode('utf-8')]
            word_length = len(word_token)
            # print(f"word_length: {word_length}")
            new_word_token = []
            while word_length > 1:
                i = 0
                while i < word_length - 1:
                    p1, p2 = word_token[i], word_token[i+1]
                    has_token = p1 + p2
                    # print(f"i: {i}")
                    # print(f"fist two pair is: ({p1}, {p2})")
                    # if (p1, p2) in self.merges:
                    if has_token in self.vocab.values():
                        new_word_token.append(has_token)
                        i += 2
                        # print(f"They got paired!")
                        # print(f"i+2: {i}")
                        if i == word_length - 1:
                            new_word_token.append(word_token[-1])
                    else:
                        new_word_token.append(p1)
                        i += 1
                        if i == word_length - 1:
                            new_word_token.append(p2)
                if word_token == new_word_token:
                    break
                word_token = new_word_token
                # print(f"word_token: {word_token}")
                new_word_token = []
                word_length = len(word_token)
                # print(f"word_token: {word_token}")

            text_token.extend(word_token)
                    

        # print(f"text_token: {text_token}")
        # print(f"len_text_token: {len(text_token)}")
        
        # create tokens_to_ids for fast encoding
        tokens_to_ids = {token: id for id, token in self.vocab.items()}
        for token in text_token:
            text_token_encode.append(tokens_to_ids[token])
        
        return text_token_encode
    
    
    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int]:
        
        iterator = iter(iterable)
        tokens_to_ids = {token: id for id, token in self.vocab.items()}
        for fisrt_str in iterator:
            first_encode = self.encode(fisrt_str)
            first_last_token = self.vocab[first_encode[-1]]
            try:
                second_str = next(iterator)
                second_encode = self.encode(second_str)
                second_last_token = self.vocab[second_encode[-1]]
                if_cross_token = first_last_token + second_last_token
                
                if if_cross_token in tokens_to_ids.keys():
                    for id in first_encode[:-1] + [tokens_to_ids[if_cross_token]] + second_encode[1:]:
                        yield id
            except StopIteration:
                for id in first_encode:
                    yield id         
    
    def decode(
        self,
        ids: list[int]
    ) -> str:
        
        string_bytes = b''
        for id in ids:
            string_bytes += self.vocab[id]
        return string_bytes.decode('utf-8', errors='replace')
    
def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> tokenizer:

        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab = {int(index): bytes.fromhex(token) for index, token in json.load(f).items()}
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for p1, p2 in json.load(f):
                merges.append((bytes.fromhex(p1), bytes.fromhex(p2)))
        
        return tokenizer(vocab, merges, special_tokens)