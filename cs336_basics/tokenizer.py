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
                    # print(f"special_tokens added to vocab is {bst}")
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
        # print("")
        
        if self.special_tokens:
            # FOR OVERLAPPING SPECIAL TOKENS!!
            sorted_special_tokens = (re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
            remove_tokens_pat = '|'.join(sorted_special_tokens)
            all_special_tokens = re.findall(remove_tokens_pat, text)
            all_special_tokens_length = len(all_special_tokens)
            index_special_tokens = 0
            # print(f"all special tokens: {all_special_tokens}")
            # print(f"len_of_special_tokens_in_this_text_is: {len(all_special_tokens)}")
            
            # print(f"splited chunks: {re.split(remove_tokens_pat, text)}")
            for chunk in re.split(remove_tokens_pat, text):
                if chunk:
                    words.extend(re.findall(PAT, chunk))
                # print(f"index_is: {index_special_tokens}")
                if all_special_tokens_length:
                    if index_special_tokens == all_special_tokens_length:
                        break
                    words.append(all_special_tokens[index_special_tokens])
                    # print(f"current words is {words}")
                    index_special_tokens += 1             
        else:
            words = re.findall(PAT, text)
        
        # for bebug
        # print(words)
        
        text_token = []
        text_token_encode = []
        merges_dict = {}
        
        for index, merge in enumerate(self.merges):
            merges_dict[merge] = index

        for word in words:
            if self.special_tokens != None and word in self.special_tokens:
                text_token.append(word.encode('utf-8'))
                continue
            
            # word class tokenize
            word_token = [bytes([c]) for c in word.encode('utf-8')]
            word_length = len(word_token)
            # print(f"word_length: {word_length}")
            new_word_token = []
            while word_length > 1:
                i = 0             
                pair = None
                pairs_dict = {}
                pair_index = {}
                
                while i < word_length - 1:
                    pair = (word_token[i], word_token[i+1])
                    if pair in self.merges:
                        pairs_dict[pair] = merges_dict[pair]
                        pair_index[pair] = (i, i + 1)
                    i += 1  
                        
                if pairs_dict == {}:
                    # no pair, break
                    break
                # print("-----------PRE_WORD TOKEN----------")
                # print(word_token)
                # print("-----------PAIR DICT----------")
                # print(pairs_dict)
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  
                best_pair = min(pairs_dict, key=lambda p: (pairs_dict[p], p))
                paired_token = best_pair[0] + best_pair[1] 
                j = 0
                while j < word_length:
                    if j == pair_index[best_pair][0]:
                        new_word_token.append(paired_token)
                        j += 2
                    else:
                        new_word_token.append(word_token[j])
                        j += 1
                
                word_token = new_word_token
                # print(f"word_token: {word_token}")
                new_word_token = []
                word_length = len(word_token)
                # print("-----------AFTER_WORD TOKEN----------")
                # print(word_token)
                # print("")

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
        
        for str_i in iterable:
            yield from self.encode(str_i)
                         
    
    def decode(
        self,
        ids: list[int]
    ) -> str:
        
        string_bytes = b''
        for id in ids:
            string_bytes += self.vocab[id]
        # print(string_bytes)
        decode_string = string_bytes.decode('utf-8', errors='replace')
        # print("Final decode is ", decode_string)
        return decode_string
    
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