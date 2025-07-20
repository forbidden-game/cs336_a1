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
            for chunk in re.split(remove_tokens_pat, text):
                if chunk:
                    words.extend(re.findall(PAT, chunk))
        else:
            words = re.findall(PAT, text)
        
        # for bebug
        print(words)
        
        text_token = []
        text_token_encode = []
        
        for word in words:
            word_token = [bytes([c]) for c in word.encode('utf-8')]
            word_length = len(word_token)
            new_word_token = []
            while word_length > 1:
                i = 0
                while i < word_length - 1:
                    p1, p2 = word_token[i], word_token[i+1]
                    if (p1, p2) in self.merges:
                        print((p1, p2))
                        new_word_token.append(p1+p2)
                        i += 2
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
                new_word_token = []
                word_length = len(word_token)
                print(f"word_token: {word_token}")

            text_token.extend(word_token)
                    

        print(f"text_token: {text_token}")
        print(f"len_text_token: {len(text_token)}")
        
        # create tokens_to_ids for fast encoding
        tokens_to_ids = {token: id for id, token in self.vocab.items()}
        for token in text_token:
            text_token_encode.append(tokens_to_ids[token])
        
        return text_token_encode
    
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
    

if __name__ == "__main__":
    
    
    test_str = "hello! I am.      panxiezhao 980402..... "
    st = ["he", "    ", "98", "123"]
    vocab_filepath = "/home/lucain/workspace/assignment1-basics/train_bpe_expts_owt/hex_vocab.json"
    merges_filepath = "/home/lucain/workspace/assignment1-basics/train_bpe_expts_owt/hex_merges.json"
    
    test_tokenizer = from_files(tokenizer, vocab_filepath, merges_filepath, st)
    
    result = test_tokenizer.encode(test_str)
    print(result)
    print(len(result))