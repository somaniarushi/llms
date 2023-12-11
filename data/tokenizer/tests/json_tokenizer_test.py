import pytest
from pathlib import Path
from data.tokenizer.json_tokenizer import BaseJSONTokenizer

ALL_CHARACTER_VOCAB_FILE = Path("data/tokenizer/all_chars.json")


class AllCharsTokenizer(BaseJSONTokenizer):
    def __init__(self):
        super().__init__(ALL_CHARACTER_VOCAB_FILE)


class TestJSONTokenizer:
    def test_stoi(self):
        tokenizer = AllCharsTokenizer()
        assert tokenizer.stoi("a") == 4
        assert tokenizer.stoi("b") == 22
        assert tokenizer.stoi("c") == 19
        assert tokenizer.stoi(" ") == 0
        assert tokenizer.stoi("e") == 1
        assert tokenizer.stoi("!") == 46
        # Value error if char not in vocab
        with pytest.raises(ValueError):
            print(tokenizer.stoi("*"))

    def test_itos(self):
        tokenizer = AllCharsTokenizer()
        assert tokenizer.itos(4) == "a"
        assert tokenizer.itos(22) == "b"
        assert tokenizer.itos(19) == "c"
        assert tokenizer.itos(0) == " "
        assert tokenizer.itos(1) == "e"
        assert tokenizer.itos(46) == "!"
        # Value error if index not in vocab
        with pytest.raises(ValueError):
            print(tokenizer.itos(100))

    def test_encode(self):
        tokenizer = AllCharsTokenizer()
        assert tokenizer.encode("abc") == [4, 22, 19]
        assert tokenizer.encode("hi there") == [5, 9, 0, 2, 5, 1, 7, 1]
        assert len(tokenizer.encode("hi there")) == len("hi there")
        # Value error if char not in vocab
        with pytest.raises(ValueError):
            print(tokenizer.encode("hi there*"))

    def test_decode(self):
        tokenizer = AllCharsTokenizer()
        assert tokenizer.decode([4, 22, 19]) == "abc"
        assert tokenizer.decode([5, 9, 0, 2, 5, 1, 7, 1]) == "hi there"

    def test_encode_decode(self):
        tokenizer = AllCharsTokenizer()
        assert tokenizer.decode(tokenizer.encode("abc")) == "abc"
        assert tokenizer.decode(tokenizer.encode("hi there")) == "hi there"
