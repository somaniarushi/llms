from __future__ import annotations

from pathlib import Path

import pytest

from data.tokenizer.json_tokenizer import BaseJSONTokenizer

ALL_CHARACTER_VOCAB_FILE = Path('data/tokenizer/all_chars.json')


class AllCharsTokenizer(BaseJSONTokenizer):
    def __init__(self):
        super().__init__(ALL_CHARACTER_VOCAB_FILE)


class TestJSONTokenizer:
    def test_stoi(self):
        tokenizer = AllCharsTokenizer()
        assert tokenizer.stoi('a') == 39
        assert tokenizer.stoi('b') == 40
        assert tokenizer.stoi('c') == 41
        assert tokenizer.stoi(' ') == 1
        assert tokenizer.stoi('e') == 43
        assert tokenizer.stoi('!') == 2
        # Value error if char not in vocab
        with pytest.raises(ValueError):
            print(tokenizer.stoi('*'))

    def test_itos(self):
        tokenizer = AllCharsTokenizer()
        assert tokenizer.itos(4) == '&'
        assert tokenizer.itos(40) == 'b'
        assert tokenizer.itos(41) == 'c'
        assert tokenizer.itos(1) == ' '
        assert tokenizer.itos(43) == 'e'
        assert tokenizer.itos(2) == '!'
        # Value error if index not in vocab
        with pytest.raises(ValueError):
            print(tokenizer.itos(100))

    def test_encode(self):
        tokenizer = AllCharsTokenizer()
        assert tokenizer.encode('abc') == [39, 40, 41]
        assert tokenizer.encode('hii there') == [46, 47, 47, 1, 58, 46, 43, 56, 43]
        assert len(tokenizer.encode('hi there')) == len('hi there')
        # Value error if char not in vocab
        with pytest.raises(ValueError):
            print(tokenizer.encode('hi there*'))

    def test_decode(self):
        tokenizer = AllCharsTokenizer()
        assert tokenizer.decode([39, 40, 41]) == 'abc'
        assert tokenizer.decode([46, 47, 1, 58, 46, 43, 56, 43]) == 'hi there'

    def test_encode_decode(self):
        tokenizer = AllCharsTokenizer()
        assert tokenizer.decode(tokenizer.encode('abc')) == 'abc'
        assert tokenizer.decode(tokenizer.encode('hi there')) == 'hi there'
