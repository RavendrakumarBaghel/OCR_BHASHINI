import json
import unicodedata


VIRAMA = "\u094d"
JOINERS = {"\u200c", "\u200d"}

class CTCTextEncoder:
    def __init__(self, vocab):
        self.blank_id = 0
        self.id_to_token = {0: "<blank>", **{i + 1: ch for i, ch in enumerate(vocab)}}
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}

    def __len__(self):
        return len(self.id_to_token)

    @staticmethod
    def normalize(text: str) -> str:
        return unicodedata.normalize("NFC", text.strip())

    @classmethod
    def split_text(cls, text: str):
        text = cls.normalize(text)
        tokens = []
        current = ""
        join_next = False

        for ch in text:
            category = unicodedata.category(ch)
            attaches_to_previous = category.startswith("M") or ch in JOINERS

            if not current:
                current = ch
            elif join_next or attaches_to_previous:
                current += ch
            else:
                tokens.append(current)
                current = ch

            join_next = ch == VIRAMA or (join_next and ch in JOINERS)

        if current:
            tokens.append(current)
        return tokens

    def encode(self, text: str):
        tokens = self.split_text(text)
        try:
            return [self.token_to_id[token] for token in tokens]
        except KeyError as exc:
            raise KeyError(
                f"Token {exc.args[0]!r} is missing from the vocabulary. "
                "Run prepare_data.py again after tokenizer changes."
            ) from exc


    def decode(self, ids):
        result = []
        prev = None
        for idx in ids:
            if idx != self.blank_id and idx != prev:
                result.append(self.id_to_token[idx])
            prev = idx
        return "".join(result)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)
