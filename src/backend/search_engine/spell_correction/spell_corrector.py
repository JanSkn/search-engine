from dataclasses import dataclass
from neuspell import SclstmChecker


MODEL_PATH = "../models/neuspell-scrnn-probwordnoise"

@dataclass
class SpellCorrector:
    checker: SclstmChecker

    @classmethod
    def load(cls) -> "SpellCorrector":
        print("load neuspell SCLSTM checker…")
        checker = SclstmChecker()
        checker.from_pretrained(MODEL_PATH)
        print("model loaded.\n")
        return cls(checker=checker)

    def correct(self, text: str) -> str:
        return self.checker.correct(text)