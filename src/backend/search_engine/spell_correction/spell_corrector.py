from pathlib import Path
from dataclasses import dataclass
from neuspell import SclstmChecker

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = _THIS_DIR.parent
CHECKPOINT_DIR = PROJECT_DIR / "models" / "neuspell-scrnn-probwordnoise"
MODEL_PATH = CHECKPOINT_DIR.resolve()


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
