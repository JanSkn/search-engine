class ParenthesesWarning(UserWarning):
    def __init__(
        self,
        message: str | None = None,
        open_count: int | None = None,
        close_count: int | None = None,
    ):
        self.message = message
        self.open_count = open_count
        self.close_count = close_count
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "type": "ParenthesesWarning",
            "message": self.message,
            "open_count": self.open_count,
            "close_count": self.close_count,
        }


class InvalidQueryError(Exception):
    pass


class InvalidOperatorError(InvalidQueryError):
    pass
