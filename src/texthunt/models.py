"""Core domain types shared across ingest, profiling, and scoring."""

from pydantic import BaseModel


class Message(BaseModel, frozen=True):
    """A single chat message after ingest, before any text normalisation."""

    author_id: str
    channel: str
    timestamp: float
    text: str


class Block(BaseModel, frozen=True):
    """Several consecutive messages from one author, joined into one unit of analysis.

    A single short message rarely carries enough signal to identify an author, so blocks are the
    granularity at which profiles are built and queries are scored.
    """

    author_id: str
    channel: str
    text: str
    message_count: int

    @property
    def char_count(self) -> int:
        return len(self.text)
