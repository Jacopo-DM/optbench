"""The base of every record in optbench."""

from pydantic import BaseModel, ConfigDict


class Model(BaseModel):
    """Base for every record here: immutable, no unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")
