"""Dataset contracts shared by tabular and text projects."""

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    target: str
    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    text: str | None = None
    source: str = "unspecified"
    kind: str = "tabular"

    @property
    def features(self) -> tuple[str, ...]:
        return (self.text,) if self.text else self.numeric + self.categorical

    def __post_init__(self):
        for field in ("numeric", "categorical"):
            value = getattr(self, field)
            if not isinstance(value, (tuple, list)):
                raise ValueError("Feature collections must be tuples or lists")
            object.__setattr__(self, field, tuple(value))
        columns = self.features + (self.target,)
        if not isinstance(self.name, str) or not self.name.strip() or not self.features:
            raise ValueError("Dataset needs a name and features")
        if any(not isinstance(column, str) or not column.strip() for column in columns):
            raise ValueError("Column names must be nonempty strings")
        if len(set(columns)) != len(columns):
            raise ValueError("Dataset needs distinct feature/target columns")
        if self.text and (self.numeric or self.categorical):
            raise ValueError("Text and tabular features cannot be mixed")
        if self.kind not in ("tabular", "text") or (self.kind == "text" and not self.text):
            raise ValueError("Dataset kind must agree with its features")
        if self.text:
            object.__setattr__(self, "kind", "text")


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    spec: DatasetSpec

    @property
    def X(self) -> pd.DataFrame:
        return self.frame.loc[:, list(self.spec.features)].copy()

    @property
    def y(self) -> pd.Series:
        return self.frame[self.spec.target].copy()
