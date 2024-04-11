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
        columns = self.features + (self.target,)
        if not self.name or not self.features or len(set(columns)) != len(columns):
            raise ValueError("Dataset needs a name and distinct feature/target columns")
        if self.text and (self.numeric or self.categorical):
            raise ValueError("Text and tabular features cannot be mixed")


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
