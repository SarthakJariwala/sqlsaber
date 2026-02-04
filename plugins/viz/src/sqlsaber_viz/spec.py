"""Pydantic models for visualization specs."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class FieldEncoding(BaseModel):
    field: str
    type: Literal["category", "number", "time"] = "number"


class ChartOptions(BaseModel):
    width: int | None = Field(default=None, ge=20, le=200)
    height: int | None = Field(default=None, ge=10, le=100)
    x_label: str | None = None
    y_label: str | None = None
    color: str | None = None
    marker: str | None = None


class BarEncoding(BaseModel):
    x: FieldEncoding
    y: FieldEncoding
    series: FieldEncoding | None = None


class BarChart(BaseModel):
    type: Literal["bar"]
    encoding: BarEncoding
    orientation: Literal["vertical", "horizontal"] = "vertical"
    mode: Literal["grouped", "stacked"] = "grouped"
    options: ChartOptions = Field(default_factory=ChartOptions)


class LineEncoding(BaseModel):
    x: FieldEncoding
    y: FieldEncoding
    series: FieldEncoding | None = None


class LineChart(BaseModel):
    type: Literal["line"]
    encoding: LineEncoding
    options: ChartOptions = Field(default_factory=ChartOptions)


class ScatterEncoding(BaseModel):
    x: FieldEncoding
    y: FieldEncoding
    series: FieldEncoding | None = None


class ScatterChart(BaseModel):
    type: Literal["scatter"]
    encoding: ScatterEncoding
    options: ChartOptions = Field(default_factory=ChartOptions)


class BoxplotConfig(BaseModel):
    label_field: str
    value_field: str


class BoxplotChart(BaseModel):
    type: Literal["boxplot"]
    boxplot: BoxplotConfig
    options: ChartOptions = Field(default_factory=ChartOptions)


class HistogramConfig(BaseModel):
    field: str
    bins: int = Field(default=20, ge=2, le=100)


class HistogramChart(BaseModel):
    type: Literal["histogram"]
    histogram: HistogramConfig
    options: ChartOptions = Field(default_factory=ChartOptions)


ChartSpec = Annotated[
    BarChart | LineChart | ScatterChart | BoxplotChart | HistogramChart,
    Field(discriminator="type"),
]


class SortItem(BaseModel):
    field: str
    dir: Literal["asc", "desc"] = "asc"


class SortTransform(BaseModel):
    sort: list[SortItem]


class LimitTransform(BaseModel):
    limit: int = Field(ge=1)


class FilterConfig(BaseModel):
    field: str
    op: Literal["==", "!=", ">", "<", ">=", "<="]
    value: str | int | float | bool | None


class FilterTransform(BaseModel):
    filter: FilterConfig


Transform = SortTransform | LimitTransform | FilterTransform


class DataSource(BaseModel):
    file: str = Field(pattern=r"^result_[A-Za-z0-9._-]+\.json$")


class DataConfig(BaseModel):
    source: DataSource


class VizSpec(BaseModel):
    version: Literal["1"] = "1"
    title: str | None = None
    description: str | None = None
    data: DataConfig
    chart: ChartSpec
    transform: list[Transform] = Field(default_factory=list)
