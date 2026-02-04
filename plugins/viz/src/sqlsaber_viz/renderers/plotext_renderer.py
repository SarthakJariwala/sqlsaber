"""Plotext renderer for terminal charts."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, time
from typing import Iterable

from ..spec import (
    BarChart,
    BoxplotChart,
    HistogramChart,
    LineChart,
    ScatterChart,
    VizSpec,
)


class PlotextRenderer:
    """Render VizSpec to terminal using plotext."""

    _series_colors = [
        "cyan+",
        "yellow+",
        "red+",
        "green+",
        "blue+",
        "magenta+",
        "white+",
    ]
    _default_width = 80
    _default_height = 25

    def render(self, spec: VizSpec, rows: list[dict]) -> str:
        """Render spec with data to ASCII chart string.

        Returns:
            ASCII chart string from plt.build(), or error message if rendering fails.
        """
        import plotext as plt

        plt.clf()
        plt.clear_figure()

        chart = spec.chart
        options = chart.options

        width = options.width or self._default_width
        height = options.height or self._default_height
        plt.plot_size(width=width, height=height)

        if spec.title:
            plt.title(spec.title)

        error_msg: str | None = None
        try:
            if isinstance(chart, BarChart):
                error_msg = self._render_bar(chart, rows, plt)
            elif isinstance(chart, LineChart):
                error_msg = self._render_line(chart, rows, plt)
            elif isinstance(chart, ScatterChart):
                error_msg = self._render_scatter(chart, rows, plt)
            elif isinstance(chart, BoxplotChart):
                error_msg = self._render_boxplot(chart, rows, plt)
            elif isinstance(chart, HistogramChart):
                error_msg = self._render_histogram(chart, rows, plt)
            else:
                return f"[Unsupported chart type: {type(chart).__name__}]"
        except Exception as e:
            return f"[Chart rendering error: {e}]"

        if error_msg:
            return error_msg

        if options.x_label:
            plt.xlabel(options.x_label)
        if options.y_label:
            plt.ylabel(options.y_label)

        return plt.build()

    def _render_bar(self, chart: BarChart, rows: list[dict], plt) -> str | None:
        x_field = chart.encoding.x.field
        y_field = chart.encoding.y.field
        series_field = chart.encoding.series.field if chart.encoding.series else None

        orientation = "h" if chart.orientation == "horizontal" else "v"

        if series_field:
            categories, series_names, series_values = self._build_series_matrix(
                rows, x_field, y_field, series_field
            )
            if not categories or not series_names:
                return f"[No data: no valid values for '{x_field}' / '{y_field}']"
            if chart.mode == "stacked":
                plt.stacked_bar(
                    categories,
                    series_values,
                    labels=series_names,
                    orientation=orientation,
                )
            else:
                plt.multiple_bar(
                    categories,
                    series_values,
                    labels=series_names,
                    orientation=orientation,
                )
            return None

        # Aggregate by category (sum) for consistency with series path
        aggregated: dict[str, float] = {}
        for row in rows:
            category = str(row.get(x_field, ""))
            value = self._to_number(row.get(y_field))
            if value is None:
                continue
            aggregated[category] = aggregated.get(category, 0.0) + value

        if not aggregated:
            return f"[No data: no valid numeric values for '{y_field}']"

        categories = list(aggregated.keys())
        values = list(aggregated.values())

        color = self._safe_color(chart.options.color, "blue+")
        plt.bar(categories, values, color=color, orientation=orientation)
        return None

    def _render_line(self, chart: LineChart, rows: list[dict], plt) -> str | None:
        x_field = chart.encoding.x.field
        y_field = chart.encoding.y.field
        series_field = chart.encoding.series.field if chart.encoding.series else None

        marker = self._safe_marker(chart.options.marker, "braille")

        if series_field:
            series_map = self._group_series(rows, series_field)
            any_plotted = False
            for idx, (series_name, series_rows) in enumerate(series_map.items()):
                x, y = self._extract_xy_sorted(series_rows, x_field, y_field)
                if not x or not y:
                    continue
                any_plotted = True
                color = self._series_colors[idx % len(self._series_colors)]
                plt.plot(x, y, color=color, marker=marker, label=series_name)
            if not any_plotted:
                return f"[No data: no valid values for '{x_field}' / '{y_field}']"
            return None

        x, y = self._extract_xy_sorted(rows, x_field, y_field)
        if not x or not y:
            return f"[No data: no valid values for '{x_field}' / '{y_field}']"
        color = self._safe_color(chart.options.color, "cyan+")
        plt.plot(x, y, color=color, marker=marker)
        return None

    def _render_scatter(self, chart: ScatterChart, rows: list[dict], plt) -> str | None:
        x_field = chart.encoding.x.field
        y_field = chart.encoding.y.field
        series_field = chart.encoding.series.field if chart.encoding.series else None

        marker = self._safe_marker(chart.options.marker, "dot")

        if series_field:
            series_map = self._group_series(rows, series_field)
            any_plotted = False
            for idx, (series_name, series_rows) in enumerate(series_map.items()):
                x, y = self._extract_xy(series_rows, x_field, y_field)
                if not x or not y:
                    continue
                any_plotted = True
                color = self._series_colors[idx % len(self._series_colors)]
                plt.scatter(x, y, color=color, marker=marker, label=series_name)
            if not any_plotted:
                return f"[No data: no valid values for '{x_field}' / '{y_field}']"
            return None

        x, y = self._extract_xy(rows, x_field, y_field)
        if not x or not y:
            return f"[No data: no valid values for '{x_field}' / '{y_field}']"
        color = self._safe_color(chart.options.color, "red+")
        plt.scatter(x, y, color=color, marker=marker)
        return None

    def _render_boxplot(self, chart: BoxplotChart, rows: list[dict], plt) -> str | None:
        label_field = chart.boxplot.label_field
        value_field = chart.boxplot.value_field

        groups: dict[str, list[float]] = {}
        for row in rows:
            label = str(row.get(label_field, ""))
            value = self._to_number(row.get(value_field))
            if value is None:
                continue
            groups.setdefault(label, []).append(value)

        if not groups:
            return f"[No data: no valid numeric values for '{value_field}']"

        labels = list(groups.keys())
        data = [groups[label] for label in labels]

        plt.box(labels, data)
        return None

    def _render_histogram(self, chart: HistogramChart, rows: list[dict], plt) -> str | None:
        field = chart.histogram.field
        bins = chart.histogram.bins

        values: list[float] = []
        for row in rows:
            val = self._to_number(row.get(field))
            if val is not None:
                values.append(val)

        if not values:
            return f"[No data: no valid numeric values for '{field}']"

        color = self._safe_color(chart.options.color, "green+")

        plt.hist(values, bins=bins, color=color)
        return None

    def _extract_xy(
        self, rows: Iterable[dict], x_field: str, y_field: str
    ) -> tuple[list[float], list[float]]:
        x: list[float] = []
        y: list[float] = []
        for row in rows:
            x_val = self._to_number(row.get(x_field))
            y_val = self._to_number(row.get(y_field))
            if x_val is None or y_val is None:
                continue
            x.append(x_val)
            y.append(y_val)
        return x, y

    def _extract_xy_sorted(
        self, rows: Iterable[dict], x_field: str, y_field: str
    ) -> tuple[list[float], list[float]]:
        """Extract x/y pairs and sort by x for proper line chart rendering."""
        pairs: list[tuple[float, float]] = []
        for row in rows:
            x_val = self._to_number(row.get(x_field))
            y_val = self._to_number(row.get(y_field))
            if x_val is None or y_val is None:
                continue
            pairs.append((x_val, y_val))
        pairs.sort(key=lambda p: p[0])
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        return x, y

    def _group_series(
        self, rows: Iterable[dict], series_field: str
    ) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            key = str(row.get(series_field, ""))
            groups[key].append(row)
        return dict(groups)

    def _build_series_matrix(
        self,
        rows: Iterable[dict],
        x_field: str,
        y_field: str,
        series_field: str,
    ) -> tuple[list[str], list[str], list[list[float]]]:
        categories: list[str] = []
        series_names: list[str] = []
        data: dict[str, dict[str, float]] = {}

        for row in rows:
            category = str(row.get(x_field, ""))
            series_name = str(row.get(series_field, ""))
            value = self._to_number(row.get(y_field))
            if value is None:
                continue

            if category not in categories:
                categories.append(category)
            if series_name not in series_names:
                series_names.append(series_name)

            data.setdefault(series_name, {})
            data[series_name][category] = data[series_name].get(category, 0.0) + value

        series_values: list[list[float]] = []
        for series_name in series_names:
            values = [
                data.get(series_name, {}).get(category, 0.0) for category in categories
            ]
            series_values.append(values)

        return categories, series_names, series_values

    def _to_number(self, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, time):
            return self._time_to_seconds(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                # Handle Z suffix (e.g., "2024-01-01T00:00:00Z")
                normalized = value
                if value.endswith("Z"):
                    normalized = value[:-1] + "+00:00"
                try:
                    return datetime.fromisoformat(normalized).timestamp()
                except ValueError:
                    pass
                try:
                    return self._time_to_seconds(time.fromisoformat(normalized))
                except ValueError:
                    pass
                # Try YYYY-MM format (e.g., "2023-06")
                if re.match(r"^\d{4}-\d{2}$", value):
                    try:
                        return datetime.fromisoformat(f"{value}-01").timestamp()
                    except ValueError:
                        pass
                return None
        return None

    def _time_to_seconds(self, value: time) -> float:
        """Convert time-only values to seconds since midnight."""
        return (
            value.hour * 3600
            + value.minute * 60
            + value.second
            + value.microsecond / 1_000_000
        )

    def _safe_color(self, color: str | None, default: str) -> str:
        """Return validated color or default if invalid."""
        if not color:
            return default
        # plotext accepts color names like "red+", "blue", etc.
        # If an invalid color is used, plotext may throw; keep known-good defaults
        valid_colors = {
            "black",
            "red",
            "green",
            "yellow",
            "blue",
            "magenta",
            "cyan",
            "white",
            "black+",
            "red+",
            "green+",
            "yellow+",
            "blue+",
            "magenta+",
            "cyan+",
            "white+",
        }
        return color if color in valid_colors else default

    def _safe_marker(self, marker: str | None, default: str) -> str:
        """Return validated marker or default if invalid."""
        if not marker:
            return default
        # plotext marker options
        valid_markers = {
            "sd",
            "dot",
            "hd",
            "fhd",
            "braille",
            "heart",
            "point",
        }
        return marker if marker in valid_markers else default
