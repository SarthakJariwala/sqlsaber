"""Visualization tools for data plotting."""

import json
from typing import Any

from uniplot import histogram, plot

from .base import Tool
from .enums import ToolCategory, WorkflowPosition
from .registry import register_tool


@register_tool
class PlotDataTool(Tool):
    """Tool for creating terminal plots using uniplot."""

    @property
    def name(self) -> str:
        return "plot_data"

    @property
    def description(self) -> str:
        return "Create a plot of query results."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "y_values": {
                    "type": "array",
                    "items": {"type": ["number", "null"]},
                    "description": "Y-axis data points (required)",
                },
                "x_values": {
                    "type": "array",
                    "items": {"type": ["number", "null"]},
                    "description": "X-axis data points (optional, will use indices if not provided)",
                },
                "plot_type": {
                    "type": "string",
                    "enum": ["line", "scatter", "histogram"],
                    "description": "Type of plot to create (default: line)",
                    "default": "line",
                },
                "title": {
                    "type": "string",
                    "description": "Title for the plot",
                },
                "x_label": {
                    "type": "string",
                    "description": "Label for X-axis",
                },
                "y_label": {
                    "type": "string",
                    "description": "Label for Y-axis",
                },
            },
            "required": ["y_values"],
        }

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VISUALIZATION

    def get_usage_instructions(self) -> str | None:
        """Return usage instructions for this tool."""
        return "Create terminal plots from query results when visualization would enhance understanding of the data."

    def get_priority(self) -> int:
        """Return priority for tool ordering."""
        return 40  # Should come after SQL execution

    def get_workflow_position(self) -> WorkflowPosition:
        """Return workflow position."""
        return WorkflowPosition.VISUALIZATION

    async def execute(self, **kwargs) -> str:
        """Create a terminal plot."""
        y_values = kwargs.get("y_values", [])
        x_values = kwargs.get("x_values")
        plot_type = kwargs.get("plot_type", "line")
        title = kwargs.get("title")
        x_label = kwargs.get("x_label")
        y_label = kwargs.get("y_label")

        try:
            # Validate inputs
            if not y_values:
                return json.dumps({"error": "No data provided for plotting"})

            # Convert to floats if needed
            try:
                y_values = [float(v) if v is not None else None for v in y_values]
                if x_values:
                    x_values = [float(v) if v is not None else None for v in x_values]
            except (ValueError, TypeError) as e:
                return json.dumps({"error": f"Invalid data format: {str(e)}"})

            # Create the plot
            if plot_type == "histogram":
                # For histogram, we only need y_values
                histogram(
                    y_values,
                    title=title,
                    bins=min(20, len(set(y_values))),  # Adaptive bin count
                )
                plot_info = {
                    "type": "histogram",
                    "data_points": len(y_values),
                    "title": title or "Histogram",
                }
            elif plot_type in ["line", "scatter"]:
                # For line/scatter plots
                plot_kwargs = {
                    "ys": y_values,
                    "title": title,
                    "lines": plot_type == "line",
                }

                if x_values:
                    plot_kwargs["xs"] = x_values
                if x_label:
                    plot_kwargs["x_unit"] = x_label
                if y_label:
                    plot_kwargs["y_unit"] = y_label

                plot(**plot_kwargs)

                plot_info = {
                    "type": plot_type,
                    "data_points": len(y_values),
                    "title": title or f"{plot_type.capitalize()} Plot",
                    "has_x_values": x_values is not None,
                }
            else:
                return json.dumps({"error": f"Unsupported plot type: {plot_type}"})

            return json.dumps(
                {"success": True, "plot_rendered": True, "plot_info": plot_info}
            )

        except Exception as e:
            return json.dumps({"error": f"Error creating plot: {str(e)}"})
