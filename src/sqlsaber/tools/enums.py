"""Enums for tool categories and workflow positions."""

from enum import Enum


class ToolCategory(Enum):
    """Tool categories for organizing and filtering tools."""

    GENERAL = "general"
    SQL = "sql"
    VISUALIZATION = "visualization"


class WorkflowPosition(Enum):
    """Workflow positions for organizing tools by usage order."""

    DISCOVERY = "discovery"
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    VISUALIZATION = "visualization"
    OTHER = "other"
