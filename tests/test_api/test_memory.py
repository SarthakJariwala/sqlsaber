from __future__ import annotations

import pytest

from sqlsaber import SQLSaber
from sqlsaber.memory.manager import MemoryManager


@pytest.mark.asyncio
async def test_api_memory_text_injected(temp_dir, monkeypatch):
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    saber = SQLSaber(
        database="sqlite:///:memory:",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
        memory="hello memory",
    )

    try:
        prompt = saber.agent.system_prompt_text(include_memory=True)
        assert "hello memory" in prompt
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_api_memory_file_injected(temp_dir, monkeypatch):
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    memory_file = temp_dir / "memory.txt"
    memory_file.write_text("memory from file", encoding="utf-8")

    saber = SQLSaber(
        database="sqlite:///:memory:",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
        memory=memory_file,
    )

    try:
        prompt = saber.agent.system_prompt_text(include_memory=True)
        assert "memory from file" in prompt
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_api_empty_memory_disables_saved_memories(temp_dir, monkeypatch):
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    db_path = temp_dir / "my.db"
    db_path.write_text("", encoding="utf-8")

    memory_manager = MemoryManager()
    memory_manager.add_memory("my", "saved-memory")

    saber = SQLSaber(
        database=f"sqlite:///{db_path}",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
        memory="",
    )

    try:
        prompt = saber.agent.system_prompt_text(include_memory=True)
        assert "saved-memory" not in prompt
    finally:
        await saber.close()
