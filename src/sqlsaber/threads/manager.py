from typing import TYPE_CHECKING

from sqlsaber.config.logging import get_logger
from sqlsaber.threads.storage import ThreadStorage

if TYPE_CHECKING:
    from pydantic_ai.agent import AgentRunResult

logger = get_logger(__name__)


class ThreadManager:
    """
    Manages the state of a conversation thread (session).

    Tracks the current active thread ID and handles persistence of runs
    to the underlying ThreadStorage.
    """

    def __init__(
        self,
        initial_thread_id: str | None = None,
        storage: ThreadStorage | None = None,
    ):
        self.storage = storage or ThreadStorage()
        self.current_thread_id: str | None = initial_thread_id
        # If we start without an ID, the first message will trigger metadata creation
        self.first_message: bool = not self.current_thread_id

    async def end_current_thread(self) -> str | None:
        """
        End the current thread in storage.

        Returns:
            The thread_id that was ended, or None if no thread was active.
        """
        thread_id = self.current_thread_id
        if thread_id:
            await self.storage.end_thread(thread_id)
            return thread_id
        return None

    async def clear_current_thread(self) -> None:
        """End the current thread and reset internal state to start a new one."""
        await self.end_current_thread()
        self.current_thread_id = None
        self.first_message = True

    async def save_run(
        self,
        run_result: "AgentRunResult",
        database_name: str,
        user_query: str,
        model_name: str,
    ) -> list:
        """
        Persist message history from a run result.

        Creates a new thread if one doesn't exist, or updates the existing one.
        Also handles pruning old threads.

        Returns:
            The updated message history from the run.
        """
        try:
            # Persist snapshot to thread storage (create or overwrite)
            # save_snapshot returns the (possibly new) thread_id
            self.current_thread_id = await self.storage.save_snapshot(
                messages_json=run_result.all_messages_json(),
                database_name=database_name,
                thread_id=self.current_thread_id,
            )

            # Save metadata separately (only if it's the first message of a new thread)
            if self.first_message:
                await self.storage.save_metadata(
                    thread_id=self.current_thread_id,
                    title=user_query,
                    model_name=model_name,
                )
                self.first_message = False
        except Exception as e:
            logger.warning("thread_manager.save_failed", error=str(e))
        finally:
            await self.storage.prune_threads()

        return run_result.all_messages()
