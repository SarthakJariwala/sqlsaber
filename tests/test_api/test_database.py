import pytest

from sqlsaber import SQLSaber, SQLSaberOptions
from sqlsaber.config.settings import Config


@pytest.mark.asyncio
async def test_api_multiple_csvs_create_multiple_views(temp_dir):
    users = temp_dir / "users.csv"
    orders = temp_dir / "orders.csv"

    users.write_text("id,name\n1,Alice\n2,Bob\n", encoding="utf-8")
    orders.write_text("id,user_id,total\n10,1,9.99\n11,2,20.00\n", encoding="utf-8")

    saber = SQLSaber(
        options=SQLSaberOptions(
            database=[str(users), str(orders)],
            settings=Config.in_memory(
                model_name="anthropic:claude-3-5-sonnet",
                api_keys={"anthropic": "test-key"},
            ),
        )
    )

    try:
        rows = await saber.connection.execute_query(
            'SELECT u.name, o.total FROM "users" u JOIN "orders" o ON u.id = o.user_id ORDER BY o.id'
        )
        assert rows == [
            {"name": "Alice", "total": 9.99},
            {"name": "Bob", "total": 20.0},
        ]
    finally:
        await saber.close()
