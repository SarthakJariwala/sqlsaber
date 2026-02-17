from sqlsaber.knowledge.models import KnowledgeEntry


def test_knowledge_entry_serialization_round_trip():
    entry = KnowledgeEntry(
        id="entry-1",
        database_name="analytics",
        name="Monthly Revenue",
        description="Use this metric for monthly reporting.",
        sql="SELECT date_trunc('month', created_at) AS month FROM orders",
        source="team-wiki",
        created_at=1700000000.0,
        updated_at=1700000100.0,
    )

    serialized = entry.to_dict()
    restored = KnowledgeEntry.from_dict(serialized)

    assert restored == entry
    assert restored.formatted_created_at()
    assert restored.formatted_updated_at()
