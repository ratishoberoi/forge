import pytest
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.artifact_compression import ArtifactCompressor
from backend.runtime.artifact_summary import ArtifactSummary
from backend.runtime.compression_policy import CompressionPolicy
from backend.runtime.summary_context import SummaryContextBuilder


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_artifacts(
    count: int,
    role: str = "PRIMARY_CODER",
    content_prefix: str = "Artifact content",
) -> list[CognitionArtifact]:
    return [
        CognitionArtifact(
            artifact_id=f"a{i}",
            role=role,
            round_id=i + 1,
            task="Task",
            content=f"{content_prefix} {i + 1}",
        )
        for i in range(count)
    ]


def make_compressor(
    max_artifacts: int = 5,
    preserve_recent: int = 2,
    **kwargs,
) -> ArtifactCompressor:
    return ArtifactCompressor(
        CompressionPolicy(
            max_artifacts=max_artifacts,
            preserve_recent=preserve_recent,
            **kwargs,
        )
    )


# ── should_compress ───────────────────────────────────────────────────────────

def test_should_compress_true_when_over_limit():
    compressor = make_compressor(max_artifacts=3)
    assert compressor.should_compress(build_artifacts(5)) is True


def test_should_compress_false_when_at_limit():
    compressor = make_compressor(max_artifacts=5)
    assert compressor.should_compress(build_artifacts(5)) is False


def test_should_compress_false_when_under_limit():
    compressor = make_compressor(max_artifacts=10)
    assert compressor.should_compress(build_artifacts(3)) is False


# ── compress ──────────────────────────────────────────────────────────────────

def test_compress_returns_artifact_summary():
    compressor = make_compressor(max_artifacts=3, preserve_recent=2)
    summary = compressor.compress(build_artifacts(5))
    assert isinstance(summary, ArtifactSummary)


def test_compress_preserves_recent_excluded_from_summary():
    compressor = make_compressor(max_artifacts=5, preserve_recent=2)
    artifacts = build_artifacts(5)
    summary = compressor.compress(artifacts)

    # rounds 1-3 compressible, rounds 4-5 preserved (excluded from summary)
    assert 1 in summary.rounds
    assert 2 in summary.rounds
    assert 3 in summary.rounds
    assert 4 not in summary.rounds
    assert 5 not in summary.rounds


def test_compress_source_roles_populated():
    compressor = make_compressor(max_artifacts=3, preserve_recent=1)
    summary = compressor.compress(build_artifacts(5))
    assert "PRIMARY_CODER" in summary.source_roles


def test_compress_content_contains_artifact_content():
    compressor = make_compressor(max_artifacts=3, preserve_recent=1)
    summary = compressor.compress(build_artifacts(5))
    assert "Artifact content" in summary.content


def test_compress_metadata_populated():
    compressor = make_compressor(max_artifacts=5, preserve_recent=2)
    summary = compressor.compress(build_artifacts(6))
    assert summary.metadata["policy_max_artifacts"] == 5
    assert summary.metadata["policy_preserve_recent"] == 2
    assert summary.metadata["original_count"] == 6


def test_compress_empty_raises():
    compressor = make_compressor()
    with pytest.raises(ValueError, match="empty"):
        compressor.compress([])


def test_compress_drops_empty_content():
    compressor = ArtifactCompressor(
        CompressionPolicy(
            max_artifacts=3,
            preserve_recent=1,
            drop_empty_content=True,
        )
    )
    artifacts = [
        CognitionArtifact(
            artifact_id=f"a{i}",
            role="PRIMARY_CODER",
            round_id=i + 1,
            task="Task",
            content="" if i % 2 == 0 else f"Content {i + 1}",
        )
        for i in range(5)
    ]
    summary = compressor.compress(artifacts)
    assert "(no compressible content" not in summary.content or "Content" in summary.content


def test_compress_deduplicates():
    compressor = ArtifactCompressor(
        CompressionPolicy(max_artifacts=3, preserve_recent=1, deduplicate=True)
    )
    artifact = CognitionArtifact(
        artifact_id="dup",
        role="PRIMARY_CODER",
        round_id=1,
        task="Task",
        content="Duplicate content",
    )
    summary = compressor.compress([artifact, artifact, artifact])
    assert summary.content.count("Duplicate content") == 1


# ── split ─────────────────────────────────────────────────────────────────────

def test_split_returns_correct_windows():
    compressor = make_compressor(max_artifacts=5, preserve_recent=2)
    artifacts = build_artifacts(5)
    compressible, preserved = compressor.split(artifacts)

    assert len(compressible) == 3
    assert len(preserved) == 2
    assert preserved[-1].round_id == 5


def test_split_preserve_zero_returns_all_compressible():
    compressor = ArtifactCompressor(
        CompressionPolicy(max_artifacts=5, preserve_recent=0)
    )
    artifacts = build_artifacts(5)
    compressible, preserved = compressor.split(artifacts)

    assert len(compressible) == 5
    assert preserved == []


# ── priority roles ────────────────────────────────────────────────────────────

def test_compress_priority_roles_appear_first_in_content():
    artifacts = [
        CognitionArtifact(artifact_id="a1", role="PRIMARY_CODER", round_id=1, task="T", content="Coder content"),
        CognitionArtifact(artifact_id="a2", role="JUDGE", round_id=2, task="T", content="Judge content"),
        CognitionArtifact(artifact_id="a3", role="PRIMARY_CODER", round_id=3, task="T", content="Coder 2"),
        CognitionArtifact(artifact_id="a4", role="JUDGE", round_id=4, task="T", content="Judge 2"),
        CognitionArtifact(artifact_id="a5", role="PRIMARY_CODER", round_id=5, task="T", content="Coder 3"),
    ]
    compressor = ArtifactCompressor(
        CompressionPolicy(
            max_artifacts=5,
            preserve_recent=1,
            priority_roles=["JUDGE"],
        )
    )
    summary = compressor.compress(artifacts)
    assert summary.content.index("Judge") < summary.content.index("Coder")


# ── SummaryContextBuilder ─────────────────────────────────────────────────────

def test_summary_context_builder_basic():
    compressor = make_compressor(max_artifacts=3, preserve_recent=1)
    summary = compressor.compress(build_artifacts(5))
    builder = SummaryContextBuilder()
    context = builder.build([summary])

    assert "[SUMMARY" in context
    assert "Artifact content" in context


def test_summary_context_builder_empty_returns_empty():
    builder = SummaryContextBuilder()
    assert builder.build([]) == ""


def test_summary_context_builder_truncation():
    compressor = make_compressor(max_artifacts=3, preserve_recent=1)
    summary = compressor.compress(build_artifacts(5))
    builder = SummaryContextBuilder()
    context = builder.build([summary], max_chars=20)

    assert "truncated" in context


def test_summary_context_builder_for_role():
    compressor = make_compressor(max_artifacts=3, preserve_recent=1)
    summary = compressor.compress(build_artifacts(5))
    builder = SummaryContextBuilder()

    context = builder.build_for_role([summary], "PRIMARY_CODER")
    assert "Artifact content" in context

    empty = builder.build_for_role([summary], "NONEXISTENT")
    assert empty == ""


def test_summary_context_builder_summarize():
    compressor = make_compressor(max_artifacts=3, preserve_recent=1)
    summary = compressor.compress(build_artifacts(5))
    builder = SummaryContextBuilder()
    debug = builder.summarize([summary])

    assert "roles=" in debug
    assert "rounds=" in debug
    assert "span=" in debug