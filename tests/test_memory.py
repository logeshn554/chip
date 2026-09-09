"""Tests for the Memory System."""

import os
import shutil
import tempfile

import pytest

from agent.schemas import Document
from memory.knowledge import KnowledgeMemory
from memory.experience import ExperienceMemory
from memory.design_memory import DesignMemory
from memory.trajectory import TrajectoryStore
from agent.schemas import Episode, TrajectoryStep


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestKnowledgeMemory:
    def test_ingest_and_query(self, tmp_dir):
        mem = KnowledgeMemory(
            chroma_path=os.path.join(tmp_dir, "chroma"),
            config={"collection": "test_knowledge"},
        )

        mem.ingest(
            "SystemVerilog is a hardware description and verification language. "
            "It supports modules, interfaces, and assertions.",
            metadata={"source": "test", "topic": "systemverilog"},
        )

        results = mem.query("What is SystemVerilog?", n=3)
        assert len(results) > 0
        assert "SystemVerilog" in results[0].content

    def test_empty_query(self, tmp_dir):
        mem = KnowledgeMemory(
            chroma_path=os.path.join(tmp_dir, "chroma"),
            config={"collection": "test_empty"},
        )
        results = mem.query("anything")
        assert results == []

    def test_chunking(self, tmp_dir):
        mem = KnowledgeMemory(
            chroma_path=os.path.join(tmp_dir, "chroma"),
            config={"collection": "test_chunks", "chunk_size": 10, "chunk_overlap": 2},
        )
        text = " ".join(["word"] * 50)
        count = mem.ingest(text)
        assert count > 1  # Should be chunked


class TestExperienceMemory:
    def test_record_and_query(self, tmp_dir):
        mem = ExperienceMemory(
            chroma_path=os.path.join(tmp_dir, "chroma"),
            config={"collection": "test_experience"},
        )

        mem.record(
            task="generate_rtl",
            action="Generated ALU module",
            result="Compilation failed: missing semicolon",
            errors=["Error: missing semicolon on line 42"],
            success=False,
        )

        results = mem.query("semicolon error", n=3)
        assert len(results) > 0
        assert "semicolon" in results[0].content.lower()

    def test_find_similar_errors(self, tmp_dir):
        mem = ExperienceMemory(
            chroma_path=os.path.join(tmp_dir, "chroma"),
            config={"collection": "test_errors"},
        )

        mem.record(
            task="generate_rtl",
            action="Generated module",
            result="Width mismatch on port A",
            errors=["Width mismatch"],
            success=False,
            fix="Changed port A width from 8 to 16 bits",
        )

        results = mem.find_similar_errors("port width mismatch", n=3)
        assert len(results) > 0


class TestDesignMemory:
    def test_save_and_query(self, tmp_dir):
        mem = DesignMemory(
            chroma_path=os.path.join(tmp_dir, "chroma"),
            config={
                "collection": "test_designs",
                "designs_dir": os.path.join(tmp_dir, "designs"),
            },
        )

        design_id = mem.save(
            name="test_alu",
            code="module alu(input a, input b, output y); assign y = a + b; endmodule",
            metadata={"type": "alu"},
        )

        assert design_id is not None

        results = mem.query("ALU adder module", n=3)
        assert len(results) > 0
        assert "alu" in results[0].content.lower()

    def test_get_design_code(self, tmp_dir):
        mem = DesignMemory(
            chroma_path=os.path.join(tmp_dir, "chroma"),
            config={
                "collection": "test_get_code",
                "designs_dir": os.path.join(tmp_dir, "designs"),
            },
        )

        code = "module test; endmodule"
        design_id = mem.save(name="test_mod", code=code)

        retrieved = mem.get_design_code(design_id)
        assert retrieved == code


class TestTrajectoryStore:
    def test_save_and_load(self, tmp_dir):
        store = TrajectoryStore({"store_dir": os.path.join(tmp_dir, "traj")})

        episode = Episode(task="Test task", final_reward=0.85, success=True)
        episode.steps.append(TrajectoryStep(
            step_index=0,
            state_summary="initial",
            action="GENERATE_RTL",
            observation="Generated module",
            reward=0.85,
        ))

        import asyncio
        filepath = asyncio.run(store.save_episode(episode))

        assert os.path.exists(filepath)

        loaded = store.load_episode(episode.episode_id)
        assert loaded is not None
        assert loaded.task == "Test task"
        assert loaded.final_reward == 0.85
        assert len(loaded.steps) == 1

    def test_stats(self, tmp_dir):
        store = TrajectoryStore({"store_dir": os.path.join(tmp_dir, "traj")})
        stats = store.get_stats()
        assert stats["total_episodes"] == 0

    def test_filtered_loading(self, tmp_dir):
        store = TrajectoryStore({"store_dir": os.path.join(tmp_dir, "traj")})

        import asyncio

        for i, reward in enumerate([0.3, 0.6, 0.9]):
            ep = Episode(
                task=f"Task {i}",
                final_reward=reward,
                success=reward >= 0.7,
            )
            asyncio.run(store.save_episode(ep))

        high_reward = store.load_all_episodes(min_reward=0.5)
        assert len(high_reward) == 2

        successful = store.load_all_episodes(success_only=True)
        assert len(successful) == 1
