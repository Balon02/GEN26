from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gen26.chunking import ChunkPlan, TokenBudget
from gen26.paper_tree import DigestMode, IncludeStatus, PaperNode


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, output_file: Path) -> None:
        self.output_file = output_file
        self.state_file = output_file.with_suffix(".json")
        self.log_file = output_file.with_suffix(".log.jsonl")
        self.state: dict[str, Any] = {}

    def create(
        self,
        source: Path,
        runtime,
        budget: TokenBudget,
        root: PaperNode,
        chunks: list[ChunkPlan],
    ) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.write_text("", encoding="utf-8")
        self.state = {
            "version": 1,
            "status": "running",
            "source": str(source),
            "output_file": str(self.output_file),
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "runtime": {
                "model": "google/gemma-3/flax/gemma3-4b-it",
                "cache_length": runtime.cache_length,
                "safe_input_tokens": runtime.safe_input_tokens,
                "max_output_tokens": runtime.max_output_tokens,
                "image_size": runtime.image_size,
                "image_tokens": 256,
            },
            "budget": budget_to_dict(budget),
            "plan_version": 1,
            "node_states": node_states(root),
            "chunks": chunk_records(chunks),
            "completed_summaries": [],
            "rolling_memory": "",
            "last_completed_chunk": 0,
        }
        self.save()
        self.log("run_started", source=str(source), chunks=len(chunks))

    def load(self) -> dict[str, Any]:
        self.state = json.loads(self.state_file.read_text(encoding="utf-8"))
        return self.state

    def save(self) -> None:
        self.state["updated_at"] = now_iso()
        self.state_file.write_text(
            json.dumps(self.state, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def log(self, event: str, **fields) -> None:
        record = {"time": now_iso(), "event": event, **fields}
        with self.log_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True))
            file.write("\n")

    def append_markdown(self, text: str) -> None:
        with self.output_file.open("a", encoding="utf-8") as file:
            file.write(text)

    def mark_interrupted_chunks(self) -> None:
        changed = False
        for chunk in self.state.get("chunks", []):
            if chunk.get("status") == "running":
                chunk["status"] = "interrupted"
                chunk["error"] = "process stopped before marking chunk complete"
                changed = True
                self.log("chunk_interrupted", chunk=chunk["index"])
        if changed:
            self.state["status"] = "interrupted"
            self.save()

    def update_plan(self, root: PaperNode, chunks: list[ChunkPlan]) -> int:
        old_chunks = self.state.get("chunks", [])
        completed = [chunk for chunk in old_chunks if chunk.get("status") == "complete"]

        prefix = 0
        while prefix < len(chunks) and prefix < len(completed):
            if completed[prefix].get("node_orders") != chunk_signature(chunks[prefix]):
                break
            prefix += 1

        new_records = chunk_records(chunks)
        for index in range(prefix):
            new_records[index].update(
                {
                    "status": "complete",
                    "completed_at": completed[index].get("completed_at"),
                    "prompt_stats": completed[index].get("prompt_stats", {}),
                    "summary": completed[index].get("summary", ""),
                }
            )

        self.state["plan_version"] = int(self.state.get("plan_version", 1)) + 1
        self.state["node_states"] = node_states(root)
        self.state["chunks"] = new_records
        self.state["last_completed_chunk"] = prefix
        self.state["completed_summaries"] = self.state.get("completed_summaries", [])[:prefix]
        self.state["status"] = "running"
        self.save()
        self.log("plan_updated", plan_version=self.state["plan_version"], resume_at=prefix + 1)
        return prefix

    def chunk_started(self, chunk: ChunkPlan, prompt_stats: dict[str, int]) -> None:
        record = self.chunk_record(chunk.index)
        record["status"] = "running"
        record["started_at"] = now_iso()
        record["prompt_stats"] = prompt_stats
        self.save()
        self.log("chunk_started", chunk=chunk.index, **prompt_stats)

    def chunk_completed(
        self,
        chunk: ChunkPlan,
        summary: str,
        rolling_memory: str,
    ) -> None:
        record = self.chunk_record(chunk.index)
        record["status"] = "complete"
        record["completed_at"] = now_iso()
        record["summary"] = summary
        self.state["last_completed_chunk"] = chunk.index
        summaries = self.state.setdefault("completed_summaries", [])
        while len(summaries) < chunk.index:
            summaries.append("")
        summaries[chunk.index - 1] = summary
        self.state["rolling_memory"] = rolling_memory
        self.save()
        self.log("chunk_completed", chunk=chunk.index, summary_chars=len(summary))

    def image_started(
        self,
        chunk_index: int,
        image_index: int,
        image_name: str,
        prompt_tokens: int,
    ) -> None:
        record = self.chunk_record(chunk_index)
        record["status"] = "running"
        image_records = record.setdefault("image_prepass", [])
        image_records.append(
            {
                "index": image_index,
                "name": image_name,
                "status": "running",
                "prompt_tokens": prompt_tokens,
                "started_at": now_iso(),
            }
        )
        self.save()
        self.log(
            "image_started",
            chunk=chunk_index,
            image=image_index,
            image_name=image_name,
            prompt_tokens=prompt_tokens,
            image_tokens=256,
        )

    def image_completed(
        self,
        chunk_index: int,
        image_index: int,
        image_name: str,
        summary_chars: int,
    ) -> None:
        record = self.chunk_record(chunk_index)
        for image_record in reversed(record.get("image_prepass", [])):
            if image_record.get("index") == image_index:
                image_record["status"] = "complete"
                image_record["completed_at"] = now_iso()
                image_record["summary_chars"] = summary_chars
                break
        self.save()
        self.log(
            "image_completed",
            chunk=chunk_index,
            image=image_index,
            image_name=image_name,
            summary_chars=summary_chars,
        )

    def image_failed(
        self,
        chunk_index: int,
        image_index: int,
        image_name: str,
        error: BaseException,
    ) -> None:
        record = self.chunk_record(chunk_index)
        for image_record in reversed(record.get("image_prepass", [])):
            if image_record.get("index") == image_index:
                image_record["status"] = "failed"
                image_record["failed_at"] = now_iso()
                image_record["error_type"] = type(error).__name__
                image_record["error"] = str(error)
                break
        self.save()
        self.log(
            "image_failed",
            chunk=chunk_index,
            image=image_index,
            image_name=image_name,
            error_type=type(error).__name__,
            error=str(error),
        )

    def chunk_failed(self, chunk: ChunkPlan, error: BaseException) -> None:
        record = self.chunk_record(chunk.index)
        record["status"] = "failed"
        record["failed_at"] = now_iso()
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
        self.state["status"] = "failed"
        self.save()
        self.log(
            "chunk_failed",
            chunk=chunk.index,
            error_type=type(error).__name__,
            error=str(error),
        )

    def final_started(self, prompt_tokens: int) -> None:
        self.state["final_prompt_tokens"] = prompt_tokens
        self.save()
        self.log("final_started", prompt_tokens=prompt_tokens)

    def memory_compaction_started(self, before_tokens: int, target_tokens: int) -> None:
        self.state["memory_compaction_running"] = True
        self.state["memory_compaction_before_tokens"] = before_tokens
        self.state["memory_compaction_target_tokens"] = target_tokens
        self.save()
        self.log(
            "memory_compaction_started",
            before_tokens=before_tokens,
            target_tokens=target_tokens,
        )

    def memory_compaction_completed(self, before_tokens: int, after_tokens: int) -> None:
        self.state["memory_compaction_running"] = False
        self.state["memory_compaction_after_tokens"] = after_tokens
        self.save()
        self.log(
            "memory_compaction_completed",
            before_tokens=before_tokens,
            after_tokens=after_tokens,
        )

    def final_failed(self, error: BaseException) -> None:
        self.state["status"] = "failed"
        self.state["final_error_type"] = type(error).__name__
        self.state["final_error"] = str(error)
        self.save()
        self.log("final_failed", error_type=type(error).__name__, error=str(error))

    def finish(self, final_abstract: str) -> None:
        self.state["status"] = "complete"
        self.state["final_abstract_chars"] = len(final_abstract)
        self.save()
        self.log("run_completed")

    def chunk_record(self, index: int) -> dict[str, Any]:
        for chunk in self.state["chunks"]:
            if chunk["index"] == index:
                return chunk
        raise ValueError(f"No chunk {index}")


def budget_to_dict(budget: TokenBudget) -> dict[str, int]:
    return {
        "cache_length": budget.cache_length,
        "usable_input_tokens": budget.usable_input_tokens,
        "reserved_output_tokens": budget.reserved_output_tokens,
        "rolling_memory_tokens": budget.rolling_memory_tokens,
        "instruction_tokens": budget.instruction_tokens,
        "chunk_text_tokens": budget.chunk_text_tokens,
    }


def chunk_records(chunks: list[ChunkPlan]) -> list[dict[str, Any]]:
    return [
        {
            "index": chunk.index,
            "node_orders": chunk_signature(chunk),
            "node_titles": [node.display_label() for node in chunk.nodes],
            "node_types": [node.node_type for node in chunk.nodes],
            "token_count": chunk.token_count,
            "status": "pending",
        }
        for chunk in chunks
    ]


def chunk_signature(chunk: ChunkPlan) -> list[int]:
    return [node.order for node in chunk.nodes]


def node_states(root: PaperNode) -> list[dict[str, Any]]:
    return [
        {
            "order": node.order,
            "include_status": node.include_status.value,
            "digest_mode": node.digest_mode.value,
        }
        for node in root.walk()
    ]


def apply_node_states(root: PaperNode, states: list[dict[str, Any]]) -> None:
    by_order = {node.order: node for node in root.walk()}
    for state in states:
        node = by_order.get(state["order"])
        if node is None:
            continue
        node.include_status = IncludeStatus(state["include_status"])
        node.digest_mode = DigestMode(state["digest_mode"])
