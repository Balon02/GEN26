from __future__ import annotations

from gen26.chunking import TokenBudget, pack_chunks
from gen26.paper_tree import DigestMode, IncludeStatus, PaperNode, print_tree


HELP = """Commands:
  ls                 show children of the current node
  tree               show the full tree
  cd ID              move to a node, for example: cd 0015
  up                 move to the parent node
  in ID              include a node/subtree
  out ID             exclude a node/subtree
  whole ID           digest a node/subtree in one prompt
  split ID           split a node into its children
  auto ID            clear whole/split override
  plan               show planned digestion chunks
  done               accept the plan
  help               show this help
"""


def interactive_plan(root: PaperNode, budget: TokenBudget) -> list:
    parents = parent_map(root)
    current = root
    print("\nInteractive paper planner")
    print("Choose what to include and whether each node is digested whole or split.")
    print("Default planning digests at subsection level and splits oversized nodes.")
    print("Type 'help' for commands.\n")
    print_node(current)

    while True:
        command = input(f"planner:{current.order:04d}> ").strip()
        if not command:
            continue
        parts = command.split()
        action = parts[0].lower()

        try:
            if action == "help":
                print(HELP)
            elif action == "ls":
                print_children(current)
            elif action == "tree":
                print(print_tree(root))
            elif action == "cd":
                current = find_required(root, parts)
                print_node(current)
            elif action == "up":
                current = parents.get(current.order, root)
                print_node(current)
            elif action == "in":
                find_required(root, parts).include_status = IncludeStatus.INCLUDE
            elif action == "out":
                find_required(root, parts).include_status = IncludeStatus.EXCLUDE
            elif action == "whole":
                find_required(root, parts).digest_mode = DigestMode.WHOLE
            elif action == "split":
                find_required(root, parts).digest_mode = DigestMode.SPLIT
            elif action == "auto":
                find_required(root, parts).digest_mode = DigestMode.AUTO
            elif action == "plan":
                show_plan(root, budget)
            elif action == "done":
                chunks = pack_chunks(root, budget)
                print(f"Accepted {len(chunks)} planned chunks.")
                return chunks
            else:
                print("Unknown command. Type 'help'.")
        except ValueError as exc:
            print(exc)


def parent_map(root: PaperNode) -> dict[int, PaperNode]:
    parents: dict[int, PaperNode] = {}

    def visit(node: PaperNode) -> None:
        for child in node.children:
            parents[child.order] = node
            visit(child)

    visit(root)
    return parents


def find_required(root: PaperNode, parts: list[str]) -> PaperNode:
    if len(parts) != 2:
        raise ValueError("Command needs exactly one node ID.")
    try:
        order = int(parts[1])
    except ValueError as exc:
        raise ValueError("Node ID must be a number like 0015.") from exc
    for node in root.walk():
        if node.order == order:
            return node
    raise ValueError(f"No node {order:04d}.")


def print_node(node: PaperNode) -> None:
    print(
        f"{node.order:04d} {node.node_type} {node.token_count} tok "
        f"{node.include_status.value} {node.digest_mode.value} "
        f"{node.display_label()}"
    )
    print_children(node)


def print_children(node: PaperNode) -> None:
    if not node.children:
        print("  no children")
        return
    for child in node.children:
        print(
            f"  {child.order:04d} {child.node_type:<12} "
            f"{child.token_count:>5} tok {child.include_status.value:<7} "
            f"{child.digest_mode.value:<5} {child.display_label()}"
        )


def show_plan(root: PaperNode, budget: TokenBudget) -> None:
    chunks = pack_chunks(root, budget)
    for chunk in chunks:
        node = chunk.nodes[0]
        print(
            f"{chunk.index:02d}. {chunk.token_count:>5} tok "
            f"{node.order:04d} {node.node_type:<12} {node.display_label()}"
        )
