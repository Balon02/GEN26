from __future__ import annotations

import curses
from dataclasses import dataclass

from gen26.chunking import TokenBudget, pack_chunks
from gen26.paper_tree import (
    DigestMode,
    IncludeStatus,
    PaperNode,
    recompute_parent_totals,
)


@dataclass
class Row:
    node: PaperNode
    depth: int
    bundled_by: PaperNode | None = None
    excluded_by: PaperNode | None = None


@dataclass
class PlannerState:
    root: PaperNode
    budget: TokenBudget
    expanded: set[int]
    cursor: int = 0
    message: str = ""


def terminal_plan(root: PaperNode, budget: TokenBudget):
    state = PlannerState(
        root=root,
        budget=budget,
        expanded={root.order, *(child.order for child in root.children)},
    )
    return curses.wrapper(run_planner, state)


def run_planner(screen, state: PlannerState):
    curses.curs_set(0)
    screen.keypad(True)
    curses.use_default_colors()
    init_colors()

    while True:
        rows = visible_rows(state.root, state.expanded)
        state.cursor = min(state.cursor, max(0, len(rows) - 1))
        draw(screen, state, rows)
        key = screen.getch()

        if key in (curses.KEY_UP, ord("k")):
            state.cursor = max(0, state.cursor - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            state.cursor = min(len(rows) - 1, state.cursor + 1)
        elif key == curses.KEY_RIGHT:
            expand_current(state, rows)
        elif key == curses.KEY_LEFT:
            collapse_or_parent(state, rows)
        elif key == ord("i"):
            current(rows, state).include_status = IncludeStatus.INCLUDE
            recompute_parent_totals(state.root)
        elif key == ord("x"):
            current(rows, state).include_status = IncludeStatus.EXCLUDE
            recompute_parent_totals(state.root)
        elif key == ord("b"):
            current(rows, state).digest_mode = DigestMode.WHOLE
        elif key == ord("s"):
            current(rows, state).digest_mode = DigestMode.SPLIT
            state.expanded.add(current(rows, state).order)
        elif key == ord("a"):
            current(rows, state).digest_mode = DigestMode.AUTO
        elif key in (ord("\n"), curses.KEY_ENTER, 10, 13):
            chunks, errors = validate_plan(state.root, state.budget)
            if not errors:
                return chunks
            state.message = "Cannot continue: split over-budget bundled nodes first."
        elif key == ord("q"):
            raise KeyboardInterrupt("Planner cancelled")


def init_colors() -> None:
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)
    curses.init_pair(3, curses.COLOR_CYAN, -1)
    curses.init_pair(4, curses.COLOR_YELLOW, -1)
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_RED)


def visible_rows(root: PaperNode, expanded: set[int]) -> list[Row]:
    rows: list[Row] = []

    def visit(
        node: PaperNode,
        depth: int,
        bundled_by: PaperNode | None,
        excluded_by: PaperNode | None,
    ) -> None:
        rows.append(
            Row(
                node=node,
                depth=depth,
                bundled_by=bundled_by,
                excluded_by=excluded_by,
            )
        )
        if node.order not in expanded:
            return
        child_bundled_by = bundled_by
        if node.digest_mode == DigestMode.WHOLE:
            child_bundled_by = node
        child_excluded_by = excluded_by
        if node.include_status == IncludeStatus.EXCLUDE:
            child_excluded_by = node
        for child in node.children:
            visit(child, depth + 1, child_bundled_by, child_excluded_by)

    visit(root, 0, None, None)
    return rows


def draw(screen, state: PlannerState, rows: list[Row]) -> None:
    screen.erase()
    height, width = screen.getmaxyx()
    chunks, errors = validate_plan(state.root, state.budget)

    add_line(screen, 0, 0, "GEN26 Paper Planner", curses.A_BOLD)
    add_line(
        screen,
        1,
        0,
        "arrows move/expand/collapse  i include  x exclude  b bundle  s split  a auto  Enter continue  q quit",
    )
    add_line(
        screen,
        2,
        0,
        f"planned chunks: {len(chunks)}  over budget: {len(errors)}  "
        f"safe input: {state.budget.usable_input_tokens}  "
        f"text limit: {state.budget.chunk_text_tokens}",
        curses.color_pair(6) if errors else curses.color_pair(1),
    )
    if state.message:
        add_line(screen, 3, 0, state.message, curses.color_pair(6))

    start = max(0, state.cursor - max(0, height - 8))
    body_top = 5
    body_height = max(1, height - body_top - 1)
    for line_index, row in enumerate(rows[start : start + body_height], start=body_top):
        row_index = start + line_index - body_top
        is_cursor = row_index == state.cursor
        text = format_row(row, state.expanded, row.node.order in errors, is_cursor)
        attr = row_attr(row, is_cursor, row.node.order in errors)
        add_line(screen, line_index, 0, text[: width - 1], attr)

    screen.refresh()


def format_row(
    row: Row,
    expanded: set[int],
    over_budget: bool,
    is_cursor: bool = False,
) -> str:
    node = row.node
    if node.children:
        marker = "-" if node.order in expanded else "+"
    else:
        marker = " "
    cursor = ">" if is_cursor else " "
    warning = " OVER" if over_budget else ""
    indent = "  " * row.depth
    return (
        f"{cursor}{indent}{marker} {display_status(row):<3} "
        f"{display_mode(row):<7} {node.order:04d} "
        f"{node.node_type:<12} {node.token_count:>6} tok "
        f"{node.display_label()}{warning}"
    )


def display_status(row: Row) -> str:
    if row.excluded_by is not None and row.excluded_by is not row.node:
        return "exc"
    return row.node.include_status.value[:3]


def display_mode(row: Row) -> str:
    if row.bundled_by is not None and row.bundled_by is not row.node:
        return "in-bndl"
    if row.node.digest_mode == DigestMode.WHOLE:
        return "bundle"
    return row.node.digest_mode.value


def row_attr(row: Row, is_cursor: bool, over_budget: bool) -> int:
    node = row.node
    if over_budget:
        attr = curses.color_pair(6) | curses.A_BOLD
    elif row.excluded_by is not None:
        attr = curses.color_pair(2) | curses.A_DIM
    elif row.bundled_by is not None and row.bundled_by is not node:
        attr = curses.color_pair(3) | curses.A_DIM
    elif node.digest_mode == DigestMode.WHOLE:
        attr = curses.color_pair(3)
    elif node.digest_mode == DigestMode.SPLIT:
        attr = curses.color_pair(4)
    else:
        attr = curses.color_pair(1)
    if is_cursor:
        attr |= curses.A_REVERSE
    return attr


def add_line(screen, y: int, x: int, text: str, attr: int = 0) -> None:
    height, width = screen.getmaxyx()
    if y >= height:
        return
    screen.addstr(y, x, text[: max(0, width - x - 1)], attr)


def current(rows: list[Row], state: PlannerState) -> PaperNode:
    return rows[state.cursor].node


def expand_current(state: PlannerState, rows: list[Row]) -> None:
    node = current(rows, state)
    if node.children:
        state.expanded.add(node.order)


def collapse_or_parent(state: PlannerState, rows: list[Row]) -> None:
    row = rows[state.cursor]
    node = row.node
    if node.children and node.order in state.expanded:
        state.expanded.remove(node.order)
        return

    for index in range(state.cursor - 1, -1, -1):
        if rows[index].depth < row.depth:
            state.cursor = index
            return


def validate_plan(root: PaperNode, budget: TokenBudget):
    try:
        chunks = pack_chunks(root, budget)
    except ValueError as exc:
        return [], over_budget_nodes_from_error(root, str(exc))

    errors = {
        chunk.nodes[0].order
        for chunk in chunks
        if chunk.token_count > budget.chunk_text_tokens
    }
    return chunks, errors


def over_budget_nodes_from_error(root: PaperNode, message: str) -> set[int]:
    for node in root.walk():
        if f"{node.order:04d}" in message:
            return {node.order}
    return set()
