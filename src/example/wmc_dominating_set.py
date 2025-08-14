# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import TypeAlias

from evaluation.graph_structure import Graph

MOD = 1_000_000_007


@dataclass(frozen=True)
class DPState:
    """
    Dynamic Programming state for the Dominating Set problem on tree decompositions.
    This dataclass encapsulates the complete state information needed at each bag
    during the tree decomposition-based dynamic programming algorithm.

    Fields:
    ( 1 ) assign_mask : int        Bitmask indicating which vertices in the current bag
                                   are assigned to the dominating set (1 = IN, 0 = OUT).
                                   Bit i corresponds to the i-th vertex in the sorted bag.

    ( 2 ) need_mask   : int        Bitmask indicating which OUT vertices in the current bag
                                   still require domination (1 = needs domination, 0 = already dominated).
                                   Only meaningful for OUT vertices (assign_mask bit = 0).
                                   IN vertices never have the need bit set since they dominate themselves.

    State Invariants:
    - For any bit position i: if (assign_mask >> i) & 1 == 1, then (need_mask >> i) & 1 == 0
    - The need_mask only tracks vertices that are OUT and not yet dominated by adjacent IN vertices
    - When a vertex is forgotten, it must not have the need bit set (invalid state otherwise)

    Usage in Algorithm:
    - LEAF: Initialize with vertex either IN (assign=1, need=0) or OUT (assign=0, need=1)
    - INTRODUCE: Insert new bit at appropriate position, update domination status
    - FORGET: Remove bit at appropriate position, reject states with undominated OUT vertices
    - JOIN: Merge compatible states (same assignment), combine need masks with AND operation

    All fields are immutable and hashable, making this object suitable as a dictionary key.
    """

    assign_mask: int
    need_mask: int


@dataclass
class DPValue:
    """
    Dynamic Programming value representing the computational result for a given state.
    This dataclass replaces the previous Tuple[int, int] representation with a more
    structured and self-documenting approach for weighted model counting.

    Fields:
    ( 1 ) count  : int             Number of distinct vertex subsets (dominating sets) that
                                   achieve the current state configuration. This counts the
                                   multiplicity of solutions that lead to the same DP state.

    ( 2 ) weight : int             Total weighted sum across all vertex subsets that achieve
                                   the current state configuration. Each dominating set contributes
                                   its total vertex weight sum to this field.

    Implementation Details:
    - Both fields are maintained modulo MOD (1,000,000,007)
    - The count field enables tracking the number of valid dominating sets
    - The weight field accumulates the total weight contribution from all valid sets
    - When combining values from different DP branches:
      * Counts are multiplied for independent choices
      * Weights are combined using inclusion-exclusion principle to avoid double-counting

    This structure enables simultaneous tracking of both solution count and cumulative
    weight during the tree decomposition DP computation.
    """

    count: int
    weight: int


Bag: TypeAlias = set[int]  # a bag is a set of vertices


def insert_bit(
    mask: int,
    pos: int,
    bit: int,
) -> int:
    """
    Insert `bit` (0/1) at position `pos` (LSB == position 0) in `mask`
    shifting higher bits left by one.
    """
    lower = mask & ((1 << pos) - 1)
    higher = mask >> pos
    return lower | (bit << pos) | (higher << (pos + 1))


def remove_bit(
    mask: int,
    pos: int,
) -> int:
    """
    Delete the bit at position `pos` from `mask`, shifting higher bits right.
    """
    lower = mask & ((1 << pos) - 1)
    higher = mask >> (pos + 1)
    return lower | (higher << pos)


def bag_tuple(bag: Bag) -> tuple[int, ...]:
    return tuple(sorted(bag))


def bag_selected_weight(
    assign_mask: int,
    bag_vertices: tuple[int, ...],
    vertex_weights: dict[int, int],
) -> int:
    """Sum of weights of vertices in the bag that are selected (IN)."""
    s = 0
    for idx, v in enumerate(bag_vertices):
        if (assign_mask >> idx) & 1:
            s += vertex_weights[v]
    return s % MOD


def accumulate(
    table: dict[DPState, DPValue],
    state: DPState,
    cnt: int,
    wsum: int,
) -> None:
    """Add (cnt, wsum) to existing entry of state inside table (MOD arithmetic)."""
    cnt %= MOD
    wsum %= MOD
    if state in table:
        existing = table[state]
        table[state] = DPValue(count=(existing.count + cnt) % MOD, weight=(existing.weight + wsum) % MOD)
    else:
        table[state] = DPValue(count=cnt, weight=wsum)


def leaf_callback(
    graph: Graph,
    cur_table: dict[DPState, DPValue],
    cur_bag_info: tuple[int, Bag],
    leaf_vertex: int,
):
    bag_vertices = bag_tuple(cur_bag_info[1])  # (leaf_vertex,)
    assert len(bag_vertices) == 1 and bag_vertices[0] == leaf_vertex

    w_v = graph.vertex_weights[leaf_vertex] if graph.vertex_weights else 1

    # Case 1: vertex is IN the dominating set
    assign_mask = 1  # bit 0 == 1
    need_mask = 0  # IN vertices never need domination
    state_in = DPState(assign_mask=assign_mask, need_mask=need_mask)
    accumulate(cur_table, state_in, cnt=1, wsum=w_v % MOD)

    # Case 2: vertex is OUT - needs domination
    assign_mask = 0
    need_mask = 1  # NEEDS_DOMINATION
    state_out = DPState(assign_mask=assign_mask, need_mask=need_mask)
    accumulate(cur_table, state_out, cnt=1, wsum=0)


def introduce_callback(
    graph: Graph,
    cur_table: dict[DPState, DPValue],
    cur_bag_info: tuple[int, Bag],
    child_table: dict[DPState, DPValue],
    child_bag_info: tuple[int, Bag],
    introduced_vertex: int,
):
    parent_vertices = bag_tuple(cur_bag_info[1])

    # index at which the new vertex was inserted
    idx_new = parent_vertices.index(introduced_vertex)
    w_new = graph.vertex_weights[introduced_vertex] if graph.vertex_weights else 1

    # pre-compute adjacency between introduced vertex and vertices in parent bag
    is_adj = [(v in graph.neighbors(introduced_vertex)) for v in parent_vertices]

    for child_state, dp_value in child_table.items():
        child_assign, child_need = child_state.assign_mask, child_state.need_mask
        cnt_child, wsum_child = dp_value.count, dp_value.weight
        # ────────────────────────────────────
        # Choice A: new vertex is IN_X
        # ────────────────────────────────────
        assign_in = insert_bit(child_assign, idx_new, 1)
        need_in = insert_bit(child_need, idx_new, 0)

        # when y is IN it may dominate some previously undominated OUT vertices
        for idx, adj in enumerate(is_adj):
            if idx == idx_new or not adj:
                continue
            # vertex idx is OUT?
            if (assign_in >> idx) & 1:
                continue  # IN vertices never carry NEED flag
            # if it was NEED, clear it
            if (need_in >> idx) & 1:
                need_in &= ~(1 << idx)

        cnt_new = cnt_child
        wsum_new = (wsum_child + cnt_child * w_new) % MOD
        state_in = DPState(assign_mask=assign_in, need_mask=need_in)
        accumulate(cur_table, state_in, cnt_new, wsum_new)

        # ────────────────────────────────────
        # Choice B: new vertex is NOT_IN_X
        # ────────────────────────────────────
        assign_out = insert_bit(child_assign, idx_new, 0)

        # Determine if introduced vertex is already dominated by some
        # IN vertex present in the (extended) bag.
        dominated = False
        for idx, adj in enumerate(is_adj):
            if idx == idx_new or not adj:
                continue
            if (assign_out >> idx) & 1:  # neighbor is IN
                dominated = True
                break
        need_bit = 0 if dominated else 1
        need_out = insert_bit(child_need, idx_new, need_bit)

        # ( no other vertices change status )
        state_out = DPState(assign_mask=assign_out, need_mask=need_out)
        accumulate(cur_table, state_out, cnt_child, wsum_child)


def forget_callback(
    graph: Graph,
    cur_table: dict[DPState, DPValue],
    cur_bag_info: tuple[int, Bag],
    child_table: dict[DPState, DPValue],
    child_bag_info: tuple[int, Bag],
    forgotten_vertex: int,
):
    child_vertices = bag_tuple(child_bag_info[1])
    idx_forgot = child_vertices.index(forgotten_vertex)

    for child_state, dp_value in child_table.items():
        assign_child, need_child = child_state.assign_mask, child_state.need_mask
        cnt_child, wsum_child = dp_value.count, dp_value.weight
        bit_assign = (assign_child >> idx_forgot) & 1
        bit_need = (need_child >> idx_forgot) & 1

        # If forgotten vertex is OUT and still needs domination -> invalid state
        if bit_assign == 0 and bit_need == 1:
            continue

        assign_par = remove_bit(assign_child, idx_forgot)
        need_par = remove_bit(need_child, idx_forgot)

        state_par = DPState(assign_mask=assign_par, need_mask=need_par)
        accumulate(cur_table, state_par, cnt_child, wsum_child)


def join_callback(
    graph: Graph,
    cur_table: dict[DPState, DPValue],
    cur_bag_info: tuple[int, Bag],
    left_child_table: dict[DPState, DPValue],
    left_child_bag_info: tuple[int, Bag],
    right_child_table: dict[DPState, DPValue],
    right_child_bag_info: tuple[int, Bag],
):
    bag_vertices = bag_tuple(cur_bag_info[1])
    vertex_weights = graph.vertex_weights
    assert vertex_weights is not None

    # Group right states by assignment mask for O(|L| + |R|) compatibility
    right_by_assign: dict[int, list[tuple[int, int, int]]] = {}
    for right_state, dp_value in right_child_table.items():
        assign_r, need_r = right_state.assign_mask, right_state.need_mask
        cnt_r, wsum_r = dp_value.count, dp_value.weight
        right_by_assign.setdefault(assign_r, []).append((need_r, cnt_r, wsum_r))

    for left_state, dp_value in left_child_table.items():
        assign_l, need_l = left_state.assign_mask, left_state.need_mask
        cnt_l, wsum_l = dp_value.count, dp_value.weight
        if assign_l not in right_by_assign:
            continue
        for need_r, cnt_r, wsum_r in right_by_assign[assign_l]:
            # Merge NEED flags: dominated if dominated in either side
            need_merge = need_l & need_r  # bitwise AND keeps 1 only if both have NEED

            cnt_merge = (cnt_l * cnt_r) % MOD

            w_bag_sel = bag_selected_weight(assign_l, bag_vertices, vertex_weights)
            w_merge = (wsum_l * cnt_r + wsum_r * cnt_l - cnt_merge * w_bag_sel) % MOD
            if w_merge < 0:
                w_merge += MOD

            state_merge = DPState(assign_mask=assign_l, need_mask=need_merge)
            accumulate(cur_table, state_merge, cnt_merge, w_merge)


def extract_solution(root_table: dict[DPState, DPValue]) -> int:
    """
    Sum the total weights of all globally valid dominating sets.
    Return -1 if none exist.
    """
    answer = 0
    found = False
    for state, dp_value in root_table.items():
        assign_mask, need_mask = state.assign_mask, state.need_mask
        cnt, wsum = dp_value.count, dp_value.weight
        # Bag may be empty or not
        if assign_mask == 0 and need_mask == 0 and cnt == 0:
            # Defensive - shouldn't happen
            continue
        if need_mask != 0:
            # some vertex in root bag still needs domination -> invalid
            continue
        answer = (answer + wsum) % MOD
        found = True
    return answer if found else -1
