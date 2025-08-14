# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from evaluation.graph_structure import Bag, Graph


@dataclass(frozen=True)
class DPState:
    pass


@dataclass
class DPValue:
    pass


def leaf_callback(
    graph: Graph,
    cur_table: dict[DPState, int],
    cur_bag_info: tuple[int, Bag],
    leaf_vertex: int,
):
    """
    Handles a leaf node in the nice tree decomposition.

    At a leaf node t with bag B_t = {leaf_vertex}, we initialize the DP table with all valid
    base states involving only the single vertex `leaf_vertex`. These states represent trivial
    partial solutions X over a single vertex and set the foundation for the DP recursion.

    Args:
        graph: The input graph, induced over the vertices in the bag.
        cur_table: The DP table for the current node t to be populated - DP[t].
        cur_bag_info: Tuple (t, B_t), where B_t is the bag at node t (B_t = {leaf_vertex}), and t is the bag index.
        leaf_vertex: The single vertex in the leaf bag.
    """
    raise NotImplementedError()


def introduce_callback(
    graph: Graph,
    cur_table: dict[DPState, int],
    cur_bag_info: tuple[int, Bag],
    child_table: dict[DPState, int],
    child_bag_info: tuple[int, Bag],
    introduced_vertex: int,
):
    """
    Handles an introduce node in the nice tree decomposition.

    In an introduce node t with child t', we have B_t = B_{t'} \\cup {introduced_vertex}.
    The vertex `introduced_vertex` is being introduced and appears for the first time here.
    We must update the DP table by extending each child state from t' to include `introduced_vertex`
    in all valid ways. This typically involves checking how `introduced_vertex` connects to the rest
    of the bag and adjusting state properties.

    If multiple configurations in the child DP table DP[t'] project to the same configuration in in the parent
    DP table DP[t], aggregate their results by taking min/max according to the problem.
    Store the resulting configurations and their optimal weights in DP[t].

    Args:
        graph: The input graph, induced over the vertices in the bag.
        cur_table: The DP table for the current node t to be populated - DP[t']
        cur_bag_info: Tuple (t, B_t), where B_t = B_{t'} \\cup {introduced_vertex}, and t is the bag index.
        child_table: The DP table from the child node t' - DP[t'].
        child_bag_info: Tuple (t', B_{t'}), the child node and its bag.
        introduced_vertex: The vertex being introduced.
    """
    raise NotImplementedError()


def forget_callback(
    graph: Graph,
    cur_table: dict[DPState, int],
    cur_bag_info: tuple[int, Bag],
    child_table: dict[DPState, int],
    child_bag_info: tuple[int, Bag],
    forgotten_vertex: int,
):
    """
    Handles a forget node in the nice tree decomposition.

    In a forget node t with child t', we have B_t = B_{t'} \\ {forgotten_vertex}.
    The vertex `forgotten_vertex` is being forgotten and will not appear in any future bag.
    For each state in DP[t'], remove `forgotten_vertex` to get a projected state over B_t.

    If multiple child states project to the same state in DP[t], aggregate their values
    (e.g., using min or max depending on the problem).
    Store the resulting configurations and their optimal weights in DP[t].

    Args:
        graph: The input graph, induced over the vertices in B_t \\cup {forgotten_vertex}.
        cur_table: The DP table for the current node t to be populated - DP[t].
        cur_bag_info: Tuple (t, B_t), where B_t = B_{t'} \\ {forgotten_vertex}.
        child_table: The DP table from the child node t' - DP[t'].
        child_bag_info: Tuple (t', B_{t'}), the child node and its bag.
        forgotten_vertex: The vertex being forgotten.
    """
    raise NotImplementedError()


def join_callback(
    graph: Graph,
    cur_table: dict[DPState, int],
    cur_bag_info: tuple[int, Bag],
    left_child_table: dict[DPState, int],
    left_child_bag_info: tuple[int, Bag],
    right_child_table: dict[DPState, int],
    right_child_bag_info: tuple[int, Bag],
):
    """
    Handles a join node in the nice tree decomposition.

    A join node t has two children t_1 and t_2 with the same bag: B_t = B_{t_1} = B_{t_2}.
    The subtree rooted at t represents the union of solutions from both children.
    For every pair of compatible states from DP[t_1] and DP[t_2], merge them into a valid state over B_t.

    If multiple such merges result in the same final state, aggregate their values
    (e.g., using min or max depending on the problem).
    Store the resulting configurations and their optimal weights in DP[t].

    Args:
        graph: The input graph, induced over the vertices in the bag.
        cur_table: The DP table for the current node t to be populated - DP[t].
        cur_bag_info: Tuple (t, B_t), the current node and its bag.
        left_child_table: The DP table from the left child t_1 - DP[t_1].
        left_child_bag_info: Tuple (t_1, B_{t_1}), the left child and its bag.
        right_child_table: The DP table from the right child t_2 - DP[t_2].
        right_child_bag_info: Tuple (t_2, B_{t_2}), the right child and its bag.
    """
    raise NotImplementedError()


def extract_solution(root_table: dict[DPState, int]) -> int:
    raise NotImplementedError()
