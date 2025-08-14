# SPDX-License-Identifier: Apache-2.0

from queue import Queue
from typing import Callable

LEAF = 0
INTRODUCE = 1
FORGET = 2
JOIN = 3

NO_VERTEX = -1


class TreeDecomposition:
    """
    @brief A class representing a tree decomposition of an undirected graph.

    Each bag is a set of vertices. Bags are connected to form a tree such that:
    - Every vertex of the graph appears in at least one bag
    - Every edge is fully contained in at least one bag
    - For any vertex, the bags containing it form a connected subtree
    """

    # these fields are always populated at instantiation time:
    b: list[set[int]]  # b[i] = {j, k, l} iff vertices v_j, v_k, v_l are in bag i (and no others)
    e2b: list[tuple[int, int]]  # e2bs[i] = (j, k), means edge i is connecting bag j and bag k (j<k)

    # optional, automatically-populated fields:
    b2b: list[set[int],]
    # b2b[i] = {j, k, l} iff bags j,k,l are the neighbors of bag i
    # (consequently - i is also in b2b[k], b2b[j], b2b[l])

    def __init__(self, b: list[set[int]], e2b: list[tuple[int, int]]) -> None:
        """
        @brief Construct a TreeDecomposition object.
        @param b A list of bags, each a set of vertex indices.
        @param e2b A list of edges connecting bags (as tuples of bag indices).
        """
        assert len(b) > 0, "There must be at least one bag"
        self.width = max([len(bag) for bag in b]) - 1
        self.b = b[:]
        self.e2b = e2b[:]
        self.populate_b2b()

    @property
    def num_bags(self) -> int:
        """
        @brief Get the number of bags in the tree decomposition.
        @return Number of bags.
        """
        return len(self.b)

    @property
    def num_edges(self) -> int:
        """
        @brief Get the number of edges in the tree decomposition.
        @return Number of edges.
        """
        return len(self.e2b)

    def populate_b2b(self) -> None:
        """
        @brief Populate the bag-to-bag adjacency structure.
        """
        self.b2b = [set() for _ in range(len(self.b))]
        for edge in self.e2b:
            self.b2b[edge[0]].add(edge[1])
            self.b2b[edge[1]].add(edge[0])


class NiceTreeDecomposition(TreeDecomposition):
    """
    @brief A class representing a nice tree decomposition of an undirected graph.

    A nice tree decomposition satisfies:
    - The tree is binary
    - Each node is one of four types:
        - LEAF: a singleton bag
        - INTRODUCE: adds one vertex compared to its child
        - FORGET: removes one vertex compared to its child
        - JOIN: has two children with the same bag

    In this implementation, the root is always bag 0.
    """

    # A list of bag types, where each is: (type_id, vertex), for example (FORGET, 3) means
    # that the bag is a FORGET node, forgetting vertex 3.
    b2type: list[tuple[int, int]]

    # A flat list of the children of a bag, as Nice Tree Decomposition is binary, so the
    # children of bag i are at 2i and 2i + 1.
    # 1. b2c[2i] = left child of i (or -1 if no child).
    # 2. b2c[2i + 1] = right child of i (or -1 if no child).
    b2c: list[int]

    # A list mapping each bag to its parent in the tree.
    b2p: list[int]

    def __init__(self, td: TreeDecomposition) -> None:
        """
        @brief Construct a NiceTreeDecomposition from a given TreeDecomposition.
        @param td An existing tree decomposition to convert.
        """
        self.b2type: list[tuple[int, int]] = []
        super().__init__(td.b, td.e2b)
        self._make_nice()
        assert self.width == td.width, "Width of the nice decomposition must be equal to the original decomposition"

    def __str__(self):
        """
        @brief String representation of the NiceTreeDecomposition.
        @return A string describing the decomposition.
        """
        return f"{self.__class__.__name__}(bags={self.b}, parents={self.b2p}, bag_types={self.b2type})"

    def _populate_b2b_from_b2p(self) -> None:
        self.b2b = [set() for _ in range(len(self.b))]
        for c, p in enumerate(self.b2p):
            if p != -1:
                self.b2b[c].add(p)
                self.b2b[p].add(c)

    def _populate_e2b_from_b2p(self) -> None:
        self.e2b = []
        for a, b in enumerate(self.b2p):
            if b != -1:
                self.e2b.append((min(a, b), max(a, b)))

    def _make_binary(self) -> None:
        """
        @brief Convert the bag tree into a binary tree by splitting nodes with more than two children,
        also populating b2c and b2p
        """
        self.b2c = [-1 for _ in range(2 * len(self.b2b))]
        self.b2p = [-1 for _ in range(len(self.b2b))]
        q = Queue()
        q.put((0, -1))
        while not q.empty():
            v, parent = q.get()
            for child in self.b2b[v]:
                if child != parent:
                    self.b2p[child] = v
                    q.put((child, v))

        q = Queue()
        q.put(0)
        while not q.empty():
            v = q.get()
            p = self.b2p[v]
            is_not_root = p != -1
            child_count = len(self.b2b[v]) - is_not_root  # exclude parent, except for root
            if child_count <= 2:
                i = 0
                for n in self.b2b[v]:
                    if n != p:
                        self.b2c[v * 2 + i] = n
                        i += 1
                        q.put(n)
            else:
                # A binary tree with k leaves has k-1 internal nodes *including the root*
                new_node_count = child_count - 2
                next_new = len(self.b)
                self.b2b.extend([set() for _ in range(new_node_count)])
                self.b2c.extend([-1 for _ in range(2 * new_node_count)])
                self.b2p.extend([-1 for _ in range(new_node_count)])
                self.b.extend(
                    [self.b[v] for _ in range(new_node_count)]
                )  # Intentional reference to save memory on immutable objects
                # Construct the subtree bottom-up
                old_children = [c for c in self.b2b[v] if c != p]
                sub_queue = Queue()
                for c in old_children:
                    sub_queue.put(c)
                    q.put(c)
                while not sub_queue.empty():
                    left_child = sub_queue.get()
                    right_child = sub_queue.get()
                    new_node = next_new if not sub_queue.empty() else v
                    next_new += 1
                    self.b2c[2 * new_node] = left_child
                    self.b2c[2 * new_node + 1] = right_child
                    self.b2b[new_node] = {left_child, right_child}
                    if new_node == v:
                        if self.b2p[v] != -1:
                            self.b2b[new_node].add(self.b2p[v])
                    else:
                        sub_queue.put(new_node)
                    for c in [left_child, right_child]:
                        self.b2p[c] = new_node
                        self.b2b[c].discard(v)
                        self.b2b[c].add(new_node)

    def _add_introduce_tail(self, b_id: int) -> None:
        """
        @brief Add a tail of INTRODUCE nodes for a leaf bag with more than one vertex.
        @param b_id The ID of the bag to process.
        """
        previous_bag_id = b_id
        bag_list = list(self.b[b_id])
        bag_id = len(self.b)
        tail_length = len(bag_list) - 1
        self.b.extend([set() for _ in range(tail_length)])
        self.b2p.extend([-1 for _ in range(tail_length)])
        self.b2c.extend([-1 for _ in range(2 * tail_length)])
        self.b2type.extend([() for _ in range(tail_length)])
        for i in range(tail_length):
            new_bag_id = bag_id
            bag_id += 1
            self.b[new_bag_id] = set(bag_list[i + 1 :])
            self.b2p[new_bag_id] = previous_bag_id
            self.b2c[2 * previous_bag_id] = new_bag_id
            if i == tail_length - 1:
                self.b2type[new_bag_id] = (LEAF, NO_VERTEX)
            self.b2type[previous_bag_id] = (INTRODUCE, bag_list[i])
            previous_bag_id = new_bag_id

    def _split_join_node(self, cur: int, left_child: int, right_child: int) -> tuple[int, int]:
        """
        @brief Splits a JOIN node with non-identical children into a valid binary structure.
               The splitting logic is to introduce extra copies of the parent bag for every child that is different,
               where the copy takes the place of the child and becomes the new parent of the replaced child.

        @param cur The join node.
        @param left_child The left child index.
        @param right_child The right child index.
        @return Tuple of new left and right child indices.
        """
        out_nodes = [left_child, right_child]
        for i in range(2):
            if self.b[cur] != self.b[out_nodes[i]]:
                new_node = len(self.b)
                self.b.append(self.b[cur])
                self.b2p.append(cur)
                self.b2c.extend([out_nodes[i], -1])
                self.b2type.append(())  # Set an empty type for the new node
                self.b2c[2 * cur + i] = new_node
                self.b2p[out_nodes[i]] = new_node
                out_nodes[i] = new_node
        assert out_nodes[0] != left_child or out_nodes[1] != right_child, "No new nodes were created"
        return out_nodes[0], out_nodes[1]

    def _add_introduce_forget_path(self, from_bag_id: int, to_bag_id: int) -> None:
        """
        @brief Add intermediate INTRODUCE and FORGET nodes between two bags with differing vertex sets.
        @param from_bag_id The parent bag index.
        @param to_bag_id The child bag index.
        """
        to_forget = self.b[to_bag_id].difference(self.b[from_bag_id])
        to_introduce = self.b[from_bag_id].difference(self.b[to_bag_id])
        if len(to_forget) + len(to_introduce) < 2:
            # symmetrical difference in nice tree must be <= 1
            assert len(to_forget) + len(to_introduce) == 1
            # simple case
            if len(to_forget) == 1:
                self.b2type[from_bag_id] = (FORGET, next(iter(to_forget)))
            else:
                self.b2type[from_bag_id] = (INTRODUCE, next(iter(to_introduce)))
        else:
            # node needs to be expanded to a path of introduce and/or forget nodes
            prev_bag_id = to_bag_id
            intersection = self.b[to_bag_id].intersection(self.b[from_bag_id])
            # iterating over this list, adjusting the delimiters is more efficient
            # than using a sequence of set operations
            set_list = list(to_forget) + list(intersection) + list(to_introduce)
            s = 0
            left_delimiter = len(to_forget)
            e = len(to_forget) + len(intersection)
            action = FORGET if s < left_delimiter else INTRODUCE
            new_bag_id = from_bag_id
            while e < len(set_list) or action == FORGET:
                if action == FORGET:
                    s += 1
                    v = set_list[s - 1]
                else:
                    e += 1
                    v = set_list[e - 1]
                new_bag_set = set(set_list[s:e])
                # replace these set ifs with something better:
                if new_bag_set == self.b[from_bag_id]:
                    new_bag_id = from_bag_id
                elif new_bag_set == self.b[to_bag_id]:
                    # TODO(Shir): replace with check of e and s
                    continue
                else:
                    new_bag_id = len(self.b)
                    self.b.append(new_bag_set)
                    self.b2p.append(-1)
                    self.b2c.extend([-1, -1])
                    self.b2type.append(())
                self.b2type[new_bag_id] = (action, v)
                if s == left_delimiter:
                    action = INTRODUCE
                self.b2p[prev_bag_id] = new_bag_id
                self.b2c[2 * new_bag_id] = prev_bag_id
                prev_bag_id = new_bag_id

    def _convert_to_typed_bags(self) -> None:
        """
        @brief Converts the current tree into a nice form with bag types assigned to every node.
        """
        stack = []
        stack.append(0)
        self.b2type = [() for _ in range(len(self.b))]
        assert len(self.b2c) == len(self.b) * 2, "b2c must be populated correctly"
        while stack:
            b_id = stack.pop()
            l = self.b2c[2 * b_id]
            r = self.b2c[2 * b_id + 1]
            children_count = (l != -1) + (r != -1)
            if children_count == 0:
                if len(self.b[b_id]) == 1:
                    self.b2type[b_id] = (LEAF, -1)
                else:
                    self._add_introduce_tail(b_id)
            elif children_count == 2:
                self.b2type[b_id] = (JOIN, -1)
                if self.b[l] == self.b[r] == self.b[b_id]:
                    stack.extend([l, r])
                    continue
                new_left_child, new_right_child = self._split_join_node(cur=b_id, left_child=l, right_child=r)
                stack.extend([new_left_child, new_right_child])
            else:
                assert children_count == 1, "There should exactly 1 unvisited neighbor"
                child = l if l != -1 else r
                assert self.b[child] != self.b[b_id], "Child bag must differ from parent bag"
                self._add_introduce_forget_path(from_bag_id=b_id, to_bag_id=child)
                stack.append(child)

    def _make_nice(self):
        """
        @brief Make the current decomposition into a nice tree decomposition.
        """
        self._make_binary()
        self._convert_to_typed_bags()
        self._populate_b2b_from_b2p()

    def traverse(
        self,
        callback: Callable[[tuple[int, set[int]], list[tuple[int, set[int]]]], None],
    ):
        """
        @brief Traverse the tree decomposition in depth-first post-order.
        @param callback Function called with (cur_bag=(i, bag), children=[(child_i, child_bag), ...])
        """
        assert 2 * len(self.b) == len(self.b2c)
        stack = [(0, False)]  # bag_id, are all children visited
        while stack:
            b_index, visited_children = stack.pop()
            if visited_children:
                children_refs = []
                for i in range(2):
                    child_i = self.b2c[2 * b_index + i]
                    if child_i != -1:
                        children_refs.append((child_i, self.b[child_i]))
                callback((b_index, self.b[b_index]), sorted(children_refs))
            else:
                stack.append((b_index, True))
                if self.b2c[2 * b_index] != -1:
                    stack.append((self.b2c[2 * b_index], False))
                if self.b2c[2 * b_index + 1] != -1:
                    stack.append((self.b2c[2 * b_index + 1], False))
