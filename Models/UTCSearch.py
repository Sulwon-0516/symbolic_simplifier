import numpy as np
from Data.data_utils import new_node_by_val
from Data.reward import scores


class MCTNode(object):
    def __init__(self, state):
        self.state = state
        self.children = list()
        self.parent = None
        self.N = 0.0
        self.Q = 0.0
        self.meta_info = dict()

    def score(self, c):
        """
        :param c: const ( C * sqrt(2 * ln(N_parent))
        :return: UCT function
        """
        if self.N == 0:
            return np.inf
        return self.Q / self.N + c / np.sqrt(self.N)

    def get_tried_actions(self):
        res = list()
        for c in self.children:
            if isinstance(c.state, list):
                res.append(c.state[-1])  # last char as action
            else:
                res.append(c.state)
        return res

    def add_child(self, v):
        self.children.append(v)
        v.parent = self

    def is_fully_expanded(self, action_space):
        return len(self.children) >= len(action_space)

    def best_child(self, c):
        c_prime = c * np.sqrt(2 * np.log(self.N))
        return max(self.children, key=lambda x: x.score(c_prime))

    def __repr__(self):
        return "<MCTNode: {}, N={}, Q={}, C={}>".format(self.state, self.N, self.Q, len(self.children))


def SequentialUCTSearch(action_space, s0, c, input_seq, vocab, max_len, root=None, max_iter=100):
    """
    :param action_space: list of actions can be taken at s0
    :param s0: current state, a list of actions that have been taken
    :param c: constant for exploit-explore tradeoff
    :param input_seq: for tester
    :param vocab: for testing
    :param max_len: max length of sequence, i.g. length of s0
    :param root: MCTS node
    :param max_iter: computation budget
    :return: action, next state as MCTS node
    """
    action_list = list(action_space)
    if root is None:
        root = MCTNode(s0)
    for _ in range(max_iter):
        v = root
        # Tree policy
        while not v.state[-1] == vocab.end():  # terminal node
            if v.is_fully_expanded(action_space):
                v = v.best_child(c)
            else:
                # Expand
                tried_actions = v.get_tried_actions()
                untried_actions = action_space.difference(tried_actions)
                a = np.random.choice(list(untried_actions)).tolist()
                new_v = MCTNode(v.state + [a])  # str concatenate
                v.add_child(new_v)
                v = new_v
                break

        # roll out
        state = v.state
        while not state[-1] == vocab.end() and len(state) < max_len:
            a = np.random.choice(action_list).tolist()
            state = state + [a]
        delta = scores(state[1:], input_seq, vocab)[2]    # skip <s> token in state
        # if delta != 0:
        #     print("correct exp found:", state)

        # back up
        while v is not None:
            v.N += 1
            v.Q += delta
            v = v.parent

    bc = root.best_child(0)
    action = bc.state[-1]
    bc.parent = None   # trimming from root
    return action, bc


def SequentialMCTS(input_seq, vocab, c, max_len, max_iter):
    action_space = set(vocab.word2id.keys())
    action_space.remove(vocab.id2w(vocab.start()))
    action_space.remove(vocab.id2w(vocab.pad()))
    seq = [vocab.id2w(vocab.start())]
    root = None
    while seq[-1] != vocab.end() and len(seq) < max_len:
        action, root = SequentialUCTSearch(action_space, seq, c, input_seq, vocab, max_len, root, max_iter)
        # print("action", action)
        # print("root", root.N, root.Q)
        seq.append(action)
    res_score = scores(seq[1:], input_seq, vocab)
    return seq, res_score


def TreeUCTSearch(action_space, queue, pointer, c, input_seq, vocab, max_depth, max_iter=100):
    """
    :param action_space: action space at current state
    :param queue: list of actions that have been taken, in BFS manner, each item in queue is
                    [MCTSNode, point_to, num_child_in_queue, current depth]
                    MCTSnode takes the token itself as state
    :param pointer: the idx of current state in the queue
    :param c: constant for exploit-explore tradeoff
    :param input_seq: for testing
    :param vocab: for testing
    :param max_depth: max depth of the expression tree, depth of root node in queue
    :param max_iter: computation budget
    :return: action, next state as MCTS node
    """
    action_list = list(action_space)
    variable_list = vocab.get_const_vars()
    root = queue[pointer][0]
    if root is None:
        root = MCTNode(None)

    for _ in range(max_iter):
        v = root
        p0 = pointer
        # Tree policy

        # deep copy
        tmp_q = [[i[0], i[1], i[2], i[3]] for i in queue]  # copy queue
        # print("initial queue", tmp_q)
        while p0 < len(tmp_q):
            if p0 and vocab.num_operands(tmp_q[p0][0].state) == tmp_q[p0][2]:  # do not have child state
                p0 += 1
                continue

            if tmp_q[p0][3] >= max_depth - 1:
                this_action_space = set(variable_list)
            else:
                this_action_space = action_space

            if v.is_fully_expanded(this_action_space):
                # print("is fully_expanded")
                v = v.best_child(c)
                # print("v best child is", v.state)
                tmp_q.append([v, v.meta_info["point_to"], 0, tmp_q[v.meta_info["point_to"]][3] + 1])
                tmp_q[p0][2] += 1
                if p0 == 0 or tmp_q[p0][2] == vocab.num_operands(tmp_q[p0][0].state):
                    # first element is placeholder, thus equivalent to only has one child
                    p0 += 1
                    # move pointer if all children of current node is enqueued
            else:
                # expand
                tried_actions = v.get_tried_actions()
                untried_actions = this_action_space.difference(tried_actions)
                a = np.random.choice(list(untried_actions)).tolist()
                new_v = MCTNode(a)
                new_v.meta_info["point_to"] = p0
                v.add_child(new_v)
                v = new_v
                tmp_q.append([v, p0, 0, tmp_q[p0][3] + 1])
                tmp_q[p0][2] += 1
                # if p0 == 0 or tmp_q[p0][2] == vocab.num_operands(tmp_q[p0][0].state):
                #     # print("moving pointer forward", tmp_q)
                #     # first element is placeholder, thus equivalent to only has one child
                #     p0 += 1
                #     # move pointer if all children of current node is enqueued
                break
        # print("Tree Policy", v, v.state)
        # print("Queue:", tmp_q)
        # print("p0", p0)

        # roll out
        # print("\nRolling out")
        while p0 < len(tmp_q):
            if p0 == 0 or vocab.num_operands(tmp_q[p0][0].state) == tmp_q[p0][2]:  # do not need children
                # print("skipping and move forward")
                p0 += 1
                continue
            # print("Current p0", p0)
            if tmp_q[p0][3] >= max_depth - 1:
                # print("reached max_depth")
                this_action_space = variable_list
            else:
                this_action_space = action_list
            a = np.random.choice(this_action_space).tolist()
            # print("random choice made:", a)
            # print("Queue is ", tmp_q)
            tmp_q.append([MCTNode(a), p0, 0, tmp_q[p0][3] + 1])
            tmp_q[p0][2] += 1
            # if tmp_q[p0][2] == vocab.num_operands(tmp_q[p0][0].state):
            #     p0 += 1
            # input()
        # print("queue before assembling", tmp_q)
        # assemble a tree
        output_seq = __assemble_tree(tmp_q[1:], vocab)
        # print("output sequence", output_seq, end=" ")
        delta = scores(output_seq, input_seq, vocab)[2]

        # back up
        while v is not None:
            v.N += 1
            v.Q += delta
            v = v.parent
    bc = root.best_child(0)
    action = bc.state
    bc.parent = None   # trimming from root
    return action, bc


def __assemble_tree(queue, vocab):
    head, rear = 0, 1
    root = new_node_by_val(vocab, queue[0][0].state)
    node_queue = [root]
    while rear < len(queue):
        try:
            num_child = vocab.num_operands(queue[head][0].state)
        except:
            print("head", head, queue)
            exit()
        for offset in range(num_child):
            try:
                child_node = new_node_by_val(vocab, queue[rear + offset][0].state)
                node_queue[head].add_child(child_node)
                node_queue.append(child_node)
            except IndexError:
                print(queue)
                exit()
        head += 1
        rear += num_child
    return root.to_tokens()


def TreeMCTS(input_seq, vocab, c, max_depth, max_iter):
    action_space = set(vocab.word2id.keys())
    for token in [vocab.id2w(vocab.start()), vocab.id2w(vocab.end()), vocab.id2w(vocab.pad()), "(", ")", ","]:
        action_space.remove(token)

    queue = [[None, -1, 0, 0]]
    # TreeNode, MCTS Node, point_to, number of children enqueued, current depth
    pointer = 0

    # bfs
    while pointer < len(queue):
        if pointer != 0 and vocab.is_const_or_var(queue[pointer][0].state):
            # variable do not have subtrees
            pointer += 1
            continue

        action, root = TreeUCTSearch(action_space, queue, pointer, c, input_seq, vocab, max_depth, max_iter)
        queue.append([root, pointer, 0, queue[pointer][3] + 1])
        queue[pointer][2] += 1
        if pointer == 0 or queue[pointer][2] == vocab.num_operands(queue[pointer][0].state):
            pointer += 1

    output_seq = __assemble_tree(queue[1:], vocab)
    res_score = scores(output_seq, input_seq, vocab)
    return output_seq, res_score


if __name__ == "__main__":
    from Models.train_utils import get_halide_vocab

    # exp = ["max", "(", "x", ",", "y", ")"]
    # exp = ["1", "+", "x"]
    exp = ['max', '(', '0', ',', 'x', ')']
    vocab = get_halide_vocab()
    res = TreeMCTS(exp, vocab, 1, max_depth=3, max_iter=20)[0]
    print(res)
