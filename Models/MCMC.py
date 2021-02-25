import numpy as np
from Data.data_utils import new_node_by_val, TreeNode
from Data.reward import scores
from Data.random_generator import random_expression


def replace_op(tree: TreeNode, vocab) -> TreeNode:
    """random select an operation/func node, replace it with a random operation
       generate additional operands if needed
       trim extra operands if needed """
    ops = list()
    idx = list()
    q = [(-1, tree)]
    p = 0
    while p < len(q):
        cid, root = q[p]
        if vocab.is_op(root.get_value()):
            ops.append(root)
            idx.append(cid)
        for cid, c in enumerate(root.iter_child()):
            q.append((cid, c))

        p += 1
    select_id = np.random.choice(range(len(idx)))
    op = ops[select_id]
    cid = idx[select_id]

    new_op = op.get_value()
    while op.get_value() == new_op:
        new_op = vocab.random_func()

    new_op = new_node_by_val(vocab, new_op)
    for c in op.iter_child():
        new_op.add_child(c)
        if new_op.get_child_num() == vocab.num_operands(new_op.get_value()):
            break
    while new_op.get_child_num() < vocab.num_operands(new_op.get_value()):
        new_op.add_child(random_expression(vocab, 2, None))

    parent = op.get_parent()
    if parent is not None:
        op.get_parent().set_child(cid, new_op)
    else:
        tree = new_op
    return tree


def replace_var(tree: TreeNode, vocab) -> TreeNode:
    """random select an var node, replace it with a random different var"""
    vars = list()
    idx = list()
    q = [(-1, tree)]
    p = 0
    while p < len(q):
        cid, root = q[p]
        if vocab.is_const_or_var(root.get_value()):
            vars.append(root)
            idx.append(cid)
        for cid, c in enumerate(root.iter_child()):
            q.append((cid, c))

        p += 1
    select_id = np.random.choice(range(len(idx)))
    v = vars[select_id]
    cid = idx[select_id]

    new_op = v.get_value()
    while v.get_value() == new_op:
        new_op = vocab.random_const_and_var()

    new_op = new_node_by_val(vocab, new_op)

    parent = v.get_parent()
    if parent is not None:
        v.get_parent().set_child(cid, new_op)
    else:
        tree = new_op
    return tree


def reduce(tree: TreeNode, vocab) -> TreeNode:
    """random select an operation/func node, replace it with a random variable/const"""
    ops = list()
    idx = list()
    q = [(-1, tree)]
    p = 0
    while p < len(q):
        cid, root = q[p]
        if vocab.is_op(root.get_value()):
            ops.append(root)
            idx.append(cid)
        for cid, c in enumerate(root.iter_child()):
            q.append((cid, c))

        p += 1
    select_id = np.random.choice(range(len(idx)))
    op = ops[select_id]
    cid = idx[select_id]

    new_op = vocab.random_const_and_var()
    new_op = new_node_by_val(vocab, new_op)

    parent = op.get_parent()
    if parent is not None:
        op.get_parent().set_child(cid, new_op)
    else:
        tree = new_op
    return tree


def expand(tree: TreeNode, vocab) -> TreeNode:
    """random select an variable node, replace it with a random depth=2 expression"""
    vars = list()
    idx = list()
    q = [(-1, tree)]
    p = 0
    while p < len(q):
        cid, root = q[p]
        if vocab.is_const_or_var(root.get_value()):
            vars.append(root)
            idx.append(cid)
        for cid, c in enumerate(root.iter_child()):
            q.append((cid, c))
        p += 1

    select_id = np.random.choice(range(len(idx)))
    v = vars[select_id]
    cid = idx[select_id]
    new_op = vocab.random_func()
    new_op = new_node_by_val(vocab, new_op)

    while new_op.get_child_num() < vocab.num_operands(new_op.get_value()):
        new_op.add_child(random_expression(vocab, 1, None))

    parent = v.get_parent()
    if parent is not None:
        v.get_parent().set_child(cid, new_op)
    else:
        tree = new_op
    return tree


def MCMC(expression: TreeNode, vocab, max_iter, performance_weight=1, beta=1.0):
    trials = 0
    reward = 1.0   # 1.0 (equal) + 0.0 (no performance gain)
    state = expression
    input_ref = expression.to_tokens()
    best_equal = None
    while trials < max_iter:
        if state.depth() > 1:
            transform = np.random.choice([replace_op, replace_var, reduce, expand],
                                         p=[0.2, 0.3, 0.4, 0.1])
        else:
            transform = np.random.choice([replace_var, expand], p=[0.8, 0.2])
        tree_star = transform(state.clone(), vocab)
        s = scores(tree_star.to_tokens(), input_ref, vocab, True)
        r_star = s[2] + performance_weight * s[0] + 1e-7
        accept_prob = np.exp(- beta * reward / r_star)
        if s[2] == 1:
            if best_equal is None:
                best_equal = tree_star
            elif best_equal.size() > tree_star.size():
                best_equal = tree_star
            if best_equal.size() == 1:
                # print("Early stop", best_equal)
                break
        # print(tree_star, expression, s, accept_prob)
        if accept_prob >= 1 or accept_prob > np.random.random():
            state = tree_star
            reward = r_star
        trials += 1
    output_ref = state.to_tokens() if best_equal is None else best_equal.to_tokens()
    if len(output_ref) == len(input_ref):
        if all(i == o for i, o in zip(input_ref, output_ref)):
            output_ref = []
    return output_ref, scores(output_ref, input_ref, vocab, True)


if __name__ == "__main__":
    from Data.data_utils import parse_expression
    from Models.train_utils import get_halide_vocab
    vocab = get_halide_vocab()

    # tree = parse_expression(['y', '&&', '(', '(', 'z', '||', 'y', ')', '&&', 'x', ')'], vocab)

    tree = parse_expression(['(', 'y', '||', 'True', ')', '&&', 'x'], vocab)
    output, score = MCMC(tree, vocab, 5000, 0.1, 1)
