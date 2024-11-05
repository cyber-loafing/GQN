import numpy as np
import random
from numba import jit

import networkx as nx
import random

from typing import List


def indicator(S, n):
    x = np.zeros(n)
    x[list(S)] = 1
    return x


def sample_live_icm(g: nx.Graph, num_graphs: int) -> List[nx.Graph]:
    """
    Samples num_graphs live edge graphs from the input graph g.
    Each edge is included in the live edge graph with probability given by the edge's 'p' attribute.
    # 从给定的图 g 中采样出多个“活跃边图”（live edge graphs）
    :param g: input graph
    :param num_graphs: number of live edge graphs to sample
    :return: list of sampled live edge graphs
    """
    live_edge_graphs = []
    for _ in range(num_graphs):
        h = nx.Graph()
        h.add_nodes_from(g.nodes())
        for u, v in g.edges():
            if random.random() < g[u][v]['p']:
                h.add_edge(u, v)
        live_edge_graphs.append(h)
    return live_edge_graphs


def f_all_influmax_multlinear(x: np.ndarray, Gs: List[np.ndarray], Ps: List[np.ndarray], ws: List[np.ndarray]) -> float:
    """
    :param x: continuous decision variables
    :param Gs:
    :param Ps:
    :param ws:
    活跃边影响最大化问题的多线性扩展的目标函数。

    x：连续决策变量

    Gs/Ps/ws：将影响最大化问题表示为对采样的概率覆盖函数集合的期望。
    """
    n = len(Gs) # number of samples(就是采样数量)
    sample_weights = 1. / n * np.ones(n)
    return objective_live_edge(x, Gs, Ps, ws, sample_weights)


def make_multilinear_objective_samples(live_graphs: List[nx.Graph],
                                       target_nodes: List[int],
                                       selectable_nodes: List[int],
                                        p_attend: np.ndarray):
    """
    Given a set of sampled live edge graphs, returns an function evaluating the
    multilinear extension for the corresponding influence maximization problem.

    :param live_graphs: list of networkx graphs containing sampled live edges
    :param target_nodes: nodes that should be counted towards the objective
    :param selectable_nodes: nodes that are eligible to be chosen as seeds
    :param p_attend: probability that each node will be influenced if it is chosen as a seed
    :return: a function evaluating the multilinear extension for the corresponding influence maximization problem
    """
    Gs, Ps, ws = live_edge_to_adjlist(live_graphs, target_nodes, p_attend)

    def f_all(x):
        x_expand = np.zeros(len(live_graphs[0]))
        x_expand[selectable_nodes] = x
        return f_all_influmax_multlinear(x_expand, Gs, Ps, ws)

    return f_all


def make_multilinear_gradient_samples(live_graphs, target_nodes, selectable_nodes, p_attend):
    '''
    Given a set of sampled live edge graphs, returns an stochastic gradient
    oracle for the multilinear extension of the corresponding influence
    maximization problem.

    live_graphs: list of networkx graphs containing sampled live edges

    target_nodes: nodes that should be counted towards the objective

    selectable_nodes: nodes that are eligible to be chosen as seeds

    p_attend: probability that each node will be influenced if it is chosen as
    a seed.
    '''

    Gs, Ps, ws = live_edge_to_adjlist(live_graphs, target_nodes, p_attend)

    def gradient(x, batch_size):
        x_expand = np.zeros(len(live_graphs[0]))
        x_expand[selectable_nodes] = x
        samples = random.sample(range(len(Gs)), batch_size)
        grad = gradient_live_edge(x_expand, [Gs[i] for i in samples], [Ps[i] for i in samples],
                                  [ws[i] for i in samples], 1. / batch_size * np.ones(len(Gs)))
        return grad[selectable_nodes]

    return gradient


def live_edge_to_adjlist(live_edge_graphs: List[nx.Graph], target_nodes: List[int], p_attend: np.ndarray) -> tuple:
    """
    :param live_edge_graphs: list of networkx graphs containing sampled live edges
    :param target_nodes: nodes that should be counted towards the objective
    :param p_attend: probability that each node will be influenced if it is chosen as a seed
    :return: Gs, Ps, ws
    """
    Gs = []
    Ps = []
    ws = []
    target_nodes = set(target_nodes)
    for g in live_edge_graphs:
        cc = list(nx.connected_components(g))
        n = len(cc)
        max_degree = max([len(c) for c in cc])
        G_array = np.zeros((n, max_degree), dtype=int)
        P = np.zeros((n, max_degree))
        G_array[:] = -1
        for i in range(n):
            for j, v in enumerate(cc[i]):
                G_array[i, j] = v
                P[i, j] = p_attend[v]
        Gs.append(G_array)
        Ps.append(P)
        w = np.zeros((n))
        for i in range(n):
            w[i] = len(target_nodes.intersection(cc[i]))
        ws.append(w)
    return Gs, Ps, ws


@jit
def gradient_live_edge(x, Gs, Ps, ws, weights):
    '''
    Gradient wrt x of the live edge influence maximization models.

    x: current probability of seeding each node

    Gs/Ps/ws represent the input graphs, as defined in live_edge_to_adjlist
    '''
    grad = np.zeros((len(x)))
    for i in range(len(Gs)):
        grad += weights[i] * gradient_coverage(x, Gs[i], Ps[i], ws[i])
    grad /= len(x)
    return grad


@jit
def objective_live_edge(x, Gs, Ps, ws, weights) -> float:
    """
    活跃边影响最大化模型中的目标，其中节点以对应于 x 的条目的概率被选为种子

    Gs/Ps/ws 表示输入图，具体定义见 live_edge_to_adjlist。

    weights：每个图发生的概率。
    """
    total = 0
    for i in range(len(Gs)):
        total += weights[i] * objective_coverage(x, Gs[i], Ps[i], ws[i])
    return total


'''
The following functions compute gradients/objective values for the multilinear relaxation
of a (probabilistic) coverage function. The function is represented by the arrays G and P. 

Each row of G is a set to be covered, with the entries of the row giving the items that will
cover it (terminated with -1s). The corresponding entry of P gives the probability that
the item will cover that set (independently of all others). 

Corresponding to each row of G is an entry in the vector w, which gives the contribution
to the objective from covering that set.
'''
'''
以下函数计算（概率）覆盖函数的多线性松弛的梯度/目标值。该函数由数组 G 和 P 表示。

G 的每一行是一个要被覆盖的集合，行中的条目给出了将覆盖该集合的项（以 -1 结束）。相应的 P 条目给出了该项覆盖该集合的概率（独立于其他项）。

与 G 的每一行对应的是向量 w 中的一个条目，它给出了覆盖该集合对目标的贡献。
'''


@jit
def gradient_coverage(x, G, P, w):
    '''
    Calculates gradient of the objective at fractional point x.

    x: fractional point as a vector. Should be reshapable into a matrix giving
    probability of choosing copy i of node u.

    G: graph (adjacency list)

    P: probability on each edge.

    w: weights for nodes in R
    '''
    grad = np.zeros((x.shape[0]))
    # process gradient entries one node at a time
    for v in range(G.shape[0]):
        p_all_fail = 1
        for j in range(G.shape[1]):
            if G[v, j] == -1:
                break
            p_all_fail *= 1 - x[G[v, j]] * P[v, j]
        for j in range(G.shape[1]):
            u = G[v, j]
            if u == -1:
                break
            # 0/0 should be 0 here
            if p_all_fail == 0:
                p_others_fail = 0
            else:
                p_others_fail = p_all_fail / (1 - x[u] * P[v, j])
            grad[u] += w[v] * P[v, j] * p_others_fail
    return grad


@jit
def marginal_coverage(x, G, P, w):
    """
    Returns marginal probability that each RHS vertex is reached.
    返回每个右侧顶点被到达的边际概率。
    """
    probs = np.ones((G.shape[0]))
    for v in range(G.shape[0]):
        for j in range(G.shape[1]):
            if G[v, j] == -1:
                break
            u = G[v, j]
            probs[v] *= 1 - x[u] * P[v, j]
    probs = 1 - probs
    return probs


@jit
def objective_coverage(x, G, P, w):
    """
    Weighted objective value: the expected weight of the RHS nodes that are reached.
    加权目标值：被到达的右侧节点的预期权重。
    """
    return np.dot(w, marginal_coverage(x, G, P, w))
