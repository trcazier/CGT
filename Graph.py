import networkx as nx
import random
import matplotlib.pyplot as plt

def generate_graph(n_nodes: int, degree: int):
    if degree == n_nodes-1:
        return nx.complete_graph(n_nodes, nx.DiGraph())
    if degree == 0:
        return nx.empty_graph(n_nodes, nx.DiGraph())
    else: 
        # generate a complete graph and then remove random edges until all
        # the nodes have the right degree
        graph = nx.complete_graph(n_nodes, nx.DiGraph())

        # list of all the node degrees
        n_degrees = [n_nodes-1 for _ in range(n_nodes)]
        # we're done when the list of node degrees is only the degree we want
        done = lambda: len(list(filter(lambda n: n == degree, n_degrees))) == n_nodes
        
        # returns the list of all nodes that are not of the right degree
        def get_available():
            available = []
            for i in range(len(n_degrees)):
                if n_degrees[i] != degree:
                    available.append(i)
            return available

        # remove random edges until all the nodes have the right degree
        while not done():
            first = get_available()[0] # choose a node among those with too many edges
            chosen = random.choice(list(graph.neighbors(first)))
            if graph.has_edge(first, chosen): # retry if not in the graph
                graph.remove_edge(first, chosen)
                n_degrees[first] -= 1
        
        return graph

if __name__ == "__main__":
    # test
    for n in range(5, 10):
        for deg in range(0, 4):
            print(f"{n}/10, {deg}/4")
            for i in range(10000):
                G = generate_graph(n, deg)
                for node in list(G.nodes):
                    assert G.out_degree(node) == deg
                # nx.draw(G)
                # plt.show()