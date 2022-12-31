import networkx as nx
import random
import matplotlib.pyplot as plt
import math

def generate_random_graph(n_nodes: int, degree: int, directed: bool=True):
    if directed:
        gtype = nx.DiGraph() 
    else:
        gtype = nx.Graph() 
    
    if degree == n_nodes-1:
        return nx.complete_graph(n_nodes, gtype)
    if degree == 0:
        return nx.empty_graph(n_nodes, gtype)
    else: 
        # generate a complete graph and then remove random edges until all
        # the nodes have the right degree
        graph = nx.complete_graph(n_nodes, gtype)

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


def DCOP_generate_IL():
    return generate_random_graph(7, 0)

def DCOP_generate_JAL():
    return generate_random_graph(7, 6)

def DCOP_generate_JLAL_1():
    return generate_random_graph(7, 2)

def DCOP_generate_JLAL_2():
    graph = generate_random_graph(7, 0)

    graph.add_edge(1,2)
    graph.add_edge(2,1)

    graph.add_edge(2,3)
    graph.add_edge(3,2)

    graph.add_edge(1,3)
    graph.add_edge(3,1)

    graph.add_edge(5,6)
    graph.add_edge(6,5)

    return graph

def DCOP_generate_JLAL_3():
    graph = DCOP_generate_JLAL_2()
    graph.add_edge(1,5)
    graph.add_edge(5,1)
    return graph


if __name__ == "__main__":
    # tests the validity of a lot of graphs
    for nod in range(5, 10):
        for deg in range(0, 4):
            print(f"{nod+1}/10, {deg+1}/4")
            for i in range(10000):
                G = generate_random_graph(nod, deg)
                for node in list(G.nodes):
                    assert G.out_degree(node) == deg
                # nx.draw(G)
                # plt.show()
    for G in [
        generate_random_graph(5, 0), # IG
        generate_random_graph(5, 2), # LJAL-2
        generate_random_graph(5, 3), # LJAL-3
        generate_random_graph(5, 4), # JAL

        DCOP_generate_IL(),
        DCOP_generate_JAL(),
        DCOP_generate_JLAL_1(),
        DCOP_generate_JLAL_2(),
        DCOP_generate_JLAL_3()
    ]:
        nx.draw(G)
        plt.show()
    
    
    