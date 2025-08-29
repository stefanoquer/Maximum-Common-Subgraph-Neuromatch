# Maximum-Common-Subgraph-Neuromatch
Find the Maximum Commong Subgraph between two Graphs using L2 Norm and Cosine Similarity

Refer to the folllowing article:

Quer, Stefano, Thomas Madeo, Andrea Calabrese, Giovanni Squillero, and Enrico Carraro. 2025. "Node Embedding and Cosine Similarity for Efficient Maximum Common Subgraph Discovery" Applied Sciences 15, no. 16: 8920. https://doi.org/10.3390/app15168920

Finding the maximum common induced subgraph is a fundamental problem in computer science. Proven to be NP-hard in the 1970s, nowadays, it has countless applications that still motivate the search for efficient algorithms and practical heuristics. In this work, we extend a state-of-the-art branch-and-bound exact algorithm with new techniques developed in the deep-learning domain, namely graph neural networks and node embeddings, effectively transforming an efficient yet uninformed depth-first search into an effective best-first search. The change enables the algorithm to find suitable solutions within a limited budget, pushing forward the method's time efficiency and applicability on larger graphs. We evaluate the usage of the L2 norm of the node embeddings and the Cumulative Cosine Similarity to classify the nodes of the graphs. The experimental analysis on standard graphs compares our heuristic against the original algorithm and a recently tweaked version that exploits reinforcement learning. Results demonstrate the effectiveness and scalability of the proposed approach, comparing it with the state-of-the-art algorithms. In particular, this approach results in improved results on over 90% of the larger graphs, which would be more challenging in a constrained industrial scenario.

See also:
https://github.com/ZrbTz/MCSHeuristicCpp.git
