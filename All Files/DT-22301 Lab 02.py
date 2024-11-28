#!/usr/bin/env python
# coding: utf-8

# # DT-22301
# # Syed Ahmed Ali

# # Example 01:

# In[245]:


graph = {
    'a': ['c'],
    'b': ['c','e'],
    'c': ['a','b','d','e'],
    'd': ['c'],
    'e': ['b','c'],
    'f': [],
}
for node, neighbors in graph.items():
    print(node,'->',neighbors)


# # Task 01: Generate the Graph

# In[248]:


graph = {
    'a': ['b'],
    'b': ['a','c'],
    'c': ['b','d','e'],
    'd': ['c','e','f','g'],
    'e': ['c','d','f'],
    'f': ['d','e'],
    'g': ['d']
}
for node, neighbors in graph.items():
    print(node,'->',neighbors)


# # Example 02:

# In[251]:


graph = {
    'a': ['c'],
    'b': ['c','e'],
    'c': ['a','b','d','e'],
    'd': ['c'],
    'e': ['b','c'],
    'f': [],
}

def get_edges(graph):
    edges = []
    for node in graph:
        for neighbors in graph[node]:
            edges.append((node,neighbors))
    return edges    

get_edges(graph)


# # Task 02: find Edges

# In[254]:


graph = {
    '1':['2','3'],
    '2':['1','3'],
    '3':['1','2'],
    '4':['3']
}
get_edges(graph)


# # Example 03:

# In[257]:


graph = {
    'a': ['c'],
    'b': ['c','e'],
    'c': ['a','b','d','e'],
    'd': ['c'],
    'e': ['b','c'],
    'f': [],
}

def find_isolated_nodes(graph):
    isolated_nodes = []
    for node in graph:
        if not graph[node]:
            isolated_nodes.append(node)
    return isolated_nodes    
print("Isolated nodes in the graph: ",find_isolated_nodes(graph))        


# # Task 03: find isolated node

# In[260]:


graph = {
    'a' : ['b','d'],
    'b' : ['a','d'],
    'c' : ['b'],
    'd' : ['a','d'],
    'e' : [],
}
print("Isolated nodes in the graph: ",find_isolated_nodes(graph)) 


# # Example 04:

# In[263]:


graph = {
    'a': ['c'],
    'b': ['c', 'e'],
    'c': ['a', 'b', 'd', 'e'],
    'd': ['c'],
    'e': ['b', 'c'],
    'f': []
}

def find_path(graph, start, end, path=None):
    if path is None:
        path = []

   
    path.append(start)
    
    if start == end:
        return path
    
    if start not in graph:
        return None

    for neighbor in graph[start]:
        if neighbor not in path:
            new_path = find_path(graph, neighbor, end, path.copy())
            if new_path:
                return new_path
    
    return None

path = find_path(graph, 'e', 'd')
if path:
    print("Path between 'e' and 'd':", path)
else:
    print("No path found between 'e' and 'd'")


# # Task 04: Find path between 'd' and 'b'

# In[266]:


path = find_path(graph, 'd', 'b')
if path:
    print("Path between 'd' and 'b':", path)
else:
    print("No path found between 'd' and 'b'")


# # Example 05:

# In[269]:


graph = {
    'a': ['c'],
    'b': ['c', 'e'],
    'c': ['a', 'b', 'd', 'e'],
    'd': ['c'],
    'e': ['b', 'c'],
    'f': []
}

def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    
    if start == end:
        return [path]
    
    if start not in graph:
        return []

    paths = []
    
    for node in graph[start]:
        if node not in path:
            new_paths = find_all_paths(graph, node, end, path)
            for new_path in new_paths:
                paths.append(new_path)
    
    return paths

paths = find_all_paths(graph, 'a', 'b')

if paths:
    print("Paths between 'a' and 'b':", paths)
else:
    print("No path found between 'a' and 'b'")


# # TASK 05: Find paths between 'd' and 'b'

# In[272]:


paths = find_all_paths(graph, 'd', 'b')

if paths:
    print("Paths between 'd' and 'b':", paths)
else:
    print("No path found between 'd' and 'b'")


# # Example 06

# In[275]:


graph = {
    'a': ['c'],
    'b': ['c', 'e'],
    'c': ['a', 'b', 'd', 'e'],
    'd': ['c'],
    'e': ['b', 'c'],
    'f': []
}

def find_shortest_path(graph, start, end, path=[]):
    path = path + [start] 
    
    if start == end:  
        return path
    
    if start not in graph:
        return []
        
    shortest = []
    
    for node in graph[start]:
        if node not in path: 
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:  
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath  
    
    return shortest

path = find_shortest_path(graph, 'a', 'b')

if path:
    print("Shortest path between 'a' and 'b':", path)
else:
    print("No path found between 'a' and 'b'")


# # Task 06: find the path between ‘d’ and ‘b’

# In[278]:


path = find_shortest_path(graph, 'd', 'b')

if path:
    print("Shortest path between 'd' and 'b':", path)
else:
    print("No path found between 'd' and 'b'")


# # Example 07

# In[226]:


def add_node(graph):
    while True:
        new_node = input("Enter the name of the new node (or type 'exit' to stop): ")
        
        if new_node.lower() == 'exit':
            print("Ending the function. No more nodes will be added.")
            break
        
        if new_node not in graph:
            graph[new_node] = []
            print(f"Node '{new_node}' added successfully.")
        else:
            print(f"Node '{new_node}' already exists.")
        
        add_edge_option = input(f"Do you want to create an edge from '{new_node}' to another node? (yes/no): ")
        
        if add_edge_option == 'yes':
            other_node = input(f"Enter the name of the node to create an edge with '{new_node}': ")
            
            if other_node not in graph:
                print(f"Node '{other_node}' does not exist in the graph.")
            else:
                graph[new_node].append(other_node)
                graph[other_node].append(new_node)
                print(f"Edge added between '{new_node}' and '{other_node}'.")
        
    return graph

graph = {
    'A': ['C'],
    'B': ['C', 'E'],
    'C': ['A', 'B', 'D', 'E'],
    'D': ['C'],
    'E': ['B', 'C'],
    'F': []
}

graph = add_node(graph)
print("Final graph:", graph)


# # Task 07: Add Edge to ‘d’ in ‘c’

# In[43]:


graph = {
    'a': ['c'],
    'b': ['c', 'e'],
    'c': ['a', 'b', 'e'],
    'd': [],
    'e': ['b', 'c'],
    'f': []
}
def add_edge(graph, u, v):
    if u not in graph:
        graph[u] = []
    if v not in graph:
        graph[v] = []
    
    graph[u].append(v)
    graph[v].append(u)

    return graph

print("Graph before adding node:", graph)
print("Graph after adding node:", add_edge(graph,'c', 'd'))
for node, neighbors in graph.items():
    print(node,'->',neighbors)


# # Example 08

# In[45]:


def detect_cycle(graph):
    is_directed = True

    def dfs_directed(node, visited, recursion_stack, path):
        visited[node] = True
        recursion_stack[node] = True
        path.append(node)  

        for neighbor in graph[node]:
            if not visited[neighbor]:
                cyclic, cycle_path = dfs_directed(neighbor, visited, recursion_stack, path)
                if cyclic:
                    return True, cycle_path
            elif recursion_stack[neighbor]:
                cycle_start_index = path.index(neighbor)  
                return True, path[cycle_start_index:]  

        recursion_stack[node] = False
        path.pop()  
        return False, []

    visited = {node: False for node in graph}
    recursion_stack = {node: False for node in graph}

    for node in graph:
        if not visited[node]:
            cyclic, cycle_path = dfs_directed(node, visited, recursion_stack, [])
            if cyclic:
                return f"Graph is Cyclic, cycle found involving nodes: {cycle_path}"
    
    return "Graph is Acyclic"


graph = {
    "a": ["a", "c"],
    "b": ["c", "e"],
    "c": ["a", "b", "d", "e"],
    "d": ["e"],
    "e": ["c", "b"],
    "f": []
}

graph1 = {
    "a": ["c"],
    "c": []
}


print(detect_cycle(graph))
print(detect_cycle(graph1))  


# # Task 08: Generate the output of above code(cycle)

# In[48]:


graph = {1: [2, 4],
         2: [4],
         4: [5],
         5: [4],
         3: [],
         6:[3]}
print(detect_cycle(graph))


# # Example 09:

# In[51]:


graph = {
    "a": ["c"],
    "b": ["c", "e"],
    "c": ["a", "b", "d", "e"],
    "d": ["c"],
    "e": ["c", "b"],
    "f": []
}

def find_degree(graph,node):
    degree = 0
    t = []
    for neighbour in graph[node]:
        t.append(neighbour)
        degree = degree + 1
    return degree

Degree = find_degree(graph,"c")
print("degree of the vertex: ",Degree)


# # Task 09: DEGREE OF VERTEX

# In[54]:


graph = {
    'v1':['v2','v4'],
    'v2':['v1','v3','v4'],
    'v3':['v2'],
    'v4':['v1','v2'],
    'v5':[]
}
Degree = find_degree(graph,"v1")
print("degree of the vertex: ",Degree)

Degree = find_degree(graph,"v2")
print("degree of the vertex: ",Degree)

Degree = find_degree(graph,"v3")
print("degree of the vertex: ",Degree)

Degree = find_degree(graph,"v4")
print("degree of the vertex: ",Degree)

Degree = find_degree(graph,"v5")
print("degree of the vertex: ",Degree)


# # Example 10

# In[57]:


def graph_connected(graph, seen_node=None, start=None):
    if seen_node is None:
        seen_node = set()
        nodes = list(graph.keys())
        if not start:
            start = nodes[0]
    
    seen_node.add(start)

    for othernodes in graph[start]:  
        if othernodes not in seen_node:
            graph_connected(graph, seen_node, othernodes)
    
    
    return len(seen_node) == len(graph)


graph = {
    "a": ["a", "c"],
    "b": ["c", "e"],
    "c": ["a", "b", "d", "e"],
    "d": ["c"],
    "e": ["c", "b"],
    "f": []  
}

conn = graph_connected(graph)
if conn:
    print("The graph is connected")
else:
    print("The graph is not connected")


# In[59]:


graphA = {
    'a':['b','e'],
    'b':['a','c','d'],
    'c':['b','d'],
    'd':['b','c','e'],
    'e':['a','d']
}
conn = graph_connected(graphA)
if conn:
    print("The graph is connected")
else:
    print("The graph is not connected")


# # EXERCISE
# ## Question 1: Define the following terms: 

# # Regular Graph:
# ### A regular graph is a type of graph where each vertex has the same number of edges (i.e., the same degree).In a k-regular graph, every vertex has exactly k edges connected to it.
# 
# # Null Graph:
# ### A graph with no edges at all. It may consist of one or more vertices but no connections between them.
# 
# # Trivial Graph:
# ### A graph with only one vertex and no edges.
# 
# # Simple Graph:
# ### A graph that has no loops (edges connecting a vertex to itself) or parallel edges (multiple edges between the same pair of vertices).
# 
# # Connected Graph:
# ### A graph in which there is a path between every pair of vertices, meaning all vertices are reachable from one another.
# 
# # Disconnected Graph:
# ### A graph in which not all vertices are connected, meaning there are at least two vertices that do not have a path between them.
# 
# # Complete Graph:
# ### A graph in which there is an edge between every pair of distinct vertices. In a graph with n vertices, there are exactly n(n-1)/2 edges.
# 
# # Cyclic Graph:
# ### A graph that contains at least one cycle, meaning there is a path that starts and ends at the same vertex without repeating any edges.
# 
# # Degree of Vertex:
# ### The number of edges connected to a vertex. For directed graphs, the in-degree refers to incoming edges, and the out-degree refers to outgoing edges.
# 
# # Loop:
# ### An edge that connects a vertex to itself.
# 
# # Parallel Edges:
# ### Two or more edges that connect the same pair of vertices in a graph. These occur in multigraphs, not in simple graphs.

# ## Question 2 (a): Consider the following graph:
# Find isolated nodes.
# 
# Find path between two vertex/ node 1 and 7.
# 
# Find all paths in graphs.
# 
# Find shortest path between nodes 1 and 7.
# 
# Determine cycles in graphs.
# 
# Add an edge named 9.
# 
# Find degree of vertex 4.
# 
# Find if the graph is connected.

# In[170]:


graph = {
    '1':['2','3','4'],
    '2':['1','3','4'],
    '3':['1','2','4'],
    '4':['1','2','3','5'],
    '5':['4','6','7','8'],
    '6':['5','7','8'],
    '7':['5','6','8'],
    '8':['5','6','7']
}


# In[66]:


# Example 3
print("Isolated nodes in the graph: ",find_isolated_nodes(graph))


# In[172]:


# Example 4
path = find_path(graph, '1', '7') 
if path:
    print("Path between '1' and '7':", path)
else:
    print("No path found between '1' and '7'")


# In[70]:


# Example 5
paths = find_all_paths(graph, '1', '7')
if paths:
    print("Paths between '1' and '7':", paths)
else:
    print("No path found between '1' and '7'")


# In[72]:


# Example 6
S_path = find_shortest_path(graph, '1', '7')
if S_path:
    print("Shortest path between '1' and '7':", S_path)
else:
    print("No path found between '1' and '7'")


# In[74]:


def detect_all_cycles(graph):
    def dfs_undirected(node, visited, parent, path, all_cycles):
        visited[node] = True
        path.append(node)

        for neighbor in graph[node]:
            if not visited[neighbor]:
                cyclic, cycle_path = dfs_undirected(neighbor, visited, node, path, all_cycles)
                if cyclic:
                    all_cycles.append(cycle_path)
            elif neighbor != parent and neighbor in path:
                
                cycle_start_index = path.index(neighbor)
                all_cycles.append(path[cycle_start_index:] + [neighbor])

        path.pop()  
        return False, []

    visited = {node: False for node in graph}
    all_cycles = []

    for node in graph:
        if not visited[node]:
            dfs_undirected(node, visited, None, [], all_cycles)
    
    if all_cycles:
        return f"Graph is Cyclic, cycles found: {all_cycles}"
    else:
        return "Graph is Acyclic"

print(detect_all_cycles(graph))


# In[76]:


# Example 7
graph = add_node(graph)
print(graph)


# In[78]:


for node, neighbors in graph.items():
    print(node,'->',neighbors)


# In[80]:


# Example 9
Degree = find_degree(graph,'4')
print("degree of the vertex: ",Degree)


# In[82]:


# Example 10
conn = graph_connected(graphA)
if conn:
    print("The graph is connected")
else:
    print("The graph is not connected")


# ## Question 2 (b): Consider the following graph:
# Find isolated nodes.
# 
# Find path between two vertex/node B and A.
# 
# Find all paths in graphs.
# 
# Find shortest path between nodes B and A.
# 
# Determine cycles in graphs.
# 
# Add an edge named K.
# 
# Find degree of vertex G.
# 
# Find if the graph is connected.

# In[174]:


graph = {
    'A': ['C'],
    'B': ['G', 'F'],
    'C': ['A', 'G'],
    'D': ['G', 'I'],
    'E': ['G', 'I', 'J'],
    'F': ['B','G'],
    'G': ['B', 'C', 'D', 'E','F', 'H','J'],
    'H': ['G','I','J'],
    'I': ['D', 'E','H'],
    'J': ['E','G', 'H']
}


# In[87]:


# Example 3
print("Isolated nodes in the graph: ",find_isolated_nodes(graph))


# In[176]:


# Example 4
path = find_path(graph, 'B', 'A') 
if path:
    print("Path between 'B' and 'A':", path)
else:
    print("No path found between 'B' and 'A'")


# In[91]:


# Example 5
paths = find_all_paths(graph, 'A', 'B')
if paths:
    print("Paths between 'A' and 'B':", paths)
else:
    print("No path found between 'A' and 'B'")


# In[93]:


# Example 6
S_path = find_shortest_path(graph, 'B', 'A')
if S_path:
    print("Shortest path between 'B' and 'A':", S_path)
else:
    print("No path found between 'B' and 'A'")


# In[95]:


# Example 8
print(detect_all_cycles(graph))


# In[97]:


# Example 7
graph = add_node(graph)
print(graph)


# In[99]:


for node, neighbors in graph.items():
    print(node,'->',neighbors)


# In[101]:


# Example 9
Degree = find_degree(graph,'G')
print("degree of the vertex: ",Degree)


# In[240]:


# Example 10
conn = graph_connected(graph)
if conn:
    print("The graph is connected")
else:
    print("The graph is not connected")


# ## Question 3: Consider the following graph:
# Find isolated nodes.
# 
# Find path between two vertex/node Thomas’ Farm and
# Library.
# 
# Find all paths in graph.
# 
# Finding shortest path between nodes Thomas’ Farm and
# Library.
# 
# Determine cycles in graphs.
# 
# Add an edge named John’s House.
# 
# Find degree of vertex Bakery.
# 
# Find if the graph is connected.

# In[230]:


graph = {
    "Mayor's House": ["Bakery", "Brewery"],
    "Bakery": ["Mayor's House", "McFane's Farm"],
    "McFane's Farm": ["Bakery","Brewery","Thomas' Farm"],
    "Thomas' Farm": ["McFane's Farm"],
    "Brewery": ["Mayor's House","McFane's Farm", "Inn"],
    "Inn": ["Brewery", "Library", "Dry Cleaner"],
    "Library": ["Inn", "City Hall"],
    "City Hall": ["Library", "Dry Cleaner"],
    "Dry Cleaner": ["Inn", "City Hall"]
}


# In[142]:


# Example 3
print("Isolated nodes in the graph: ",find_isolated_nodes(graph))


# In[192]:


# Example 4

path = find_path(graph, "Thomas' Farm", "Library")
if path:
    print(f"Path from Thomas' Farm to Library: {path}")
else:
    print("No path found.")


# In[204]:


# Example 5
paths = find_all_paths(graph, "Thomas' Farm", "Library")
if paths:
    print(f"Paths between Thomas' Farm and Library:{paths}")
else:
    print("No path found")


# In[216]:


# Example 6
S_path = find_shortest_path(graph, "Thomas' Farm", "Library")
if S_path:
    print(f"Shortest path between Thomas' Farm and Library:{S_path}")
else:
    print("No path found")


# In[218]:


# Example 8
print(detect_all_cycles(graph))


# In[234]:


# Example 7
graph = add_node(graph)
print(graph)


# In[236]:


for node, neighbors in graph.items():
    print(node,'->',neighbors)


# In[238]:


# Example 9
Degree = find_degree(graph,'Bakery')
print("degree of the vertex: ",Degree)


# In[243]:


# Example 10
conn = graph_connected(graph)
if conn:
    print("The graph is connected")
else:
    print("The graph is not connected")


# In[ ]:




