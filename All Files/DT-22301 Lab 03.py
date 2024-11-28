#!/usr/bin/env python
# coding: utf-8

# # LAB 03
# # DT-22301
# # Syed Ahmed Ali

# In[76]:


graph_A = {
    'A':['B','E'],
    'B':['F'],
    'C':['G'],
    'D':['E','H'],
    'E':['A','D','H'],
    'F':['B','G','I','J'],
    'G':['C','F','J'],
    'H':['D','E','I'],
    'I':['F','H'],
    'J':['F','G'] 
}


# ## TASK 01: Generate output for Graph A using this code

# In[79]:


def bfs_connected_component(graph, start):
    explored = []
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node not in explored:
            explored.append(node)
            neighbours = graph[node]
            for neighbour in neighbours:
                queue.append(neighbour)
    return explored
print(bfs_connected_component(graph, 'A'))


# In[81]:


graph_B = {
    'A':['B','C','E'],
    'B':['A','D','E'],
    'C':['A','F','G'],
    'D':['B','E'],
    'E':['A','B','D'],
    'F':['C'],
    'G':['C']    
}


# ## TASK 2: Find The shortest path FOR GRAPH A & B ,between two particular nodes.

# In[84]:


from collections import deque

def bfs_shortest_path(graph, start, end):
    queue = deque()
    queue.append((start, [start]))
    visited = set([start])
    
    while queue:
        node, path = queue.popleft()
        
        if node == end:
            return path
        
        for neighbour in graph[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, path + [neighbour]))
    
    return None  
print(bfs_shortest_path(graph_B, 'G', 'D'))  


# ## For Graph A:

# In[87]:


print(bfs_shortest_path(graph_A, 'A', 'C'))


# ## BFS All Path

# In[90]:


graph = {
    'A': set(['B', 'C', 'E']),
    'B': set(['A', 'D', 'E']),
    'C': set(['A', 'F', 'G']),
    'D': set(['B', 'E']),
    'E': set(['A', 'B', 'D']),
    'F': set(['C']),
    'G': set(['C'])
}

def bfs_all_paths(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))
print(list(bfs_all_paths(graph, 'G', 'D')))


# ## Question 1:
# 
# Consider the following grap:
# 
# Apply BFS to find to every possible node present in grph.
# 
# Starting fom 1.
# 
# Find all paths between1 & 6.
# 
# Find shortest path between 1 & 6.

# In[93]:


graph = {
 '1': set(['2', '3', '4']),
 '2': set(['1', '3', '4']),
 '3': set(['1', '2', '4']),
 '4': set(['1', '2', '3', '5']),
 '5': set(['4', '6', '7', '8']),
 '6': set(['5', '7', '8']),
 '7': set(['5', '6', '8']),
 '8': set(['5', '6', '7'])
 }


# In[95]:


reachable_nodes = bfs_connected_component(graph, '1')
print("All reachable nodes from 1:", reachable_nodes)


# In[97]:


all_paths = list(bfs_all_paths(graph, '1', '6'))
print("All paths between 1 and 6:")
for path in all_paths:
    print(path)


# In[99]:


shortest_path = bfs_shortest_path(graph, '1', '6')
print("Shortest path between 1 and 6:", shortest_path)


# ## Question 2:
# 
# Consider the following graph
# 
# Apply BFS to find to every possible node present in graph. Start fromA.
# 
# Find all paths between A& G.
# 
# Find shortest path between A & G.

# In[102]:


graph = {
    'A': set(['A', 'B', 'C', 'D']),
    'B': set(['A', 'E']),
    'C': set(['A', 'F']),
    'D': set(['A', 'E', 'G']),
    'E': set(['B', 'D', 'G']),
    'F': set(['C', 'G']),
    'G': set(['D', 'E'])
}


# In[104]:


reachable_nodes = bfs_connected_component(graph, 'A')
print("All reachable nodes from A:", reachable_nodes)


# In[106]:


all_paths = list(bfs_all_paths(graph, 'A', 'G'))
print("All paths between A and G:")
for path in all_paths:
    print(path)


# In[108]:


shortest_path = bfs_shortest_path(graph, 'A', 'G')
print("Shortest path between A and G:", shortest_path)

