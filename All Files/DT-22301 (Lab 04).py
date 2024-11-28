#!/usr/bin/env python
# coding: utf-8

# ## Syed Ahmed Ali

# ## DT-22301

# ### EXAMPLE 1: Find the traversal path for following graph

# In[21]:


graph={'A':['B','C','E'],
       'B':['D','E'],
       'C':[],
       'D':[],
       'E':[]
       }
def dfs(graph,node,visited):
    if node not in visited:
        visited.append(node)
        for n in graph[node]:
            dfs(graph,n,visited)
    return visited

visited=dfs(graph,'A',[])
print(visited)
visited=dfs(graph,'B',[])
print(visited)
visited=dfs(graph,'C',[])
print(visited)
visited=dfs(graph,'D',[])
print(visited)
visited=dfs(graph,'E',[])
print(visited)


# ### TASK 1 : Find the traversal path for following graphs.

# In[24]:


graph={'A':['B','C'],
       'B':['D','E'],
       'C':['F'],
       'D':[],
       'E':['F'],
       'F':[]
       }
def dfs(graph,node,visited):
    if node not in visited:
        visited.append(node)
        for n in graph[node]:
            dfs(graph,n,visited)
    return visited

visited=dfs(graph,'A',[])
print(visited)


# In[26]:


graph={1:[2,5],
       2:[1,3,5],
       3:[2,4],
       4:[3,5,6],
       5:[1,2,4],
       6:[4]
       }
def dfs(graph,node,visited):
    if node not in visited:
        visited.append(node)
        for n in graph[node]:
            dfs(graph,n,visited)
    return visited

visited=dfs(graph,1,[])
print(visited)


# In[28]:


graph={
    1: [2, 3],
    2: [],
    3: [2,4],
    4: [3]

}
def dfs(graph,node,visited):
    if node not in visited:
        visited.append(node)
        for n in graph[node]:
            dfs(graph,n,visited)
    return visited

visited=dfs(graph,1,[])
print(visited)


# ### EXAMPLE : SHOW A PATH BETWEEN TWO NODES.

# In[31]:


graph={'A':set(['B','C','E']),
       'B':set(['D','E']),
       'C':set([]),
       'D':set([]),
       'E':set([])
       }
def find_path(graph,start,end,path=None):
    if path is None:
        path=[]
    path=path+[start]
    if start==end:
        return path
    if start not in graph:
        return None
    for node in graph[start]:
        if node not in path:
            newpath=find_path(graph,node,end,path)
            if newpath:
                return newpath
    return None

find_path(graph,'A','E')


# ### TASK 2 : Find the single path & all paths between any two nodes for the following graphs.

# In[34]:


graph={'A':set(['B','C']),
       'B':set(['D','E']),
       'C':set(['F']),
       'D':set([]),
       'E':set(['F']),
       'F':set([])
       }
def find_path(graph,start,end,path=None):
    if path is None:
        path=[]
    path=path+[start]
    if start==end:
        return path
    if start not in graph:
        return None
    for node in graph[start]:
        if node not in path:
            newpath=find_path(graph,node,end,path)
            if newpath:
                return newpath
    return None

find_path(graph,'A','E')


# In[36]:


graph={'A':set(['B','C']),
       'B':set(['D','E']),
       'C':set(['F']),
       'D':set([]),
       'E':set(['F']),
       'F':set([])
       }
def find_all_path(graph,start,end,path=None):
    if path is None:
        path=[]
    path=path+[start]
    if start==end:
        return [path]
    if start not in graph:
        return []
    paths=[]
    for node in graph[start]:
        if node not in path:
            newpaths=find_all_path(graph,node,end,path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

find_all_path(graph,'A','E')


# In[38]:


graph={'A':set(['B','D']),
       'B':set(['C']),
       'C':set([]),
       'D':set(['E']),
       'E':set([])
       }
def find_path(graph,start,end,path=None):
    if path is None:
        path=[]
    path=path+[start]
    if start==end:
        return path
    if start not in graph:
        return None
    for node in graph[start]:
        if node not in path:
            newpath=find_path(graph,node,end,path)
            if newpath:
                return newpath
    return None

find_path(graph,'A','E')


# In[40]:


graph={'A':set(['B','D']),
       'B':set(['C']),
       'C':set([]),
       'D':set(['E']),
       'E':set([])
       }
def find_all_path(graph,start,end,path=None):
    if path is None:
        path=[]
    path=path+[start]
    if start==end:
        return [path]
    if start not in graph:
        return []
    paths=[]
    for node in graph[start]:
        if node not in path:
            newpaths=find_all_path(graph,node,end,path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

find_all_path(graph,'A','E')


# In[42]:


graph={'A':set(['B']),
       'B':set(['A','C']),
       'C':set(['B','D','E']),
       'D':set(['C','E','F','G']),
       'E':set(['C','D','F']),
       'F':set(['D','E','G']),
       'G':set(['D'])
       }
def find_path(graph,start,end,path=None):
    if path is None:
        path=[]
    path=path+[start]
    if start==end:
        return path
    if start not in graph:
        return None
    for node in graph[start]:
        if node not in path:
            newpath=find_path(graph,node,end,path)
            if newpath:
                return newpath
    return None

find_path(graph,'A','E')


# In[44]:


graph={'A':set(['B']),
       'B':set(['A','C']),
       'C':set(['B','D','E']),
       'D':set(['C','E','F','G']),
       'E':set(['C','D','F']),
       'F':set(['D','E','G']),
       'G':set(['D'])
       }
def find_all_path(graph,start,end,path=None):
    if path is None:
        path=[]
    path=path+[start]
    if start==end:
        return [path]
    if start not in graph:
        return []
    paths=[]
    for node in graph[start]:
        if node not in path:
            newpaths=find_all_path(graph,node,end,path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

find_all_path(graph,'A','E')


# In[46]:


graph={1:[2,3,4],
       2:[4,5],
       3:[6],
       4:[3,6,7],
       5:[4,7],
       6:[],
       7:[6]
       }
def find_path(graph,start,end,path=None):
    if path is None:
        path=[]
    path=path+[start]
    if start==end:
        return path
    if start not in graph:
        return None
    for node in graph[start]:
        if node not in path:
            newpath=find_path(graph,node,end,path)
            if newpath:
                return newpath
    return None

find_path(graph,1,6)


# In[48]:


graph={1:[2,3,4],
       2:[4,5],
       3:[6],
       4:[3,6,7],
       5:[4,7],
       6:[],
       7:[6]
       }
def find_all_path(graph,start,end,path=None):
    if path is None:
        path=[]
    path=path+[start]
    if start==end:
        return [path]
    if start not in graph:
        return []
    paths=[]
    for node in graph[start]:
        if node not in path:
            newpaths=find_all_path(graph,node,end,path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

find_all_path(graph,1,6)


# ### EXAMPLE : SHOW ALL THE PATHS

# In[51]:


graph={'A':set(['B','C','E']),
       'B':set(['D','E']),
       'C':set([]),
       'D':set([]),
       'E':set([])
       }
def find_all_path(graph,start,end,path=None):
    if path is None:
        path=[]
    path=path+[start]
    if start==end:
        return [path]
    if start not in graph:
        return []
    paths=[]
    for node in graph[start]:
        if node not in path:
            newpaths=find_all_path(graph,node,end,path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

find_all_path(graph,'A','E')


# ### Example shortest path

# In[54]:


from collections import deque
graph={'A':set(['B','C','E']),
       'B':set(['D','E']),
       'C':set([]),
       'D':set([]),
       'E':set([])
       }
def shortest_path(graph,start,end):
  queue=deque([(start,[start])])
  while queue:
    node,path=queue.popleft()
    if node==end:
      return path
    for neighbor in graph[node]:
      if neighbor not in path:
        queue.append((neighbor,path+[neighbor]))
  return None
print(shortest_path(graph,'A','E'))


# ### TASK 3 : Find the shortest paths between any two nodes for the following graphs.

# In[57]:


from collections import deque
graph={1:[2,3],
       2:[4,5],
       3:[6],
       4:[7],
       5:[7],
       6:[5,7],
       7:[]
       }
def shortest_path(graph,start,end):
  queue=deque([(start,[start])])
  while queue:
    node,path=queue.popleft()
    if node==end:
      return path
    for neighbor in graph[node]:
      if neighbor not in path:
        queue.append((neighbor,path+[neighbor]))
  return None
print(shortest_path(graph,1,7))


# In[59]:


from collections import deque
graph={0:[1],
       1:[0,2,3,5],
       2:[1],
       3:[1,4],
       4:[3,5],
       5:[1,4]
       }
def shortest_path(graph,start,end):
  queue=deque([(start,[start])])
  while queue:
    node,path=queue.popleft()
    if node==end:
      return path
    for neighbor in graph[node]:
      if neighbor not in path:
        queue.append((neighbor,path+[neighbor]))
  return None
print(shortest_path(graph,1,4))


# In[61]:


from collections import deque
graph={2:[],
       3:[8,10],
       5:[11],
       7:[8,11],
       8:[9],
       9:[],
       10:[],
       11:[2,9,10]
       }
def shortest_path(graph,start,end):
  queue=deque([(start,[start])])
  while queue:
    node,path=queue.popleft()
    if node==end:
      return path
    for neighbor in graph[node]:
      if neighbor not in path:
        queue.append((neighbor,path+[neighbor]))
  return None
print(shortest_path(graph,3,9))


# ### Question 1:
# 
# Consider the following graph:
# 
# Apply DFS to find traversal path.
# 
# Find single path between 1 & 6.
# 
# Find all paths between 1 & 6.
# 
# Find shortest path between 1 & 6.

# In[64]:


graph={
    1:[2,3,4],
    2:[1,3,4],
    3:[1,2,4],
    4:[1,2,3,5],
    5:[4,6,7,8],
    6:[5,7,8],
    7:[5,6,8],
    8:[5,6,7]
}
#Apply DFS to find traversal path
def dfs(graph,node,visited):
    if node not in visited:
        visited.append(node)
        for n in graph[node]:
            dfs(graph,n,visited)
    return visited

visited=dfs(graph,1,[])
print(visited)
#Find single path between 1 & 6
def find_path(graph,start,end,path=None):
    if path is None:
        path=[]
    path=path+[start]
    if start==end:
        return path
    if start not in graph:
        return None
    for node in graph[start]:
        if node not in path:
            newpath=find_path(graph,node,end,path)
            if newpath:
                return newpath
    return None

print(find_path(graph,1,6))
#Find all paths between 1 & 6
def find_all_path(graph,start,end,path=None):
    if path is None:
        path=[]
    path=path+[start]
    if start==end:
        return [path]
    if start not in graph:
        return []
    paths=[]
    for node in graph[start]:
        if node not in path:
            newpaths=find_all_path(graph,node,end,path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

print(find_all_path(graph,1,6))
#Find shortest path between 1 & 6
def shortest_path(graph,start,end):
  queue=deque([(start,[start])])
  while queue:
    node,path=queue.popleft()
    if node==end:
      return path
    for neighbor in graph[node]:
      if neighbor not in path:
        queue.append((neighbor,path+[neighbor]))
  return None
print(shortest_path(graph,1,6))


# ### Question 2:
# 
# Consider the following graph:
# 
# Apply DFS to find traversal path.
# 
# Find single path between A & G.
# 
# Find all paths between A & G.
# 
# Find shortest path between A & G.

# In[67]:


graph={
    'A':['B','C','D'],
    'B':['A','E'],
    'C':['A','F'],
    'D':['A','E','G'],
    'E':['B','D','G'],
    'F':['C','G'],
    'G':['D','E','F']
}
#Apply DFS to find traversal path
def dfs(graph,node,visited):
    if node not in visited:
        visited.append(node)
        for n in graph[node]:
            dfs(graph,n,visited)
    return visited

visited=dfs(graph,'A',[])
print(visited)
#Find single path between A & G
def find_path(graph,start,end,path=None):
    if path is None:
        path=[]
    path=path+[start]
    if start==end:
        return path
    if start not in graph:
        return None
    for node in graph[start]:
        if node not in path:
            newpath=find_path(graph,node,end,path)
            if newpath:
                return newpath
    return None

print(find_path(graph,'A','G'))
#Find all paths between A & G
def find_all_path(graph,start,end,path=None):
    if path is None:
        path=[]
    path=path+[start]
    if start==end:
        return [path]
    if start not in graph:
        return []
    paths=[]
    for node in graph[start]:
        if node not in path:
            newpaths=find_all_path(graph,node,end,path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

print(find_all_path(graph,'A','G'))
#Find shortest path between A & G
def shortest_path(graph,start,end):
  queue=deque([(start,[start])])
  while queue:
    node,path=queue.popleft()
    if node==end:
      return path
    for neighbor in graph[node]:
      if neighbor not in path:
        queue.append((neighbor,path+[neighbor]))
  return None
print(shortest_path(graph,'A','G'))

