
#first program
print("Hello world!!!")
print("#hello hash")

#variables
name = 'John'
age = 20
new_patient = True

print(name, " ", age, " ", new_patient)

#user input
name = input("Enter your name: ")
print("Hello ", name)

#type conversion
birth_year = int(input("Enter your birth year: "))
recent_year = 2022
age = recent_year - birth_year
print("Your age: ", age)

#calculator
first = float(input("Enter first num: "))
second = float(input("Enter second num: "))
sum = first + second
print("Sum: " + str(sum))




#loop
i=1
while i<= 5:
  print(i * '#')
  i+=1

#list
names = ["A", "B", "C", "D", "E"]
names[0] = "L"
print(names)
print(names[0])
print(names[-1]) #print the element from the ending
print(names[0:3])

#list method
names.append("J")
print(names)

print(len(names))

names.insert(3, "M")
print(names)

names.remove("M")
print(names)

names.clear()
print(names)

print("D" in names)

#for loop with list
names = ["A", "B", "C", "D", "E"]
for n in names:
  print(n)

list1=[5,9,2,7,18]
list2=[1,10,11,3,19]
list3=[]
for i in range(len(list1)):
  if (list1[i]%2==0):
    list3.append(list1[i])
for i in range(len(list2)):
  if (list2[i]%2!=0):
    list3.append(list2[i])
print('Unsorted list : ',list3)
  
for i in range(0,len(list3)-1):  
    for j in range(len(list3)-1):  
       if(list3[j]>list3[j+1]):  
        temp = list3[j]  
        list3[j] = list3[j+1]  
        list3[j+1] = temp  
   
print('sorted list : ',list3)


#range function
number = range(5)
for n in number:
  print(n)
print("-------")
for k in range(2, 10):
  print(k)

# tuple
num = (1, 3, 5)
print(len(num))

#fibonacci
n=10
n1=0
n2=1
nextNum = n1+n2
print(n1)
print(n2)

for i in range(3, n+1):
  print(nextNum) 
  n1 = n2
  n2 = nextNum
  nextNum = n1 + n2

#factorial
f = 3
mul = 1
for i in range(1, f+1):
  mul *= i

print(mul)

#insertion sorting
li = [4, 3, 1, 5, 9]
for i in range(len(li)):
  temp = li[i]
  j = i-1
  while (j>=0 and li[j] > temp):
    li[j+1] = li[j]
    j-=1
  li[j+1] = temp

print(li)

#prime number approach-1
n = 10
p = [2]
for i in range(3, n+1):
  pri = False
  for j in range(2, i):
    if(i%j == 0):
      pri = True
  if(not pri):
    p.append(i)

print(p)

#prime number approach-2
n = int(input("Enter a number: "))
k = int(n/2)

if(n == 2):
  print("Prime num")
elif(n>2):
  pri = False
  for j in range(2, k+1):
    if(n%j == 0):
      pri = True
      break
  if(not pri):
    print("Prime num")
  else:
    print("Not Prime num")
else:
  print("Not Prime num")


#dfs
graph = {
    '0': ['1', '3'],
    '1': ['3', '2', '5', '6'],
    '2': ['3', '1', '5', '4'],
    '3': ['1', '2', '4'],
    '4': ['3', '2', '6'],
    '5': ['1', '2'],
    '6': ['1', '4']
}
visited = set()
def dfs(graph, visited, node):
  if node not in visited:
    print(node)
    visited.add(node)
    for adjacent in graph[node]:
      dfs(graph, visited, adjacent)

print("Graph in depth first search: ")
dfs(graph, visited, '0')

#even, odd, sort
l1 = [10, 247, 44, 73, 57, 9] 
l2 = [88, 22, 5, 2, 9, 0, 3]
lf = []

for l in l1: #picking even from l1
  if (l%2 == 0):
    lf.append(l)

for l in l2: #picking even from l1
  if (l%2 == 1):
    lf.append(l)

print(lf)

for i in range(len(lf)):
  temp = lf[i]
  j = i-1
  while (j>=0 and lf[j] > temp):
    lf[j+1] = lf[j]
    j-=1
  lf[j+1] = temp

print(lf)

list1=[10,20,23,11,17]

list2=[13,43,24,36,12]

list3 = [] #list had to be declared 

for i in list1: 

  if(i%2 == 0):

    list3.append(i)



for i in list2: 

  if(i%2 != 0):

    list3.append(i)

def st(l):

  for i in range(1,len(l)):

    temp=l[i]

    j=i-1

    while j>=0 and temp<l[j]:

      l[j+1]=l[j]

      j=j-1

      l[j+1]=temp   

st(list3)

print("The elements are: ")

print(list3)

graph = {
  '0' : ['2','1'],
  '1' : ['2'],
  '2' : ['0', '3'],
  '3' : ['3']
}

#BFS
visited = [] 
queue = []    

def bfs(visited, graph, node):
  visited.append(node)
  queue.append(node)

  while queue:
    m = queue.pop(0) 
    print (m, end = " ") 

    for neighbour in graph[m]:
      if neighbour not in visited:
        visited.append(neighbour)
        queue.append(neighbour)

bfs(visited, graph, '2')

#plot
import matplotlib.pyplot as p
x = [k for k in range(0, 55)]
y = [k*3 for k in x]

print(x)
print(y)

p.plot(x, y)

p.xlabel("x - axis")
p.ylabel("y - axis")
p.title("x vs y graph")

#plot
import matplotlib.pyplot as p

p.axis([1, 3, 1, 4])

x = [1, 2, 3]
y = [2, 4, 1]

p.plot(x, y)


#gcd with lib
import math as m

k =m.gcd(10, 20)
print(k)

# define the gcd value
from functools import reduce
from math import *

n = int(input("Enter the number: "))
v = []
for i in range(n):
  x = int(input("Enter the number "+str(i+1)+": "))
  v.append(x)
g = reduce(gcd, v)

print(g)

import random
    
class Agent:
    def __init__(self):
        def program(percept):abstract
        self.program=program

loc_A,loc_B,loc_C,loc_D='A','B','C','D'


class reflexVaccumAgent(Agent):
    def __init__(self):
        Agent.__init__(self)

        action=' '

        def program(percept):
            location=percept[0]
            status=percept[1]
            if status=='Dirty': action= 'Suck'
            elif location==loc_A: action= 'Right'
            elif location==loc_B: action= 'Down'
            elif location==loc_C: action= 'Left'
            elif location==loc_D: action= 'Up'
            

            percept=(location,status)
            print('Agent perceives %s and does %s' %(percept,action))
            return action
            
        self.program=program

        
class vaccumEnvironment():

    def __init__(self):
        self.status={ loc_A:random.choice(['Clean','Dirty']),
                      loc_B:random.choice(['Clean','Dirty']),
                      loc_C:random.choice(['Clean','Dirty']),
                      loc_D:random.choice(['Clean','Dirty'])
                      
                      }
    def add_object(self,object,location=None):
        object.location=location or self.default_location(object)

    def default_location(self,object):
        return random.choice([loc_A,loc_B,loc_C,loc_D])

    def percept(self,agent):
        return (agent.location,self.status[agent.location])

    def execute_action(self,agent,action):
        if action=='Right': agent.location=loc_B
        elif action=='Down': agent.location=loc_C
        elif action=='Left': agent.location=loc_D
        elif action=='Up': agent.location=loc_D
        elif action=='Suck':
            #if self.status[agent.location]=='Dirty'
            self.status[agent.location]='Clean'

      
Ragent=reflexVaccumAgent()
env=vaccumEnvironment()
env.add_object(Ragent)


for steps in range(5):
    print(env.percept(Ragent))
    action=Ragent.program(env.percept(Ragent))
    env.execute_action(Ragent,action)

class NumPyramid:
  def __init__(self, n):
    self.n = n
  def PrintMyNum(self):
    print('number: ', self.n)

k = NumPyramid(10)
k.PrintMyNum()

#number triangle
class NumPyramid:
  def __init__(self, n):
    self.n = n
  def PrintNumPyramid(self):
    for x in range(self.n):
      for y in range(x+1):
        print(y+1, end=' ')
      print()

k = NumPyramid(5)
k.PrintNumPyramid()

# * pyramid
class NumPyramid:
  def __init__(self, n):
    self.n = n
  def PrintNumPyramid(self):
    m = self.n
    for x in range(self.n):
      for y in range(m):
        print(end=' ')
      m -= 1
      for y in range(x+1):
        print("*", end=' ')
      print()

k = NumPyramid(5)
k.PrintNumPyramid()

# reverse number triangle
class NumPyramid:
  def __init__(self, n):
    self.n = n
  def PrintNumPyramid(self):
    x = self.n
    while(x>0):
      y = self.n
      while(y>=self.n-x+1):
        print(y, end=' ')
        y-=1
      x-=1
      print()

k = NumPyramid(5)
k.PrintNumPyramid()

# right side * pyramid
class NumPyramid:
  def __init__(self, n):
    self.n = n
  def PrintNumPyramid(self):
    m = self.n
    for x in range(self.n):
      for y in range(m):
        print(end=' ')
      m -= 1
      for y in range(x+1):
        print("*", end='')
      print()

k = NumPyramid(5)
k.PrintNumPyramid()

#simple relex agent implementation

import random as r
#creating environment
class Environment():
  def __init__(self):
    self.locationCondition = {'A':1, 'B':1}

    #randomize location condition 0 -> dirty & 1 -> clean
    self.locationCondition['A'] = r.randint(0, 1)
    self.locationCondition['B'] = r.randint(0, 1)

#testing-1:
#env = Environment()
#print(env.locationCondition)



#simple reflex agent
import random as r

class Environment():
  def __init__(self):
    self.state = {
        'A': 'clean',
        'B': 'clean'
    }
    self.state['A'] = r.choice(['clean', 'dirty'])
    self.state['B'] = r.choice(['clean', 'dirty'])

class reflexAgent():
  def __init__(self, env):
    self.env = env

    if self.env.state['A'] == 'clean':
      print('[A, clean] -> MOVE RIGHT')
    elif self.env.state['A'] == 'dirty':
      print('[A, dirty] -> SUCK -> [A, clean]')
    if self.env.state['B'] == 'clean':
      print('[B, clean] -> MOVE LEFT')
    elif self.env.state['B'] == 'dirty':
      print('[B, dirty] -> SUCK -> [B, clean]')
    print('\n')

for timeStep in range(5):
  theEnv = Environment()
  theAgent = reflexAgent(theEnv)

GRAPH = {'Arad' : {'Sibiu' : 140, 'Zerind' : 75, 'Timisoara' : 118},
           'Zerind' : {'Arad' : 75, 'Oradea' : 71},
           'Oradea' : {'Zerind' : 75, 'Sibiu' : 151},
           'Sibiu' : {'Arad' : 140, 'Oradea' : 151, 'Fagaras' : 99, 'Rimnicu' : 80},
           'Timisoara' : {'Arad' : 118, 'Lugoj' : 111 },
           'Lugoj' : {'Timisoara' : 111, 'Mahadia' : 70},
           'Mahadia' : {'Lugoj' : 70, 'Drobeta' : 75},
           'Drobeta' : {'Mahadia' : 75, 'Cariova' : 120},
           'Craiova':{'Drobeta':120,'Rimnicu':146, 'pitesti':138},
           'Rimnicu':{'sibiu':80,'Craioa':146,'Pitesti':97},
           'Fagaras':{'sibiu':99, 'Bucharest': 211},
           'pitesti':{'Rimnicu':97, 'Craiove':138, 'Bucharest': 101},
           'Bucharest':{'Fagaras':211, 'Pitesti':101, 'Giurgiu':90, 'Urziceni':85},
           'Giurgiu':{'Bucharest': 90},
           'Urziceni':{'Bucharest': 85,'Vaslui':142, 'Hirsova':98},
           'Hirsova':{'Urziceni': 98, 'Eforie': 86},
           'Eforie':{'Hirsova': 86},
           'Vaslui' : {'Iasi': 92, 'Urziceni' : 142},
           'Iasi' : {'Vaslui' : 92, 'Neamt' : 87},
           'Neamt' : {'Iasi' : 87}
}

print(GRAPH)



GRAPH = {
            'Arad': {'Sibiu': 140, 'Zerind': 75, 'Timisoara': 118},\
            'Zerind': {'Arad': 75, 'Oradea': 71},
            'Oradea': {'Zerind': 71, 'Sibiu': 151},
            'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu': 80},
            'Timisoara': {'Arad': 118, 'Lugoj': 111},
            'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
            'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
            'Drobeta': {'Mehadia': 75, 'Craiova': 120},
            'Craiova': {'Drobeta': 120, 'Rimnicu': 146, 'Pitesti': 138},
            'Rimnicu': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
            'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
            'Pitesti': {'Rimnicu': 97, 'Craiova': 138, 'Bucharest': 101},
            'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
            'Giurgiu': {'Bucharest': 90},
            'Urziceni': {'Bucharest': 85, 'Vaslui': 142, 'Hirsova': 98},
            'Hirsova': {'Urziceni': 98, 'Eforie': 86},
            'Eforie': {'Hirsova': 86},
            'Vaslui': {'Iasi': 92, 'Urziceni': 142},
            'Iasi': {'Vaslui': 92, 'Neamt': 87},
            'Neamt': {'Iasi': 87}
        }

def dfs_paths(source, destination, path=None):
    """All possible paths from source to destination using depth-first search
    :param source: Source city name
    :param destination: Destination city name
    :param path: Current traversed path (Default value = None)
    :yields: All possible paths from source to destination
    """
    if path is None:
        path = [source]
    if source == destination:
        yield path
    for next_node in set(GRAPH[source].keys()) - set(path):
        yield from dfs_paths(next_node, destination, path + [next_node])

def ucs(source, destination):
    """Cheapest path from source to destination using uniform cost search
    :param source: Source city name
    :param destination: Destination city name
    :returns: Cost and path for cheapest traversal
    """
    from queue import PriorityQueue
    priority_queue, visited = PriorityQueue(), {}
    priority_queue.put((0, source, [source]))
    visited[source] = 0
    while not priority_queue.empty():
        (cost, vertex, path) = priority_queue.get()
        if vertex == destination:
            return cost, path
        for next_node in GRAPH[vertex].keys():
            current_cost = cost + GRAPH[vertex][next_node]
            if not next_node in visited or visited[next_node] >= current_cost:
                visited[next_node] = current_cost
                priority_queue.put((current_cost, next_node, path + [next_node]))

def a_star(source, destination):
    """Optimal path from source to destination using straight line distance heuristic
    :param source: Source city name
    :param destination: Destination city name
    :returns: Heuristic value, cost and path for optimal traversal
    """
    # HERE THE STRAIGHT LINE DISTANCE VALUES ARE IN REFERENCE TO BUCHAREST AS THE DESTINATION
    straight_line = {
                        'Arad': 366,
                        'Zerind': 374,
                        'Oradea': 380,
                        'Sibiu': 253,
                        'Timisoara': 329,
                        'Lugoj': 244,
                        'Mehadia': 241,
                        'Drobeta': 242,
                        'Craiova': 160,
                        'Rimnicu': 193,
                        'Fagaras': 176,
                        'Pitesti': 100,
                        'Bucharest': 0,
                        'Giurgiu': 77,
                        'Urziceni': 80,
                        'Hirsova': 151,
                        'Eforie': 161,
                        'Vaslui': 199,
                        'Iasi': 226,
                        'Neamt': 234
                    }
    from queue import PriorityQueue
    priority_queue, visited = PriorityQueue(), {}
    priority_queue.put((straight_line[source], 0, source, [source]))
    visited[source] = straight_line[source]
    while not priority_queue.empty():
        (heuristic, cost, vertex, path) = priority_queue.get()
        if vertex == destination:
            return heuristic, cost, path
        for next_node in GRAPH[vertex].keys():
            current_cost = cost + GRAPH[vertex][next_node]
            heuristic = current_cost + straight_line[next_node]
            if not next_node in visited or visited[next_node] >= heuristic:
                visited[next_node] = heuristic
                priority_queue.put((heuristic, current_cost, next_node, path + [next_node]))

def main():
    """Main function"""
    print('ENTER SOURCE :', end=' ')
    source = input().strip()
    print('ENTER GOAL :', end=' ')
    goal = input().strip()
    if source not in GRAPH or goal not in GRAPH:
        print('ERROR: CITY DOES NOT EXIST.')
    else:
        print('\nALL POSSIBLE PATHS:')
        paths = dfs_paths(source, goal)
        for path in paths:
            print(' -> '.join(city for city in path))
        print('\nCHEAPEST PATH:')
        cost, cheapest_path = ucs(source, goal)
        print('PATH COST =', cost)
        print(' -> '.join(city for city in cheapest_path))
        print('\nOPTIMAL PATH:')
        heuristic, cost, optimal_path = a_star(source, goal)
        print('HEURISTIC =', heuristic)
        print('PATH COST =', cost)
        print(' -> '.join(city for city in optimal_path))

if __name__ == '__main__':
    main()

l = [1, 2, 3]
k = [j*4 for j in l]
print(l.index(3))
print(k)

tree = {'S': [['A', 1], ['B', 5], ['C', 8]],
        'A': [['S', 1], ['D', 3], ['E', 7], ['G', 9]],
        'B': [['S', 5], ['G', 4]],
        'C': [['S', 8], ['G', 5]],
        'D': [['A', 3]],
        'E': [['A', 7]]}



heuristic = {'S': 8, 'A': 8, 'B': 4, 'C': 3, 'D': 5000, 'E': 5000, 'G': 0}


cost = {'S': 0}             


def AStarSearch():
    global tree, heuristic
    closed = []             
    opened = [['S', 8]]    

    '''find the visited nodes'''
    while True:
        fn = [i[1] for i in opened]     
        chosen_index = fn.index(min(fn))
        node = opened[chosen_index][0]  
        closed.append(opened[chosen_index])
        del opened[chosen_index]
        if closed[-1][0] == 'G':        
            break
        for item in tree[node]:
            if item[0] in [closed_item[0] for closed_item in closed]:
                continue
            cost.update({item[0]: cost[node] + item[1]})           
            fn_node = cost[node] + heuristic[item[0]] + item[1]     
            temp = [item[0], fn_node]
            opened.append(temp)                                     

    '''find optimal sequence'''
    trace_node = 'G'                        
    optimal_sequence = ['G']                
    for i in range(len(closed)-2, -1, -1):
        check_node = closed[i][0]           
        if trace_node in [children[0] for children in tree[check_node]]:
            children_costs = [temp[1] for temp in tree[check_node]]
            children_nodes = [temp[0] for temp in tree[check_node]]

            '''check whether h(s) + g(s) = f(s). If so, append current node to optimal sequence
            change the correct optimal tracing node to current node'''
            if cost[check_node] + children_costs[children_nodes.index(trace_node)] == cost[trace_node]:
                optimal_sequence.append(check_node)
                trace_node = check_node
    optimal_sequence.reverse()             

    return closed, optimal_sequence


if __name__ == '__main__':
    visited_nodes, optimal_nodes = AStarSearch()
    print('visited nodes: ' + str(visited_nodes))
    print('optimal nodes sequence: ' + str(optimal_nodes))

tree = {'a': [['b', 1], ['c', 1], ['d', 1]],
        'b': [['a', 1]],
        'c': [['a', 1], ['e', 2], ['f', 2], ['g', 2]],
        'd': [['a', 1]],
        'e': [['h', 3], ['i', 3], ['c', 2]],
        'f': [['c', 2], ['j', 4], ['k', 4]],
        'g': [['c', 2]],
        'h': [['e', 3]],
        'i': [['e', 3]],
        'j': [['f', 4], ['l', 5]],
        'k': [['f', 4]],
        'l': [['j', 5], ['n', 6]],
        'G': [['l', 6]],
        'n': [['l', 6]]}



heuristic = {'a': 4, 'b': 5, 'c': 3, 'd': 5, 'e': 3, 'f': 3, 'g': 4, 'h': 3, 'i': 4, 'j': 2, 'k': 4, 'l': 1, 'G': 0, 'n': 2,}


cost = {'a': 0} 


def AStarSearch():
    global tree, heuristic
    closed = []             # closed nodes
    opened = [['a', 4]]     # opened nodes

    '''find the visited nodes'''
    while True:
        fn = [i[1] for i in opened]     # fn = f(n) = g(n) + h(n)
        chosen_index = fn.index(min(fn))
        node = opened[chosen_index][0]  # current node
        closed.append(opened[chosen_index])
        del opened[chosen_index]
        if closed[-1][0] == 'G':        # break the loop if node G has been found
            break
        for item in tree[node]:
            if item[0] in [closed_item[0] for closed_item in closed]:
                continue
            cost.update({item[0]: cost[node] + item[1]})            # add nodes to cost dictionary
            fn_node = cost[node] + heuristic[item[0]] + item[1]     # calculate f(n) of current node
            temp = [item[0], fn_node]
            opened.append(temp)                                     # store f(n) of current node in array opened

    '''find optimal sequence'''
    trace_node = 'G'                        # correct optimal tracing node, initialize as node G
    optimal_sequence = ['G']                # optimal node sequence
    for i in range(len(closed)-2, -1, -1):
        check_node = closed[i][0]           # current node
        if trace_node in [children[0] for children in tree[check_node]]:
            children_costs = [temp[1] for temp in tree[check_node]]
            children_nodes = [temp[0] for temp in tree[check_node]]

            '''check whether h(s) + g(s) = f(s). If so, append current node to optimal sequence
            change the correct optimal tracing node to current node'''
            if cost[check_node] + children_costs[children_nodes.index(trace_node)] == cost[trace_node]:
                optimal_sequence.append(check_node)
                trace_node = check_node
    optimal_sequence.reverse()              # reverse the optimal sequence

    return closed, optimal_sequence


if __name__ == '__main__':
    visited_nodes, optimal_nodes = AStarSearch()
    print('visited nodes: ' + str(visited_nodes))
    print('optimal nodes sequence: ' + str(optimal_nodes))