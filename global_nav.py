import copy
from scipy.spatial import distance


## this function determine the best path
#INPUT (Robot position, goal position, list of all the possible path)
#OUTPUT (list of the best points to follow in sort to reach the goal)
def dijstra(posRobot, posGoal, pp):

    possiblePath = copy.deepcopy(pp)

    #list of node to visited initialization
    node_to_visited = [posRobot]
    
    #graph initialization
    #graph = {node_i : [[node reachable from node_i], [best list of node to reach node_i from the robot], distance from robot to node_i, node visited]}
    graph = {}
    for el in possiblePath:
        if el[0] == posRobot:
            graph[el[0]] = [el[1], [], 0, False]
            continue
        graph[el[0]] = [el[1], [], 999999, True]

    
    # Dijstra algorithm
    while (len(node_to_visited) != 0): 
        noeud_actuel = node_to_visited[0]
        node_to_visited.remove(noeud_actuel)
        for noeudAtteignable in graph[noeud_actuel][0]:
            d = distance.euclidean(noeud_actuel, noeudAtteignable)
            if noeud_actuel != posRobot:
                graph[noeudAtteignable][0].remove(noeud_actuel)  
            if  graph[noeud_actuel][2]+ d< graph[noeudAtteignable][2]:
                graph[noeudAtteignable][1] = graph[noeud_actuel][1] + [noeud_actuel]
                graph[noeudAtteignable][2] = graph[noeud_actuel][2] + d
                if graph[noeudAtteignable][3]:
                    node_to_visited.append(noeudAtteignable)
                    graph[noeudAtteignable][3] = False

    return(graph[posGoal][1] + [posGoal])   

