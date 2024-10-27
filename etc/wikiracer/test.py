from wikiracer import Parser
from internet import Internet
internet=Internet()
def dijkstras(source = "/wiki/Calvin_Li", goal = "/wiki/Wikipedia", costFn = lambda x, y: len(y)):
    path = [source]
    # YOUR CODE HERE
    counter=0
    flag=False
    visited=[]
    graph={source:None}
    priority_queue=[(source,0)]
    #cost_parent=0
    while counter<10000:
        vertex,cost_parent=min(priority_queue)
        if vertex in visited:
            counter+=1
            continue
        costs=[]
        visited.append (vertex)
        

        html =internet.get_page(vertex)
        links=Parser.get_links_in_page(html)
        if goal in links :
            flag=True
            graph[goal]=vertex
            #goal found
            break
        for j in links:
            cost_child=cost_parent+costFn(j)
            costs.append (costFn(j)+cost_child)
        costs=list(map(list,zip(costs,links)))
        #costs=sorted(costs)
        for i in costs:
            graph[i[1]]=vertex
            priority_queue.append((i))
        counter+=1
        
        #links.reverse()
    # ...
    if flag:
        parent = graph[goal]
        while parent != source and parent!=None:
            print (parent)
            path.insert(-1,parent)
            parent = graph[parent]

        path.append(goal)

        return path # if no path exists, return None
    return None