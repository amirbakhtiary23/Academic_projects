from internet import Internet#py_wikiracer.internet import Internet
from typing import List
from html.parser import HTMLParser
import re
class Parser:

    @staticmethod
    def get_links_in_page(html: str) -> List[str]:
        """
        In this method, we should parse a page's HTML and return a list of links in the page.
        Be sure not to return any link with a DISALLOWED character.
        All links should be of the form "/wiki/<page name>", as to not follow external links.

        To do this, you can use str.find, regex, or you can
            instantiate your own subclass of HTMLParser in this function and feed it the html.
        """
        links = []
        disallowed = Internet.DISALLOWED

        # YOUR CODE HERE
        flag=True
        # Make sure your list doesn't have duplicates. Return the list in the same order as they appear in the HTML.
        # You can define your subclass of HTMLParser right here inside this function, or you could do it outside of this function.
        urls = re.findall(r'href=[\'"]?([^\'" >]+)', html)
        for i in urls:
            if i[:6] != "/wiki/":
                continue
            url=i[6:] 
            #print (i)
            for j in disallowed:
                if j in url :flag=False
                
            if i not in links and flag :
                #print (i)
                #print (i.split("org"))
                links.append (i)
            flag=True
                
        # This function will be mildly stress tested.

        return links

# In these methods, we are given a source page and a goal page, and we should return
#  the shortest path between the two pages. Be careful! Wikipedia is very large.

# These are all very similar algorithms, so it is advisable to make a global helper function that does all of the work, and have
#  each of these call the helper with a different data type (stack, queue, priority queue, etc.)

class BFSProblem:
    def __init__(self, internet: Internet,limit=1000):
        self.internet = internet
        self.limit=limit
    # Example in/outputs:
    #  bfs(source = "/wiki/Computer_science", goal = "/wiki/Computer_science") == ["/wiki/Computer_science", "/wiki/Computer_science"]
    #  bfs(source = "/wiki/Computer_science", goal = "/wiki/Computation") == ["/wiki/Computer_science", "/wiki/Computation"]
    # Find more in the test case file.

    # Do not try to make fancy optimizations here. we depend on you following standard BFS and will check all of the pages you download.
    # Links should be inserted into the queue as they are located in the page, and should be obtained using Parser's get_links_in_page.
    # Be very careful not to add things to the "visited" set of pages too early. You must wait for them to come out of the queue first.
    #  This applies for bfs, dfs, and dijkstra's.
    # Download a page with self.internet.get_page().
    def bfs(self, source = "/wiki/Calvin_Li", goal = "/wiki/Wikipedia"):
        path = [source]
        visited=set()
        queue=[]
        visited.add(source)
        queue.append(source)
        graph={source:None}
        flag=False
        counter=0
        while len(queue)!=0 and counter !=self.limit:
            
            counter+=1
            vertex = queue.pop(0)
            html = self.internet.get_page(vertex)
            links=Parser.get_links_in_page(html)
            if goal in links : 
                flag=True
                path.append(goal)
                parent = vertex
                while parent != source and parent!=None:
                    path.insert(-1,parent)
                    parent = graph[vertex]
                break
            
            for neighbour in links:
                if flag:break
                if neighbour not in visited:
                    html = self.internet.get_page(neighbour)
                    links_child=Parser.get_links_in_page(html)
                    
                    if goal in links_child:
                        flag=True
                        path.append(goal)
                        parent = neighbour
                        while parent != source and parent!=None:
                            path.insert(-1,parent)
                            parent = graph[vertex]
                        queue.clear()
                        break
        # ...        
        if flag :return path # if no path exists, return None
        return None
class DFSProblem:
    def __init__(self, internet: Internet,limit=100):
        self.internet = internet
        self.limit=limit
        self.graph={}
    # Links should be inserted into a stack as they are located in the page. Do not add things to the visited list until they are taken out of the stack.
    def dfs(self, source = "/wiki/Calvin_Li", goal = "/wiki/Wikipedia"):
        path = [source]
        # YOUR CODE HERE
        self.visited=[]
        queue=[]
        self.graph[source]=None        
        result=self.DFSUtil(source,goal,0)
        if result:
            parent = self.graph[goal]
            while parent != source and parent!=None:

                path.insert(-1,parent)
                parent = self.graph[parent]
            path.reverse()
            path.append(goal)
            return path # if no path exists, return None
        return None
    def DFSUtil(self, vertex,goal,depth):
            if depth>self.limit:
                return None
            depth=depth+1
            self.visited.append(vertex)
            # Recur for all the vertices
            # adjacent to this vertex
            html = self.internet.get_page(vertex)
            links=Parser.get_links_in_page(html)
            links.reverse()

            if goal in links:
                
                self.graph[goal]=vertex
                return True
            for neighbour in links:
                
                if neighbour not in self.visited:
                    result = self.DFSUtil(neighbour,goal,depth)
                    if result : 
                        self.graph[neighbour]=vertex
                        return True
            return None
class DijkstrasProblem:
    def __init__(self, internet: Internet,limit=10000):
        self.limit=limit
        self.internet = internet
    # Links should be inserted into the heap as they are located in the page.
    # By default, the cost of going to a link is the length of a particular destination link's name. For instance,
    #  if we consider /wiki/a -> /wiki/ab, then the default cost function will have a value of 8.
    # This cost function is overridable and your implementation will be tested on different cost functions. Use costFn(node1, node2)
    #  to get the cost of a particular edge.
    # You should return the path from source to goal that minimizes the total cost. Assume cost > 0 for all edges.
    def dijkstras(self, source = "/wiki/Calvin_Li", goal = "/wiki/Wikipedia", costFn = lambda x, y: len(y),goals_links=[]):
        path = [source]
        # YOUR CODE HERE
        counter=0
        flag=False
        visited=[]
        temp_links={}
        priority_queue=[[0,source,None]]
        #cost_parent=0
        parent_temp=source
        while counter<self.limit and len (priority_queue)>0:

            cost_parent,vertex,parent=min(priority_queue)

            priority_queue.remove([cost_parent,vertex,parent])
            if vertex in visited:
                counter+=1
                continue
            if parent  :
                try: 
                    while path[-1]!=parent :path.pop()
                except :
                    return None
                path.append(vertex)
            visited.append (vertex)
            
            html = self.internet.get_page(vertex)
            links=Parser.get_links_in_page(html)
            if goal in links :
                flag=True
                break
            parent_temp=vertex
            for j in links:
                #print (j)
                """if j in goals_links:
                    cost_child=cost_parent+(costFn(vertex,j)/2)"""
                cost_child=cost_parent+costFn(vertex,j)

                priority_queue.append([cost_child,j,vertex])
            counter+=1
        if flag:
            path.append(goal)

            return path # if no path exists, return None
        return None





class WikiracerProblem:
    def __init__(self, internet: Internet):
        self.internet = internet
        self.djk=DijkstrasProblem_wikiracer(internet)
        self.dfs=DFSProblem(internet)

    # Using what you know, try to efficiently find the shortest path between two wikipedia pages.
    # Your only goal here is to minimize the total amount of pages downloaded from the Internet, as that is the dominating time-consuming action.

    # One possible starting place is to get the links in `goal`, and then search for any of those from the source page, hoping that those pages lead back to goal.

    # Note: a BFS implementation with no optimizations will not get credit, and it will suck.
    # You may find Internet.get_random() useful, or you may not.

    def wikiracer(self, source = "/wiki/Calvin_Li", goal = "/wiki/Wikipedia"):
        path = [source]
        # YOUR CODE HERE
        goal_links=self.internet.get_page(goal)
        links=Parser.get_links_in_page(goal_links)
        self.list_of_urls=links
        self.goal_length=len(goal)
        path = self.djk.dijkstras(source,goal,costFn=self.cost_function)
        #path=self.dfs.dfs(source,goal)
        # ...
        if path:
            path.pop()
            path.append(goal)
            return path # if no path exists, return None
        return None
    def cost_function ( self,y,url):
        length=len(url)
        #sum=0
        """for i in self.list_of_urls:
            sum+=(lcs(i,url))"""
        return (length-self.goal_length)**2

class FindInPageProblem:
    def __init__(self, internet: Internet):
        self.internet = internet
    # In this, we give you a source page, and then ask you to make up some heuristics that will allow you to efficiently
    #  find a page containing all of the words in `query`. Again, optimize for the fewest number of internet downloads, not for the shortest path.

    def find_in_page(self, source = "/wiki/Calvin_Li", query = ["ham", "cheese"]):

        raise NotImplementedError("Karma method find_in_page")

        path = [source]

        # find a path to a page that contains ALL of the words in query in any place within the page
        # path[-1] should be the page that fulfills the query.
        # YOUR CODE HERE

        return path # if no path exists, return None

class DijkstrasProblem_wikiracer:
    def __init__(self, internet: Internet,limit=10000):
        self.limit=limit
        self.internet = internet
    def dijkstras(self, source = "/wiki/Calvin_Li", goal = "/wiki/Wikipedia", costFn = lambda x, y: len(y),goals_links=[]):
        html = self.internet.get_page(goal)
        self.goals_links=Parser.get_links_in_page(html)
        path = [source]
        # YOUR CODE HERE
        counter=0
        flag=False
        visited=[]
        temp_links={}
        priority_queue=[[0,source,None]]
        parent_temp=source
        while counter<self.limit and len (priority_queue)>0:

            cost_parent,vertex,parent=min(priority_queue)
            priority_queue.remove([cost_parent,vertex,parent])
            if vertex in visited:
                counter+=1
                continue
            if parent  :
                try: 
                    while path[-1]!=parent :path.pop()
                except :
                    print (parent,"parnettt")
                    print (path)
                    return None
                path.append(vertex)
            visited.append (vertex)

            html = self.internet.get_page(vertex)
            links=Parser.get_links_in_page(html)
            if goal in links :
                flag=True
                break
            parent_temp=vertex
            for j in links:

                cost_child=cost_parent+self.cost_function(j)

                priority_queue.append([cost_child,j,vertex])
            counter+=1
            #print (vertex)
        if flag:
            path.append(goal)

            return path # if no path exists, return None
        return None
    def cost_function(self,node_url):
        #print (node_url)
        splitted= node_url[6:].split()
        if len(node_url)<8:
            return 0
        temp = " ".join(self.goals_links)
        cost = 0
        for i in splitted:
            cost=temp.count(i)+cost
        return -cost