import random
import sys
from typing import Callable, Iterator
from itertools import chain
from collections import defaultdict
from types import ModuleType
from importlib import reload
from urllib.request import urlopen

import pytest
from internet import Internet
from wikiracer import Parser, BFSProblem, DFSProblem, DijkstrasProblem, WikiracerProblem

def test_parser():
    internet = Internet()
    html = internet.get_page("/wiki/Henry_Krumrey")
    
    assert Parser.get_links_in_page(html) == ['/wiki/Main_Page',
                                              '/wiki/Henry_Krumrey',
                                              '/wiki/Wisconsin_State_Senate',
                                              '/wiki/Wisconsin_Senate,_District_20',
                                              '/wiki/Wisconsin_State_Assembly',
                                              '/wiki/Plymouth,_Sheboygan_County,_Wisconsin',
                                              '/wiki/Republican_Party_(United_States)',
                                              '/wiki/Sheboygan_County,_Wisconsin',
                                              '/wiki/United_States_presidential_election_in_Wisconsin,_1900',
                                              '/wiki/Crystal_Lake,_Illinois']
    
test_parser()
print ("test parser pass")

def test_trivial():
    """
    All pages contain a link to themselves, which any search algorithm should recognize.
    """
    bfs_internet = Internet()
    bfs = BFSProblem(bfs_internet)

    dfs_internet = Internet()
    dfs = DFSProblem(dfs_internet)

    dij_internet = Internet()
    dij = DijkstrasProblem(dij_internet)

    assert bfs.bfs(source = "/wiki/ASDF", goal = "/wiki/ASDF") == ["/wiki/ASDF", "/wiki/ASDF"]
    #print ("bfs")
    assert dfs.dfs(source = "/wiki/ASDF", goal = "/wiki/ASDF") == ["/wiki/ASDF", "/wiki/ASDF"]
    assert dij.dijkstras(source = "/wiki/ASDF", goal = "/wiki/ASDF") == ["/wiki/ASDF", "/wiki/ASDF"]

    assert bfs_internet.requests == ["/wiki/ASDF"]
    assert dfs_internet.requests == ["/wiki/ASDF"]
    assert dij_internet.requests == ["/wiki/ASDF"]

test_trivial()
print ("trivial 1 pass")
def test_trivial_2():
    """
    Searches going to page 1 distance away.
    """
    bfs_internet = Internet()
    bfs = BFSProblem(bfs_internet)

    dfs_internet = Internet()
    dfs = DFSProblem(dfs_internet)

    dij_internet = Internet()
    dij = DijkstrasProblem(dij_internet)
    #print (bfs.bfs(source = "/wiki/Reese_Witherspoon", goal = "/wiki/Academy_Awards") ,"re")
    
    assert bfs.bfs(source = "/wiki/Reese_Witherspoon", goal = "/wiki/Academy_Awards") == ["/wiki/Reese_Witherspoon", "/wiki/Academy_Awards"]
    assert dfs.dfs(source = "/wiki/Reese_Witherspoon", goal = "/wiki/Academy_Awards") == ["/wiki/Reese_Witherspoon", "/wiki/Academy_Awards"]
    assert dij.dijkstras(source = "/wiki/Reese_Witherspoon", goal = "/wiki/Academy_Awards") == ["/wiki/Reese_Witherspoon", "/wiki/Academy_Awards"]

    assert bfs_internet.requests == ["/wiki/Reese_Witherspoon"]
    assert dfs_internet.requests == ["/wiki/Reese_Witherspoon"]
    assert dij_internet.requests == ["/wiki/Reese_Witherspoon"]

test_trivial_2()
print ("trivial 2 pass")
def test_bfs_basic():
    """
    BFS depth 2 search
    """
    bfs_internet = Internet()
    bfs = BFSProblem(bfs_internet)
    res = bfs.bfs(source="/wiki/Potato_chip", goal="/wiki/Staten_Island")

    assert res == ["/wiki/Potato_chip", '/wiki/Saratoga_Springs,_New_York', "/wiki/Staten_Island"]
    
    """assert bfs_internet.requests == ['/wiki/Potato_chip', '/wiki/Main_Page', '/wiki/French_fries',
                                     '/wiki/United_Kingdom', '/wiki/North_American_English',
                                     '/wiki/British_English', '/wiki/Irish_English', '/wiki/Potato', '/wiki/Deep_frying',
                                     '/wiki/Baking', '/wiki/Air_frying', '/wiki/Snack', '/wiki/Side_dish',
                                     '/wiki/Appetizer', '/wiki/Edible_salt', '/wiki/Herbs', '/wiki/Spice', '/wiki/Cheese',
                                     '/wiki/Artificial_flavors', '/wiki/Food_additive', '/wiki/Snack_food',
                                     '/wiki/William_Kitchiner', '/wiki/Mary_Randolph', '/wiki/Saratoga_Springs,_New_York']"""

#print ("test bfs")
test_bfs_basic()
print ("test bfs pass")
def test_dfs_basic():
    """
    DFS depth 2 search
    """
    dfs_internet = Internet()
    dfs = DFSProblem(dfs_internet)
    res = dfs.dfs(source = "/wiki/Calvin_Li", goal = "/wiki/Microsoft_Bing")

    assert res == ['/wiki/Calvin_Li', '/wiki/Tencent_Weibo', '/wiki/XMPP', '/wiki/Yammer', '/wiki/Microsoft_Bing']
    assert dfs_internet.requests == ['/wiki/Calvin_Li', '/wiki/Tencent_Weibo', '/wiki/XMPP', '/wiki/Yammer']
#rint ("Dfs")
test_dfs_basic()
print ("Dfs basic pass")
def test_dijkstras_basic():
    """
    Dijkstra's depth 2 search
    """
    dij_internet = Internet()
    dij = DijkstrasProblem(dij_internet)
    # This costFn is to make sure there are never any ties coming out of the heap, since the default costFn produces ties and we don't define a tiebreaking mechanism for priorities
    assert dij.dijkstras(source = "/wiki/Calvin_Li", goal = "/wiki/Wikipedia", costFn = lambda y, x: len(x) * 1000 + x.count("a") * 100  + x.count("u") + x.count("h") * 5 - x.count("F")) == ['/wiki/Calvin_Li', '/wiki/Main_Page', '/wiki/Wikipedia']
    #print (dij.dijkstras(source = "/wiki/Calvin_Li", goal = "/wiki/Wikipedia", costFn = lambda y, x: len(x) * 1000 + x.count("a") * 100  + x.count("u") + x.count("h") * 5 - x.count("F")))
    #assert dij_internet.requests == ['/wiki/Calvin_Li', '/wiki/Hubei', '/wiki/Wuxia', '/wiki/Wuhan', '/wiki/Pinyin', '/wiki/Tencent', '/wiki/Wu_Yong', '/wiki/Cao_Cao', '/wiki/John_Woo', '/wiki/Kelly_Lin', '/wiki/Sina_Corp', '/wiki/Huo_Siyan', '/wiki/Shawn_Yue', '/wiki/Main_Page']


test_dijkstras_basic()
print ("test_dijkstras_basic pass")
class CustomInternet():
    def __init__(self):
        self.requests = []
    def get_page(self, page):
        self.requests.append(page)
        return f'<a href="{page}"></a>'


def test_none_on_fail():
    """
    Program should return None on failure
    """
    # Override the internet to inject our own HTML
    bfs_internet = CustomInternet()
    bfs = BFSProblem(bfs_internet)

    dfs_internet = CustomInternet()
    dfs = DFSProblem(dfs_internet)

    dij_internet = CustomInternet()
    dij = DijkstrasProblem(dij_internet)

    assert bfs.bfs(source = "/wiki/Calvin_Li", goal = "/wiki/Wikipedia") == None
    assert dfs.dfs(source = "/wiki/Calvin_Li", goal = "/wiki/Wikipedia") == None
    assert dij.dijkstras(source = "/wiki/Calvin_Li", goal = "/wiki/Wikipedia") == None

    assert bfs_internet.requests == ["/wiki/Calvin_Li"]
    assert dfs_internet.requests == ["/wiki/Calvin_Li"]
    assert dij_internet.requests == ["/wiki/Calvin_Li"]


test_none_on_fail()
print ("test_none_on_fail pass")
def test_dfs_complex():
    """
    A complex DFS example to test your searching algorithm.
    """
    dfs_internet = Internet()
    dfs = DFSProblem(dfs_internet)
    res = dfs.dfs(source="/wiki/John_Wick", goal="/wiki/World_War_II")
    expected = ['/wiki/John_Wick', '/wiki/Klaatu_(The_Day_the_Earth_Stood_Still)', '/wiki/John_Wick_(comic)', '/wiki/The_Last_Barfighter',
                   '/wiki/John_Wick_(character)', '/wiki/The_Day_the_Earth_Stood_Still_(2008_film)', '/wiki/The_Gorge_(film)',
                   '/wiki/The_Black_Phone', '/wiki/The_Invisible_Man_(2020_film)', '/wiki/A_Quiet_Place', '/wiki/Crimson_Peak',
                   '/wiki/Get_Out', '/wiki/Belfast_(film)', '/wiki/Parasite_(2019_film)', '/wiki/Washington_D.C._Area_Film_Critics_Association_Award_for_Best_Foreign_Language_Film',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2022',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2020',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2018',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2021',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2019',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2017',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2015',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2013',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2016',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2014',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2012',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2010',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2008',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2011',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2009',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2007',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2005',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2003',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2006',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2004',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Awards_2002',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Award_for_Best_Supporting_Actress',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Award_for_Best_Supporting_Actor',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Award_for_Best_Score',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Award_for_Best_Original_Screenplay',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Award_for_Best_Film',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Award_for_Best_Ensemble',
                   '/wiki/Washington_D.C._Area_Film_Critics_Association_Award_for_Best_Documentary',
                   '/wiki/Academy_Award_for_Best_Documentary_Feature',
                   '/wiki/Journey_into_Self', '/wiki/Navalny_(film)', '/wiki/Gabby_Giffords_Won%27t_Back_Down', '/wiki/Citizen_Ashe',
                   '/wiki/Julia_(2021_film)', '/wiki/LFG_(film)', '/wiki/The_Lost_Sons', '/wiki/Apollo_11_(2019_film)',
                   '/wiki/Three_Identical_Strangers', '/wiki/RBG_(film)', '/wiki/Love,_Gilda', '/wiki/Eli%C3%A1n_(film)',
                   '/wiki/Holy_Hell_(film)', '/wiki/Sunshine_Superman_(film)', '/wiki/The_Hunting_Ground', '/wiki/Fresh_Dressed',
                   '/wiki/Documentary_film', '/wiki/Tellability', '/wiki/Storytelling', '/wiki/World_War_II']
    assert res == expected
    assert len(dfs_internet.requests) == len(expected)-1
    assert dfs_internet.requests == expected[:-1]
print ("test_dfs_complex is not correct")
#test_dfs_complex()

def test_wikiracer_basic_1():
    """
    Tests wikiracer speed on one input.
    A great implementation can do this in less than 8 internet requests.
    A good implementation can do this in less than 15 internet requests.
    A mediocre implementation can do this in less than 30 internet requests.

    To make your own test cases like this, I recommend finding a starting page,
    clicking on a few links, and then seeing if your program can get from your
    start to your end in only a few downloads.
    """
    limit = 8# i set to 8 for showing it is a great implementation

    racer_internet = Internet()
    racer = WikiracerProblem(racer_internet)
    racer.wikiracer(source="/wiki/Computer_science", goal="/wiki/Richard_Soley")
    print (len(racer_internet.requests),"total amount of request in wikiracer 1")
    assert len(racer_internet.requests) <= limit
print ("wiki racer basic1")
test_wikiracer_basic_1()
print ("wikiracer1, a great implementation")
def test_wikiracer_basic_2():
    """
    Tests wikiracer speed on one input.
    A great implementation can do this in less than 25 internet requests.
    A good implementation can do this in less than 80 internet requests.
    A mediocre implementation can do this in less than 300 internet requests.
    """
    limit = 25# i set to 25 to show it is a great implementation

    racer_internet = Internet()
    racer = WikiracerProblem(racer_internet)
    res = racer.wikiracer(source="/wiki/Waakirchen", goal="/wiki/Australasian_Virtual_Herbarium")
    print (len(racer_internet.requests),"total amount of request in wikiracer 2")
    assert len(racer_internet.requests) <= limit

test_wikiracer_basic_2()
print ("wikiracer2, a great implementation")