from cmath import sqrt
from re import I
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import random
from multiprocessing import  Process
import scipy.integrate as integrate
import scipy.special as special
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from numpy import linalg as LA
from numpy.linalg import inv
from statistics import pvariance
import time
from scipy.linalg import fractional_matrix_power
import statistics as stat

class coordinate: #class for coordinate
    def __init__(self,x,y):
        self.coordinate = [x,y]
    
    def Random_Process(self,SD): 
        self.coordinate[0] = self.coordinate[0]+np.random.normal(0,SD)
        self.coordinate[1] = self.coordinate[1]+np.random.normal(0,SD)

    def x(self):
        return self.coordinate[0]
    
    def y(self):
        return self.coordinate[1]

    def xto(self,x):
        self.coordinate[0] = x
    
    def yto(self,y):
        self.coordinate[1] = y

def distance_origin(node ,height ,width):
    x = node.x()
    y = node.y()
    if node.x() > width/2:
        x = x - width
    if  node.y() > height/2:
        y = y - height
    
    return (x**2) + (y**2)

def Distance_Hard(node1 , node2 , R , height , width): #calculate distance between 2 nodes in torus , if distance is greater than R ,return -1
    R  = R + 10**-6
    if node1.x() + R > width  and  node2.x() - R < 0 or node2.x() + R > width  and  node1.x() - R < 0: 
        #check whether both circle centered at node1,2 passed the vertical border
        #we want to fixed the point at right
        if node1.x() + R > width:
            LeftNode = node2
            RightNode = node1
        else:
            LeftNode = node1
            RightNode = node2 
        #check whether both circle centered at node1,2 passed the horizontal border
        if node1.y() + R > height  and  node2.y() - R < 0 or node2.y() + R > height  and  node1.y() - R < 0:
            #move both x and y
            if LeftNode.y() + R > height : #it's on the upper left ,we add length to x and minus length to y
                res =  np.sqrt((LeftNode.x()+width - RightNode.x())**2 +  (LeftNode.y() - height - RightNode.y())**2)
            else:#it's on the lower left we add length to both x and y
                res  = np.sqrt((LeftNode.x()+width - RightNode.x())**2 +  (LeftNode.y() + height - RightNode.y())**2)
        else: # it's just on left we move add length to x
            res =  np.sqrt((LeftNode.x()+width - RightNode.x())**2 +  (LeftNode.y() - RightNode.y())**2)
    elif  node1.y() + R > height  and  node2.y() - R < 0 or node2.y() + R > height  and  node1.y() - R < 0: #check y conditions
        #we want upper node be fixed
        if node1.y() + R > height:
            UpperNode = node1
            LowerNode = node2
        else:
            UpperNode = node2
            LowerNode = node1 
        res =  np.sqrt((UpperNode.x() - LowerNode.x())**2 +  (LowerNode.y()+height - UpperNode.y())**2)
    else:
        res = np.sqrt((node1.x() - node2.x())**2 +  (node1.y() - node2.y())**2)
    #check if distance is less than R
    return res 


def connect(node1 , node2 , R , height , width):
    if node1.x() == node2.x() and node1.y() == node2.y():
        return -1
    dis = Distance_Hard(node1 , node2 , R , height , width)
    p = R*dis**(-3)
    foo = random.random()
    if foo > p:
        return -1
    else:
        return 1

def clear_isolated_nodes(x):
    res = [] 
    isolated = []
    for i in range(len(x)):
        if x[i][i] == 0:
            isolated.append(i) # find isolated nbodes
    if len(isolated) == 0:
        return x 
    pointer_i = 0
    for i in range(len(x)):
        temp = []
        if i != isolated[pointer_i]:
            pointer_j = 0
            for j  in range(len(x)):
                if j != isolated[pointer_j]:
                    temp.append(x[i][j])
                elif pointer_j < len(isolated)-1:
                    pointer_j += 1
            res.append(temp)
        elif i == isolated[pointer_i]:
            if pointer_i < len(isolated) -1 :
                pointer_i = pointer_i+1
    
    return res

class Graph:
    def __init__(self , shape , R , SD , height , width):
        self.graph = []
        self.AdjacencyMatrix = []
        self.shape = shape
        self.R = R
        self.SD = SD
        self.height = height
        self.width = width
        self.length = height*width

        if self.shape == "R":
            for i in range(height):
                for j in range(width):
                    self.graph.append(coordinate(i+1/2,j+1/2))

        if self.shape == "T":
            for i in range (int(height/(np.sqrt(3)/2))):
                for j in range(width):
                    if i%2 == 0:
                        self.graph.append(coordinate(j+1/2,(1/2)*(np.sqrt(3)/2) +(np.sqrt(3)/2)*i))
                    else:
                        self.graph.append(coordinate(j+1,(1/2)*(np.sqrt(3)/2) +(np.sqrt(3)/2)*i))
        
        for i in range(len(self.graph)): #poisson process

            self.graph[i].Random_Process(self.SD) # take random movement for each node

            if self.graph[i].x() > self.width : # make sure we are working in a torus
                self.graph[i].xto(self.graph[i].x() - self.width) # checking conditions for x
            elif self.graph[i].x() < 0 :
                self.graph[i].xto(self.width + self.graph[i].x())
            
            if self.graph[i].y() > self.height : # checking conditions for y
                self.graph[i].yto(self.graph[i].y() - self.height) 
            elif self.graph[i].y() < 0 :
                self.graph[i].yto(self.height + self.graph[i].y())

        
        
        self.AdjacencyMatrix = [[0]*self.length for i in range(self.length)] #initialize
            
     
        for i in range(len(self.graph)):
            for j in range(i,len(self.graph)):
                if connect(self.graph[i],self.graph[j],self.R,self.height,self.width) == 1:
                    self.AdjacencyMatrix[i][j] = 1
                    self.AdjacencyMatrix[j][i] = 1                    
                else:
                    self.AdjacencyMatrix[i][j] = 0
                    self.AdjacencyMatrix[j][i] = 0

        self.A = np.matrix(self.AdjacencyMatrix)

        self.DegreeDistribution =  [sum(x) for x in self.AdjacencyMatrix]
        self.D = np.diag(self.DegreeDistribution)
        self.G = nx.from_numpy_matrix(self.A)
        


        #A = np.matrix(clear_isolated_nodes(adjencymatrix))
    
    def Is_Connect(self,x,y):
        if self.AdjacencyMatrix[x][y] == 1 :
            return True
        return False


    def uniform_random(self):
        for i in range(self.height * self.width):
                self.graph[i] = coordinate(random.uniform(0,self.height),random.uniform(0,self.width))

    def Print_Graph(self): #method for checking
        res = ""
        for i in range(len(self.graph)):
            res = res + str(self.graph[i].coordinate) + ","
        print(res)
        return res
    
    def Get_XY_List(self): #get a list contains 2 lists, 1st list stores x values 2nd list stores y values, used for plotting
        X = []
        Y = []
        for i in range(len(self.graph)):
            X.append(self.graph[i].x())
            Y.append(self.graph[i].y())
        return [X,Y]

    def Plot_Graph(self, loca,max): # plot the graph
        fig, ax = plt.subplots()
        plt.plot(self.Get_XY_List()[0],self.Get_XY_List()[1],".",markersize = 3)

        # for i in range(len(self.graph)):
        #     for k in range(len(self.graph)):
        #         if self.AdjacencyMatrix[i][k] == 1:
        #             plt.plot([self.graph[i].x(),self.graph[k].x()],[self.graph[i].y(),self.graph[k].y()],color = 'black',linewidth=0.5)
        
        

        while(len(max) != 0):
            location = max.pop()
            x = [self.graph[location].x()]
            y = [self.graph[location].y()]
            plt.plot(x,y,marker = '*', color = "Green",markersize = 8)
        
        while(len(loca) != 0):
            location = loca.pop()
            x = [self.graph[location].x()]
            y = [self.graph[location].y()]
            plt.plot(x,y,'.', color = "red",markersize = 3)

        ax.set_aspect('equal', adjustable='box')
        plt.savefig("E:/Grr Project/max degree/max_loca.png",dpi=400)

    def Random_Movement(self):
        for i in range(len(self.graph)):

            self.graph[i].Random_Process(self.SD) # take random movement for each node

            if self.graph[i].x() > self.width : # make sure we are working in a torus
                self.graph[i].xto(self.graph[i].x() - self.width) # checking conditions for x
            elif self.graph[i].x() < 0 :
                self.graph[i].xto(self.width + self.graph[i].x())
            
            if self.graph[i].y() > self.height : # checking conditions for y
                self.graph[i].yto(self.graph[i].y() - self.height) 
            elif self.graph[i].y() < 0 :
                self.graph[i].yto(self.height + self.graph[i].y())

    def Find_Components(self):
        #not complete yet
        #loop until there is no more element in original list
        Avaliable_Nodes = self.graph
        current_Node = Avaliable_Nodes.pop(0)
        current_queue = []
        To_Sub = []
        res = []
        current_component = []

        #loop for first time to enter while
        for i in range(len(Avaliable_Nodes)):
            if connect(current_Node,Avaliable_Nodes[i] , self.R ,self.length) != -1:
                current_queue.append(Avaliable_Nodes[i])
            else :
                To_Sub.append(Avaliable_Nodes[i])
        current_component.append(current_Node)
        Avaliable_Nodes =  To_Sub
        if len(current_queue) == 0 :
            current_Node = Avaliable_Nodes.pop(0) 
            res.append(current_component)   
            current_component = []
        else :
            current_Node = current_queue.pop(0)
        To_Sub = []


        while len(Avaliable_Nodes) != 0 :
            while True:
                for i in range(len(Avaliable_Nodes)):#loop through the current node, find nodes within R
                    if connect(current_Node,Avaliable_Nodes[i], self.R ,self.length) != -1:
                        current_queue.append(Avaliable_Nodes[i])
                    else :
                        To_Sub.append(Avaliable_Nodes[i])
                current_component.append(current_Node)
                if len(current_queue) != 0 :
                    current_Node = current_queue.pop(0) 
                Avaliable_Nodes =  To_Sub
                To_Sub = []
                if len(current_queue) == 0:
                    break
            if len(Avaliable_Nodes) != 0:
                current_Node = Avaliable_Nodes.pop(0)
            res.append(current_component)
            current_component = []
        return res
    
    def largest_component(self):
        largest_cc = max(nx.connected_components(self.G), key=len)
        return len(largest_cc)
    
    def To_Graph(self): #to generate connected networkx graph based on current graph 
        res = nx.from_numpy_matrix(self.A)
        return res
    
    def mean_degree(self):
        res = 0
        for i in range(len(self.graph)):
            D = distance_origin(self.graph[i],self.height,self.width)
            res += P(self.R,D,self.SD) + Q(self.R,D,self.SD)    
        return res
    
    def average_degree(self):
        res = (np.trace(self.D))/(self.height* self.width)
        return res 
    
    def laplacian_matrix(self):
        laplacian = []
        degreematrix = []
        adjencymatrix = []
        for i in range(len(self.graph)):
            temp1 = []
            temp2 = []
            temp3 = []
            degree = 0
            for j in range(len(self.graph)):
                if connect(self.graph[i],self.graph[j],self.R,self.height,self.width) == 1:
                    temp1.append(-1)
                    temp3.append(1)
                    degree += 1
                else:
                    temp1.append(0)
                    temp3.append(0)
                temp2.append(0) 
            temp1[i] = temp1[i] + degree
            temp2[i] = degree 
            laplacian.append(temp1)
            degreematrix.append(temp2)
            adjencymatrix.append(temp3)
        lap = np.matrix(clear_isolated_nodes(laplacian))
        A = np.matrix(degreematrix)
        D = np.matrix(clear_isolated_nodes(degreematrix))  
        #A = np.matrix(clear_isolated_nodes(adjencymatrix))
        #lap = D-A
        D = fractional_matrix_power(D, -0.5)
        foo = np.matmul(D,lap)
        res = np.matmul(foo,D)
        return res
    
    
    
    #def adjecency_matrix(self):
        adjencymatrix = []
        for i in range(len(self.graph)):
            temp3 = []
            for j in range(len(self.graph)):
                if connect(self.graph[i],self.graph[j],self.R,self.height,self.width) == 1:
                    temp3.append(1)
                else:
                    temp3.append(0)
            adjencymatrix.append(temp3)
        A = np.matrix(adjencymatrix)
        #A = np.matrix(clear_isolated_nodes(adjencymatrix))
        return A
    
    def variance_degree(self):
        degree = 0
        res = 0
        degree_list = []
        for i in range(len(self.graph)):
            for j in range(len(self.graph)):
                if connect(self.graph[i],self.graph[j],self.R,self.height,self.width) == 1:
                    degree +=1 
            degree_list.append(degree - 1)
            degree = 0
        res = pvariance(degree_list)
        return res 

    
                
def clustering_coefficient(graph): #calculate the clustering coe
    foo = graph.To_Graph()
    return nx.average_clustering(foo)

def largest_component(graph):
    G = graph.To_Graph()
    return len(max(nx.connected_components(G), key=len))


def average_degree(graph):
    G =graph.To_Graph()
    return nx.average_degree_connectivity(G)

def P(R,Lambda,SD):
    var = SD**2
    e = np.exp(1)
    respair = integrate.quad(lambda t:special.iv(0,sqrt(Lambda*t)/(sqrt(2)*SD))*e**-1*(t + Lambda/2*var),0,R**2/4*var)
    res = 0.5*respair[0]
    return res

def Q(R,Lambda,SD):
    var = SD**2
    e = np.exp(1)
    upper_limit = np.Infinity
    r = R**(-(1/3))
    lower_limit = (r**((-1)*(2/3))/(2*var))
    head = 1/((2**(5/2))*(SD**3))
    respair = integrate.quad(lambda t:special.iv(0,sqrt(Lambda*t)/(sqrt(2)*SD))*(e**((-0.5)*(t+(Lambda/(2*var)))))*(t**((-3/2))),lower_limit,upper_limit)
    res = head*respair[0]
    return res


    
def largest_component_heatmap():
    data = []
    current_D = []
    average = 0
    for i in range(10):        
        for j in range(10):
            for k in range(100):
                foo = Graph("R",0.01*i,0.1*j,37,15)
                foo.Random_Movement()
                average += largest_component(foo)
            current_D.append(average/100)
            average = 0
        data.append(current_D)
        current_D = []
    label1 = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
    label2 = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    label1 = label1[::-1]
    data = data[::-1]
    df_cm = pd.DataFrame(data, index = label1,
                  columns = label2)
    sns.set()
    ax = sns.heatmap(df_cm)
    ax.set_ylabel('R')
    ax.set_xlabel('SD')
    ax.set_title('largest component 37*15 soft connection')
    print(str(data))
    plt.show()

def clustering_heatmap():
    data = []
    current_D = []
    average = 0
    for i in range(10):        
        for j in range(10):
            for k in range(100):
                foo = Graph("T",0.7+0.1*i,0.1*j,(np.sqrt(3)/2)*10,10)
                foo.Random_Movement()
                average += clustering_coefficient(foo)
            current_D.append(average/100)
            average = 0
        data.append(current_D)
        current_D = []
    label1 = [0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6]
    label2 = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    label1 = label1[::-1]
    data = data[::-1]
    df_cm = pd.DataFrame(data, index = label1,
                  columns = label2)
    sns.set()
    ax = sns.heatmap(df_cm)
    ax.set_ylabel('R')
    ax.set_xlabel('SD')
    ax.set_title('Clustering coefficeint heatmap(triangle)')
    print(str(data))
    plt.show()

def mean_degree_heatmap():
    data = []
    current_D = []
    for i in range(10):
        for j in range(10):
            foo = Graph("R",0.7+0.1*i,0.1+0.1*j,10,10)
            current_D.append(foo.mean_degree())
        data.append(current_D)
        current_D = []
    label1 = [0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6]
    label2 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    label1 = label1[::-1]
    data = data[::-1]
    df_cm = pd.DataFrame(data, index = label1,columns = label2)
    sns.set()
    ax = sns.heatmap(df_cm)
    ax.set_ylabel('R')
    ax.set_xlabel('SD')
    ax.set_title('mean degree by distribution (SOFT)')
    print(str(data))
    plt.show()



def Soft_Connect_thread(SD,height,width):
    y_value = []
    for i in range(20):#R from 0 to 2
        sum = 0
        for j in range(100):#loop 100 times
            foo = Graph("R",i*0.01,SD,height,width)
            foo.Random_Movement()
            sum += largest_component(foo)
        average = sum/100
        y_value.append(average)
    
    print("SD = " + str(SD) + ":" + str(y_value))




def thread_One(R,SD,height,width):
    sum = 0
    for j in range(100):#loop 100 times
        foo = Graph("R",R,SD,height,width)
        foo.Random_Movement()
        sum += foo.average_degree()
    average = sum/100
    
    print("SD = " + str(SD) +" R = " + str(R) + ":" + str(average))


def multi_thread_One(R): 
    process_list = []
    R_list = []
    for i in range(26):
        R_list.append(0.02 + (0.004 * i))
    print(R_list)
    for i in range(len(R_list)): 
        p = Process(target=thread_One,args=(R_list[i],R,50,50,))
        p.start()
        process_list.append(p)
    
    for i in process_list:
        p.join()
    
    print('task done')


def thread_Inf(R,SD,height,width):
    sum = 0
    for j in range(100):
        foo = Graph("R",R,SD,height,width)
        foo.uniform_random()
        sum += foo.average_degree()
    average = sum/100
    
    print("SD = " + str(SD) +" R = " + str(R) + ":" + str(average))

def multi_thread_Inf(SD): 
    process_list = []
    R_list = []
    for i in range(26):
        R_list.append(0.02 + (0.004 * i))
    print(R_list)
    for i in range(len(R_list)): 
        p = Process(target=thread_Inf,args=(R_list[i],SD,50,50,))
        p.start()
        process_list.append(p)
    
    for i in process_list:
        p.join()

def mean_degree_plot():
    R = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
    SD = [0,0.1,0.25,0.5,0.75,1.0]
    res = []
    for i in range(len(SD)):
        for j in range(len(R)):
            for k in range(50):
                bar = 0
                foo = Graph('R',R[j],SD[i],10,10)
                foo.Random_Movement()
                bar += foo.average_degree()
            bar = bar/50
            res.append(bar)
            bar = 0
        print(res)
        plt.plot(R,res)
        res = []
    plt.show()




def eigenvalue(matrix):
    res = list(eigs(matrix,k = 2499, return_eigenvectors = False))
    bar = []
    for i in range(len(res)):
        if res[i] > 10^(-5):
            bar.append(res[i])

    return bar




def eigenvalue_plot(SD,R):
    res = []
    average = 0
    for i in range (10):
        foo = Graph('R',R,SD,50,50)
        foo.Random_Movement()
        temp = eigenvalue(foo.laplacian_matrix())
        average += max(temp)
        res.extend(temp)
    print("highest average for sd " + str(SD) + " R " + str(R) + "is " + str((average/10).real))
    print("discarded " + str(24990 - len(res)))
    return res

def var_degree_plot():
    R = [0.02, 0.024, 0.028, 0.032, 0.036, 0.04, 0.044, 0.048, 0.052, 0.056, 0.06, 0.064, 0.068, 0.072, 0.076, 0.08, 0.084, 0.088, 0.0921, 0.096, 0.1, 0.104, 0.108, 0.112, 0.116, 0.12]
    SD = [1]
    res = []
    for i in range(len(SD)):
        for j in range(len(R)):
            foo = Graph('R',R[j],SD[i],50,50)
            foo.Random_Movement()
            temp = foo.variance_degree()
            print(temp)
            res.append(temp)

        print(res)
        res = []
def max_degree_12(x):
    graph = x.To_Graph()
    degree_sequence = [d for n, d in graph.degree()] 
    max1 = max(degree_sequence)
    degree_sequence.remove(max1)
    max2 = max(degree_sequence)
    return [max1,max2]

def max_degree_location(x):
    res = [] 
    degree_sequence = x.DegreeDistribution
    maxi =  max(degree_sequence)
    for i in range (len(degree_sequence)):
        if maxi == degree_sequence[i]:
            res.append(i)
    #print(degree_sequence)
    return res 



def max2 (x):
    foo = max(x)
    x.tolist().remove(foo)
    res = max(x)
    return res 

def eig_values(G):
    res = eigs(G.laplacian_matrix())
    return res

def maxdegree_thread(SD,height,width):
    y_value1 = []
    y_value2 = []
    eig1 = []
    eig2 = []
    for i in range(16):#R from 0.4 to 0.6
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        for j in range(50):#loop 50 times
            foo = Graph("R",0+i*0.01,SD,height,width)
            foo.Random_Movement()
            sum1 += max_degree_12(foo)[0]
            sum2 += max_degree_12(foo)[1]
            temp = eig_values(foo)[0]
            sum3 += max(temp)
            sum4 += max2(temp)
        average1 = sum1/50
        average2 = sum2/50
        average3 = sum3/50
        average4 = sum4/50
        y_value1.append(average1)
        y_value2.append(average2)
        eig1.append(average3)
        eig2.append(average4)
        print(i)
    
    
    print("max1 degree SD = " + str(SD) + ":" + str(y_value1))
    print("max2 degree SD = " + str(SD) + ":" + str(y_value2))
    print("max1 eigv SD = " + str(SD) + ":" + str(eig1))
    print("max2 eigv SD = " + str(SD) + ":" + str(eig2))






def multi_thread_maxdegree():
    process_list = []
    SD_list = [0,0.25,0.5,0.75,1.0]
    for i in range(5): 
        p = Process(target=maxdegree_thread,args=(SD_list[i],20,20,))
        p.start()
        process_list.append(p)
    
    for i in process_list:
        p.join()


def eig_vs_average_multi():
    process_list = []
    SD_list = [0,0.25,0.5,0.75,1.0]
    for i in range(5): 
        p = Process(target=eigvalue_average_thread,args=(SD_list[i],15,15,))
        p.start()
        process_list.append(p)
    
    for i in process_list:
        p.join()


    time.sleep(600)
    print('task done')


def eigvalue_average_thread(SD,height,width):
    mean_degree = []
    eig1 = []
    for i in range(51):#R from 0.4 to 0.6
        sum1 = 0
        sum2 = 0
        for j in range(50):#loop 50 times
            foo = Graph("R",0.1+i*0.01,SD,height,width)
            sum1 += foo.average_degree()
            bar = foo.A
            temp = list(LA.eig(bar)[0].real)       
            sum2 += max(temp)
        average1 = sum1/50
        average2 = sum2/50
        mean_degree.append(average1)
        eig1.append(average2)
        print(i)

    print("mean degree SD = " + str(SD) + ":" + str(mean_degree))
    print("leading eigenvalue SD = " + str(SD) + ":" + str(eig1))

def eigenvector_localization(x):
    adj = x.A
    eigs = LA.eig(adj)   
    eigvalue = eigs[0] #eigen values 
    max_index = np.argmax(eigvalue)   #find the index of maximum eigenvalue
    eigvec = eigs[1]  # eigen vactors 
    eigvec = eigvec[:,max_index] #select eigen vactor that correspond the max eigenvalue
    #print(eigvec)
    #eigvec = [row[i] for row in eigvec]
    eigvector = ((eigvec.flatten()).tolist())[0]   # convert to list
    #print(eigvec)
    for i in range(len(eigvec)):
        eigvector[i] = float(abs(eigvector[i])) #change type
    loca = []
    #print(eigvec)
    loca = sorted(range(len(eigvector)), key=lambda i: eigvector[i])[-10:]
    eigvector.sort(reverse = True)   
    return [eigvector,loca]

def distance_top_5_eig(R,SD,L):
    G = Graph('R',R,SD,L,L)
    EL =  eigenvector_localization(G)
    EL = EL[1]
    ML = max_degree_location(G)
    res = 1000000000
    sum = 0
    for i in range(len(ML)):
        max_loc = ML[i]
        sum = 0
        for j in range (len(EL)):
            eig_loc = EL[j]
            sum += Distance_Hard(G.graph[max_loc],G.graph[eig_loc],R,L,L)
        if sum < res :
            res = sum
    return res

def max_degree_local_plot():
    res = []
    for i in range(500):
        res.append(distance_top_5_eig(0.4,0.25,25))
    average = stat.mean(res)
    sd = stat.stdev(res)
    return [average,sd,res]

            
def largest_cc_plot():
    SD = [0,0.25,0.5,0.75,1]
    R = 0
    bar = []
    res = []
    for i in range(len(SD)):
        temp = 0
        for j in range(50):
            for k in range(100):
                foo = Graph('R',j*0.01,SD[i],25,25)
                temp += foo.largest_component()
            temp = temp/62500
            bar.append(temp)
        res.append(bar)
        bar = []
    return res
        

def Connectivity(R,SD,height,width):
    res = 0
    for i in range(100):
        foo = Graph('R',R,SD,height,width)
        if nx.is_connected(foo.G) == True:
            res += 1
    res = res/100
    return res


def connectivity_plot():
    sd = [0,0.25,0.5,0.75,1]
    res = []
    for j in range(5):
        temp = []
        for i in range(21):
            temp.append(Connectivity(0.4+0.02*i,sd[j],25,25))
        res.append(temp)
        print(j)
    return res




    


if __name__ == '__main__':
    #process_list = []
    #SD_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]
    #for i in range(20): 
        #p = Process(target=Soft_Connect_thread,args=(SD_list[i],37,15,))
        #p.start()
        #process_list.append(p)
    
    #for i in process_list:
        #p.join()

    #print('task done')

    #foo = Graph('R',0.02,0.25,50,50)
    #foo.Random_Movement()
    #print(foo.variance_degree())
    #foo = Graph('R',0,1,10,10)
    #foo.Random_Movement()
    #text_file = open("E:/data7.txt", "w")
    #text_file.write(foo.Print_Graph())
    
    #var_degree_plot()
    #foo0 = Graph('R',0.5,0.25,15,15)
    
    #foo = Graph('R',0.2,0.25,15,15)
    #bar = Graph('R',0.4,0.25,15,15)
    #bar1 = Graph('R',0.6,0.25,15,15)
    #foo0.Random_Movement()
    #foo1.Random_Movement()
    #foo.Random_Movement()
    #bar.Random_Movement()
    #bar1.Random_Movement()
    #print(foo1.AdjacencyMatrix)

    #print(foo1.adjecency_matrix())

    


    #bar = foo.laplacian_matrix()
    #temp = eigs(bar)
    #print(temp)
    #print(list(temp[0].real))
    #eigvalue_average_thread(0.5,5,5)
    #print(eigenvector_localization(foo))
    #vec0 = eigenvector_localization(foo0)
   
    #vec2 = eigenvector_localization(foo)
    #vec3 = eigenvector_localization(bar)
    #vec4 = eigenvector_localization(bar1)

    # # eig localization visulization
    # G5 =  Graph('R',0.05,0.25,25,25)
    # G2 =  Graph('R',0.2,0.25,25,25)
    # G4 = Graph('R',0.4,0.25,25,25)
    # res5 = eigenvector_localization(G5)
    # res2 = eigenvector_localization(G2)
    # res4 = eigenvector_localization(G4)
    # vec5 = res5[0]
    # vec2 = res2[0]
    # vec4 = res4[0]
    # y = []
    # for i in range(625):
    #     y.append(i)
    # plt.plot(y,vec5,y,vec2,y,vec4)
    # plt.legend(["R = 0.05","R = 0.2","R = 0.4"])
    # plt.xlabel('i')
    # plt.ylabel('φ1i')
    # plt.title('Ordered components of the leading eigenvector')
    # plt.show()
    # loca = res2[1]
    # max = max_degree_location(G2)
    # G2.Plot_Graph(loca,max)

    #print(max_degree_local_plot())

    #print(largest_cc_plot())

    #foo = Graph('R',0.2,0.25,5,5)
    #print(foo.largest_component())
    #foo.Plot_Graph([0],[0])

    #print(connectivity_plot())
    #eigvalue_average_thread(0.25,15,15)

    eigvalue_average_thread(0.75,15,15)
