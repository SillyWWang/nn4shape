# ----------------------------------------------------------------------------------------------------
# Generate Data
# ----------------------------------------------------------------------------------------------------

import torch
import numpy as np
import pandas as pd


def filter_2d(x, edges):
    """ Filter data points located within a certain domain
    parameters
    x: coordinates of data points
    edges: edges of computational domain

    returns
    ind: if data points located within the domain
    """
    dim = 2
    ind = torch.zeros(x.shape[0])
    tmp_x = torch.zeros(x.shape[0],2)
    for i in range(edges.shape[0]):
        """ vertices of an edge """
        x0 = edges[i,:dim]
        x1 = edges[i,dim:]

        """ Generate a ray from each data point along the negative direction of x-axis, 
            then judge if the ray has an intersection with the edge """
        tmp_x[:,1] = x[:,1]
        if abs(x0[1]-x1[1]) > 1e-8:
            """ Calculate the intersection of the line on which the ray located on 
                and the line on which the edge located on """
            tmp_x[:,0] = x0[0] + (x1[0]-x0[0])/(x1[1]-x0[1]) * (tmp_x[:,1]-x0[1])
            
            """ If the ray has an intersection with the edge, 
                then the intersection must on the left of the data point,
                and the direction of two vectors from the intersection to
                the vertices of edge are opposite """
            tmp_ind = (tmp_x[:,0]<x[:,0]) & (((tmp_x-x0)*(tmp_x-x1)).sum(1) < 0)
            
            """ Update the number of intersections of the ray and domain boundary """
            ind[tmp_ind] += 1
    
    """ If the ray has an odd number of intersections with boundary, 
        then the data points located within the domain """
    ind = torch.fmod(ind,2)==1
    return ind


class Geometry2D():
    """ data structure of computational domain
    parameters
    vertices_file: data file that stores the boundary points of the computational domain
    edges_ind_file: data file that stores the indices of edges of domain boundary
    data_type: data type

    attributes
    data_type: data type
    vertices: vertices of edges
    size_vertices: number of vertices
    dim: spatial dimension
    edges_ind: indices of vertices of edges
    edges: coordinates of vertices of edges
    size_vertices: number of edges
    bounds: spatial bounds of computational domain
    """
    def __init__(self, vertices_file, edges_ind_file, data_type, csv_file=True):
        self.data_type = data_type
        
        """ Get coordinates of boundary points """
        if csv_file:
            vertices = pd.read_csv(vertices_file)
            vertices = np.array(vertices)
        else:
            vertices = vertices_file
        self.vertices = torch.tensor(vertices, dtype=self.data_type)
        self.vertices = self.vertices
        self.size_vertices = self.vertices.shape[0]
        self.dim = self.vertices.shape[1]

        """ Generate edges of the computational domain """
        if csv_file:
            edges_ind = pd.read_csv(edges_ind_file)
            edges_ind = np.array(edges_ind)
        else:
            edges_ind = edges_ind_file
        self.edges_ind = torch.tensor(edges_ind, dtype=self.data_type).long() - 1
        self.edges = torch.cat((self.vertices[self.edges_ind[:,0],:],
                                self.vertices[self.edges_ind[:,1],:]), 1)
        self.size_edges = self.edges.shape[0]
        
        """ Evaluate the spatial bounds of computational domain """
        self.bounds = torch.zeros(self.dim,2)
        self.bounds[:,0] = (self.vertices[:,:self.dim]).min(0).values
        self.bounds[:,1] = (self.vertices[:,:self.dim]).max(0).values