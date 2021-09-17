import numpy as np
import ipyvolume as ipv

class GridS:
    '''Single Buffer Grid
    Gets and sets Grid elements bases on (0,0) being bottom left
    '''

    def __init__(self, Grid):
        # Grid is numpy 2D array,
        self.Grid = Grid

    def u_(self, i, j):
        return self.Grid[len(self.Grid)-j-1][i]

    def Set_u_(self, i, j, val):
        self.Grid[len(self.Grid)-j-1][i] = val

    def Plot(self, higher, h):
        a = np.arange(0.0, higher+0.00001, h)
        U, V = np.meshgrid(a, a)
        X = U
        Y = V
        Z = self.Grid

        ipv.figure()
        ipv.plot_surface(X, Y, Z, color="orange")
        ipv.plot_wireframe(X, Y, Z, color="red")
        ipv.show()

class GridD:
    '''Double Buffer Grid'''

    def __init__(self, Grid):
        self.Grid = Grid.copy()  # Write_To
        self.Grid2 = Grid.copy()  # Read_From

    def SwapGrids(self):  # More advance version would swap the reference location of the Ram #Remove copy?
        temp = self.Grid.copy()
        self.Grid = self.Grid2.copy()
        self.Grid2 = temp.copy()

    def u_(self, i, j):
        # Read
        return self.Grid2[len(self.Grid2)-j-1][i]

    def Set_u_(self, i, j, val):
        # Write
        self.Grid[len(Grid)-j-1][i] = val
        
        