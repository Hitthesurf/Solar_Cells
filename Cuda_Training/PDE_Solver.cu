#include <iostream>
#include <math.h>

//Solves the wave equation

const int xDim = 101; //Width
const int yDim = 101; //Height

const double x_s = 0.0;
const double x_f = 1.0;
const double y_s = 0.0;
const double y_f = 1.0;

const double dt = 0.001;

const double D = 1.0;
const int save_frame = 20;


__device__
double u_(int x_off, int y_off, int index, int xpos, int ypos, double *Grid) //x_set is x_offset
{
    //Cannot be on edge boundary
    int xfind = xpos+x_off;
    int yfind = ypos+y_off;
    return Grid[yfind*xDim + xfind];
        
}

__device__
double next_val(double *Grid1, double *Grid2, double *Grid3, int index)
{
    int xpos = index % xDim;
    int ypos = index / xDim;    
    
    double dx = 1.0/double(xDim-1);
    double dy = 1.0/double(yDim-1);
    

    //Check if on boundary and set to BC condition 0
    
    if (((xpos == 0) || (xpos == xDim-1)) || ((ypos == 0) || (ypos == yDim-1)))
    {
        return 0.0 ;   
    }
    
    //Calculate return val
    double u_E = u_(1,0, index, xpos, ypos, Grid2);
    double u_W = u_(-1,0, index, xpos, ypos, Grid2);
    double u_O = u_(0,0, index, xpos, ypos, Grid2);
    double u_N = u_(0,1, index, xpos, ypos, Grid2);
    double u_S = u_(0,-1, index, xpos, ypos, Grid2);
    
    double u_D = Grid3[index];
    

    double u_xx = D*(u_E-2*u_O+u_W)/(dx*dx); 
    double u_yy = D*(u_N-2*u_O+u_S)/(dy*dy);
    double u_tt = u_xx + u_yy;
    
    return dt*dt*u_tt + 2*u_O - u_D;
    
}


//Function to run on GPU
__global__
void calc_next(double *Grid1, double *Grid2, double *Grid3)
{
    
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x*gridDim.x; //Get number of threads in block
    

    
    //Run over all elements
    //Use Grid2 and Grid3 to render to Grid1

    for (int i = index; i < xDim*yDim; i+=stride)
    {
        Grid1[i] = next_val(Grid1, Grid2, Grid3, i);
    }
}

double ic_pos(double x,double y)
{
    return x*y;
}

double ic_vel(double x,double y)
{
    return 0.0;
}

int main()
{
  double *Grid1, *Grid2, *Grid3;

  // Allocate Unified Memory accessible from CPU or GPU
  cudaMallocManaged(&Grid1, xDim*yDim*sizeof(double));
  cudaMallocManaged(&Grid2, xDim*yDim*sizeof(double));
  cudaMallocManaged(&Grid3, xDim*yDim*sizeof(double));

  //Initialise initial conditions on host
  double x;
  double y;
  
  //Set Grid2 and Grid3
  for (int y_i = 0; y_i < yDim; y_i++){
    for (int x_i = 0; x_i < xDim; x_i++)
    {
        int index = y_i*xDim + x_i;
        x = (double(x_i)/(double(xDim-1)))*(x_f-x_s) + x_s;
        y = (double(y_i)/(double(yDim-1)))*(y_f-y_s) + y_s;
        Grid3[index] = ic_pos(x,y);
        Grid2[index] = ic_pos(x,y) + dt*ic_vel(x,y);        
    }  
  }
  
  
  for (int j = 0; j < 8000; j++)
  {
  calc_next<<<30,100>>>(Grid1, Grid2, Grid3);
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  //Save frame if multiple of save number
  if (j%save_frame==0){
      for (int i = 0; i < xDim*yDim; i++ )
        {
            std::cout<<Grid1[i]<<'\n';
        }
  }
  
  //Swap Grids don't need info in Grid3 anymore
  double * swap = Grid3; 
  
  Grid3 = Grid2;
  Grid2 = Grid1;
  Grid1 = swap;
  

  
  }

  return 0;
}