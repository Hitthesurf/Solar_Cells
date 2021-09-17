#include <iostream>
#include <math.h>

//Solves the Electron Drift Diffusion Equation

const int xDim = 101; //Width
const int yDim = 101; //Height

const double x_s = 0.0;
const double x_f = 1.0;
const double y_s = 0.0;
const double y_f = 1.0;

const double small = 0.0001;
const double dt = 0.001;
const int save_frame = 40;
const double pi = 3.14159265358979323846;

const double dphydx_x= 1.0;
const double A = 1.0;
const double E_c = 2.0; //Average energy
const double o = 1.5; //Standard deviation of energy

const double v_0 = 1.0;
const double lam = 1.0;//Lambda
const double beta = 1.0;//1 over k_b T
const double Gamma = 1.0;


__device__
double g_1(double x) //Density of states in space
{
    return x;
}

__device__
double dg_1(double x) //partial g_1(x) over partial x
{
    return (g_1(x+small)-g_1(x))/small;
}

__device__
double u_(int x_off, int y_off, int index, int xpos, int ypos, double *Grid) //x_set is x_offset
{
    //Cannot be on edge boundary
    int xfind = xpos+x_off;
    int yfind = ypos+y_off;
    return Grid[yfind*xDim + xfind];
        
}

__device__
double K()
{
    return 1/(4*Gamma*Gamma*Gamma);
}

__device__
double C()
{
    return 1/Gamma;
}

__device__
double o_tilda_sq()
{
    return o*o + (2*lam)/(beta);
}

__device__
double E_n(double e)
{
    return -((lam)/(o_tilda_sq()))*(2*(e-E_c)/beta + o*o);

}

__device__
double E_p(double e)
{
    return -((lam)/(o_tilda_sq()))*(2*(e-E_c)/beta - o*o);
}

__device__
double dphydx__(double *Grid2p, double *Grid2n, int epos, int spos)
{
    double dx = 1.0/double(xDim-1);
    double temp = dphydx_x;
    for (int xpos = 0; xpos < spos; xpos++)
    {
        temp += A*(Grid2p[epos*xDim+xpos] - Grid2n[epos*xDim+xpos])*dx;
    }
    return temp;
}

__device__
double next_val_n(double *Grid2p, double *Grid2n, int index)
{
    int xpos = index % xDim;
    int ypos = index / xDim;    
    double dphydx = dphydx__(Grid2p, Grid2n, ypos, xpos);
    
    double x = (double(xpos)/(double(xDim-1)))*(x_f-x_s) + x_s;
    double e = (double(ypos)/(double(yDim-1)))*(y_f-y_s) + y_s;//Energy
    
    double dx = 1.0/double(xDim-1);
    double de = 1.0/double(yDim-1);
    

    //Check if on boundary and set to BC condition 0
    
    if (((xpos == 0) || (xpos == xDim-1)) || ((ypos == 0) || (ypos == yDim-1)))
    {
        return 1.0-x;   
    }
    
    //Calculate return val
    double n_E = u_(1,0, index, xpos, ypos, Grid2n);
    double n_W = u_(-1,0, index, xpos, ypos, Grid2n);
    double n_O = u_(0,0, index, xpos, ypos, Grid2n);
    double n_N = u_(0,1, index, xpos, ypos, Grid2n);
    double n_S = u_(0,-1, index, xpos, ypos, Grid2n);
    

    double n_e = (n_N-n_S)/(2*de); 
    double n_x = (n_E-n_W)/(2*dx);
    double n_xx = (n_E-2*n_O+n_W)/(dx*dx);
    
    double n_t = K()*(-beta*g_1(x)*dphydx/2 + dg_1(x))*n_x + K()*g_1(x)*n_xx/2 + C()*g_1(x)*E_n(e)*n_e;
    n_t *= (v_0/sqrt(2*pi*o_tilda_sq()))*exp(-(e-E_c-lam)/(2*o_tilda_sq()));
    return n_O + dt*n_t;
    
}


__device__
double next_val_p(double *Grid2p, double *Grid2n, int index)
{
    int xpos = index % xDim;
    int ypos = index / xDim;    
    double dphydx = dphydx__(Grid2p, Grid2n, ypos, xpos);
    
    double x = (double(xpos)/(double(xDim-1)))*(x_f-x_s) + x_s;
    double e = (double(ypos)/(double(yDim-1)))*(y_f-y_s) + y_s;//Energy
    
    double dx = 1.0/double(xDim-1);
    double de = 1.0/double(yDim-1);
    

    //Check if on boundary and set to BC condition 0
    
    if (((xpos == 0) || (xpos == xDim-1)) || ((ypos == 0) || (ypos == yDim-1)))
    {
        return x;   
    }
    
    //Calculate return val
    double p_E = u_(1,0, index, xpos, ypos, Grid2p);
    double p_W = u_(-1,0, index, xpos, ypos, Grid2p);
    double p_O = u_(0,0, index, xpos, ypos, Grid2p);
    double p_N = u_(0,1, index, xpos, ypos, Grid2p);
    double p_S = u_(0,-1, index, xpos, ypos, Grid2p);
    

    double p_e = (p_N-p_S)/(2*de); 
    double p_x = (p_E-p_W)/(2*dx);
    double p_xx = (p_E-2*p_O+p_W)/(dx*dx);
    
    double p_t = K()*(beta*g_1(x)*dphydx/2 + dg_1(x))*p_x + K()*g_1(x)*p_xx/2 + C()*g_1(x)*E_p(e)*p_e;
    p_t *= (v_0/sqrt(2*pi*o_tilda_sq()))*exp(-(e-E_c+lam)/(2*o_tilda_sq()));
    return p_O + dt*p_t;
    
}


//Function to run on GPU
__global__
void calc_next(double *Grid1n, double *Grid2n, double *Grid1p, double *Grid2p)
{
    
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x*gridDim.x; //Get number of threads in block
    

    
    //Run over all elements
    //Use Grid2n and Grid2p to render to Grid1n and Grid1p

    for (int i = index; i < xDim*yDim; i+=stride)
    {
        Grid1n[i] = next_val_n(Grid2p, Grid2n, i);
        Grid1p[i] = next_val_p(Grid2p, Grid2n, i);
    }
}

double ic_pos_n(double x,double e)
{
    return 1.0 - x;
}

double ic_pos_p(double x,double e)
{
    return x;
}

int main()
{
  double *Grid1p, *Grid2p, *Grid1n, *Grid2n;

  // Allocate Unified Memory accessible from CPU or GPU
  cudaMallocManaged(&Grid1p, xDim*yDim*sizeof(double));
  cudaMallocManaged(&Grid2p, xDim*yDim*sizeof(double));
  cudaMallocManaged(&Grid1n, xDim*yDim*sizeof(double));
  cudaMallocManaged(&Grid2n, xDim*yDim*sizeof(double));

  //Initialise initial conditions on host
  double x;
  double y; //Represents energy
  
  //Set Grid2 and Grid3
  for (int y_i = 0; y_i < yDim; y_i++){
    for (int x_i = 0; x_i < xDim; x_i++)
    {
        int index = y_i*xDim + x_i;
        x = (double(x_i)/(double(xDim-1)))*(x_f-x_s) + x_s;
        y = (double(y_i)/(double(yDim-1)))*(y_f-y_s) + y_s;
        Grid2n[index] = ic_pos_n(x,y);
        Grid2p[index] = ic_pos_p(x,y);        
    }  
  }
  
  
  for (int j = 0; j < 8000*6; j++)
  {
  calc_next<<<30,100>>>(Grid1n, Grid2n, Grid1p, Grid2p);
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  //Save frame if multiple of save number
  if (j%save_frame==0){
      for (int i = 0; i < xDim*yDim; i++ )
        {
            std::cout<<Grid1n[i]<<'\n';
        }
  }
  
  //Swap Grids don't need info in Grid3 anymore
    
  double * swapn = Grid2n; 
  double * swapp = Grid2p;
  Grid2n = Grid1n;
  Grid2p = Grid1p;
  
  Grid1n = swapn;
  Grid1p = swapp;

  
  }

  return 0;
}