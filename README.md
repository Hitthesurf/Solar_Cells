# Solar_Cells
Modelling the Movement of Electrons in Solar Cells by numerically solving PDEs by FDM, and by the GDM (Gaussian Disorder Model)

# Running CUDA
To run the cuda PDE_Solver on a server or local coputer. A NVIDIA graphics card is needed.
The module is loaded by,

module load cuda_location


The file is compiled by,

nvcc file_to_compile.cu -o name_of_output


An easy way to save data to a file would be print the data you need then save the output of the program, do

./name_of_output > Results.txt


# Website to solve wave equation (Need WebGL2)
This solves u_tt = u_xx + u_yy,

BC's are all zero on a square domain[0,1]x[0,1]

IC is xy.


https://hitthesurf.github.io/Solar_Cells/PDE%20Solver/


# Learning of Data Visualisation can be found at

https://github.com/Hitthesurf/Rendering/tree/main