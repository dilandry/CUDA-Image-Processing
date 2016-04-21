
#include <stdio.h> 
#include "cudaConvUtilities.h.cu"
#include "cudaConvolution.h.cu"


using namespace std;

__global__ void   convolveBasic( 
               int*   devInPtr,    
               int*   devOutPtr,    
               int    imgRows,    
               int    imgCols,    
               int*   devPsfPtr,    
               int    psfRowRad,
               int    psfColRad)
{  

   CREATE_CONVOLUTION_VARIABLES(psfRowRad, psfColRad); 
   shmOutput[localIdx] = 0.0f;

   const int psfStride = psfColRad*2+1;   
   const int psfPixels = psfStride*(psfRowRad*2+1);   
   int* shmPsf = (int*)&shmOutput[ROUNDUP32(localPixels)];   

   COPY_LIN_ARRAY_TO_SHMEM(devPsfPtr, shmPsf, psfPixels); 

   PREPARE_PADDED_RECTANGLE(psfRowRad, psfColRad); 


   __syncthreads();   


   int accum = 0.0f; 
   for(int roff=-psfRowRad; roff<=psfRowRad; roff++)   
   {   
      for(int coff=-psfColRad; coff<=psfColRad; coff++)   
      {   
         int psfRow = psfRowRad - roff;   
         int psfCol = psfColRad - coff;   
         int psfIdx = IDX_1D(psfRow, psfCol, psfStride);   
         int psfVal = shmPsf[psfIdx];   

         int shmPRRow = padRectRow + roff;   
         int shmPRCol = padRectCol + coff;   
         int shmPRIdx = IDX_1D(shmPRRow, shmPRCol, padRectStride);   
         accum += psfVal * shmPadRect[shmPRIdx];   
      }   
   }   
   shmOutput[localIdx] = accum;  
   __syncthreads();   

   devOutPtr[globalIdx] = shmOutput[localIdx];  
}

__global__ void   convolveBasicNb2( 
               uchar3*   devInPtr,    
               uchar3*   devOutPtr,    
               int       imgRows,    
               int       imgCols,    
               // int*   devPsfPtr,    
               int    psfRowRad,
               int    psfColRad)
{  

   CREATE_CONVOLUTION_VARIABLES2(psfRowRad, psfColRad); 
   shmOutput[localIdx].x = 0.0f;
   shmOutput[localIdx].y = 0.0f;
   shmOutput[localIdx].z = 0.0f;


   const int psfStride = psfColRad*2+1;   
   const int psfPixels = psfStride*(psfRowRad*2+1);   
   //int* shmPsf = (int*)&shmOutput[ROUNDUP32(localPixels)];   

   //COPY_LIN_ARRAY_TO_SHMEM(devPsfPtr, shmPsf, psfPixels); 
   int shmPsf[81] = {1, 1, 1, 1, 1, 1, 1, 1, 1, \
                       1, 1, 1, 1, 1, 1, 1, 1, 1, \
                       1, 1, 1, 1, 1, 1, 1, 1, 1, \
                       1, 1, 1, 1, 1, 1, 1, 1, 1, \
                       1, 1, 1, 1, 1, 1, 1, 1, 1, \
                       1, 1, 1, 1, 1, 1, 1, 1, 1, \
                       1, 1, 1, 1, 1, 1, 1, 1, 1, \
                       1, 1, 1, 1, 1, 1, 1, 1, 1, \
                       1, 1, 1, 1, 1, 1, 1, 1, 1};  

   PREPARE_PADDED_RECTANGLE2(psfRowRad, psfColRad);

   __syncthreads();   

   //int accum = 0.0f; 
   float3 accum = make_float3(0, 0, 0);
   for(int roff=-psfRowRad; roff<=psfRowRad; roff++)   
   {   
      for(int coff=-psfColRad; coff<=psfColRad; coff++)   
      {   
         int psfRow = psfRowRad - roff;   
         int psfCol = psfColRad - coff;   
         int psfIdx = IDX_1D(psfRow, psfCol, psfStride);   
         int psfVal = shmPsf[psfIdx];   

         int shmPRRow = padRectRow + roff;   
         int shmPRCol = padRectCol + coff;   
         int shmPRIdx = IDX_1D(shmPRRow, shmPRCol, padRectStride);   
         //accum += psfVal * shmPadRect[shmPRIdx];
         // if (roff==0 && coff==0){
         //    accum.x += (float)psfVal * static_cast<float>(shmPadRect[shmPRIdx].x);   
         //    accum.y += (float)psfVal * static_cast<float>(shmPadRect[shmPRIdx].y); 
         //    accum.z += (float)psfVal * static_cast<float>(shmPadRect[shmPRIdx].z); 
         // }
         accum.x += psfVal * static_cast<float>(shmPadRect[shmPRIdx].x);   
         accum.y += psfVal * static_cast<float>(shmPadRect[shmPRIdx].y); 
         accum.z += psfVal * static_cast<float>(shmPadRect[shmPRIdx].z); 
      }   
   }   
   //shmOutput[localIdx] = accum;  
   shmOutput[localIdx].x = accum.x/81; 
   shmOutput[localIdx].y = accum.y/81; 
   shmOutput[localIdx].z = accum.z/81; 
   __syncthreads();   

   //devOutPtr[globalIdx] = shmOutput[localIdx];  
   devOutPtr[globalIdx] = shmOutput[localIdx];
}

__global__ void   convolveBilateral( 
               int*   devInPtr,    // Input image
               int*   devOutPtr,   // Output image
               int    imgRows,     // # of rows in image
               int    imgCols,     // # of columns in image
               int*   devPsfPtr,   // Kernel?
               int    psfRowRad,   // Kernel?
               int    psfColRad,   // Kernel?
               int*   devIntPtr,   // ??Intensity kernel?? Only present in bilateral convolve
               int    intensRad)
{  

   CREATE_CONVOLUTION_VARIABLES(psfRowRad, psfColRad); 
   shmOutput[localIdx] = 0.0f;

   const int padRectIdx = IDX_1D(padRectRow, padRectCol, padRectStride);
   const int psfStride = psfColRad*2+1;   
   const int psfPixels = psfStride*(psfRowRad*2+1);   
   int* shmPsf  = (int*)&shmOutput[ROUNDUP32(localPixels)];   
   int* shmPsfI = (int*)&shmPsf[ROUNDUP32(psfPixels)];   
   
   //                      <  src  >  <dst >   <num_values>
   COPY_LIN_ARRAY_TO_SHMEM(devPsfPtr, shmPsf,  psfPixels); 
   COPY_LIN_ARRAY_TO_SHMEM(devIntPtr, shmPsfI, 2*intensRad+1);

   PREPARE_PADDED_RECTANGLE(psfRowRad, psfColRad); // Uses devInPtr and copies it into shmPadRect

   __syncthreads();   

   int accum = 0.0f; 
   int myVal = shmPadRect[padRectIdx];
   for(int roff=-psfRowRad; roff<=psfRowRad; roff++)   
   {   
      for(int coff=-psfColRad; coff<=psfColRad; coff++)   
      {   
         int psfRow = psfRowRad - roff;   
         int psfCol = psfColRad - coff;   
         int psfIdx = IDX_1D(psfRow, psfCol, psfStride);   
         int psfVal = shmPsf[psfIdx];   // Kernel val

         int shmPRRow = padRectRow + roff;   
         int shmPRCol = padRectCol + coff;   
         int shmPRIdx = IDX_1D(shmPRRow, shmPRCol, padRectStride);   
         int thatVal = shmPadRect[shmPRIdx]; // Pixel val

         int intVal = shmPsfI[(int)(thatVal-myVal+intensRad)];

         accum += psfVal * intVal *shmPadRect[shmPRIdx];   
      }   
   }   
   shmOutput[localIdx] = accum;  
   __syncthreads();   

   devOutPtr[globalIdx] = shmOutput[localIdx];  
}

