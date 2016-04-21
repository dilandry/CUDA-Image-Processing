#ifndef _IMAGEFILTER_KERNEL_H_
#define _IMAGEFILTER_KERNEL_H_

#define RADIUS 1
#define BLOCKS 12
#define THREADS 1024
#define tile_w 128
#define tile_h 128   


__global__ void imageFilterKernelPartA(uchar3* inputPixels, uchar3* outputPixels, int width, int height /*, other arguments */)
{
	
	int chunks;
	if ((float)(width*height)/(float)(BLOCKS*THREADS)-width*height/(BLOCKS*THREADS)>0)
		chunks = width*height/(BLOCKS*THREADS)+1;
	else
		chunks = width*height/(BLOCKS*THREADS);

	int location=(blockDim.x * blockIdx.x + threadIdx.x)*chunks;


	for(int k = 0; k<chunks; k ++){
		if(location<width*height){
			int3 sum = make_int3(0, 0, 0);
			int count = 0;
			for(int i = -RADIUS; i <= RADIUS; i ++)
			{
			    for(int j = -RADIUS; j <= RADIUS; j ++)
			    {
			        //proper checks to see if this pixel exists
			        if((location+i*width) && ((location+i*width)/width)<height && (location%width+j) && (location%width+j)<width) {
				        sum.x += inputPixels[location + i * width + j].x;
				        sum.y += inputPixels[location + i * width + j].y;
				        sum.z += inputPixels[location + i * width + j].z;
				        count ++;
				    }
			    }
			}
			outputPixels[location].x = sum.x / count;
			outputPixels[location].y = sum.y / count;
			outputPixels[location].z = sum.z / count;
		}
		location++;
	}

}
__global__ void imageFilterKernelPartB(uchar3* inputPixels, uchar3* outputPixels, int width, int height /*, other arguments */)
{

	int location=blockDim.x * blockIdx.x + threadIdx.x;

	while(location<width*height){
		int3 sum = make_int3(0, 0, 0);
		int count = 0;
		for(int i = -RADIUS; i <= RADIUS; i ++)
		{
			for(int j = -RADIUS; j <= RADIUS; j ++)
			{
				__syncthreads();
			    //proper checks to see if this pixel exists
			    if((location+i*width) && ((location+i*width)/width)<height && (location%width+j) && (location%width+j)<width) {
			        sum.x += inputPixels[location + i * width + j].x;
			        sum.y += inputPixels[location + i * width + j].y;
			        sum.z += inputPixels[location + i * width + j].z;
			        count ++;
			    }
			}
		}
		outputPixels[location].x = sum.x / count;
		outputPixels[location].y = sum.y / count;
		outputPixels[location].z = sum.z / count;
		location+=(BLOCKS*THREADS);
	}

}
__global__ void imageFilterKernel(uchar3* inputPixels, uchar3* outputPixels, int width, int height, char filterId/*, other arguments */)
{
	__shared__ uchar3 s_data[128][128];
	int num = 16;
	// Image kernel to apply
	int targetSum;
	const int dem = 2*RADIUS +1;
	int a[dem][dem];
	
	/*
	int **a;
  	a= (int **)malloc(sizeof(int *)*dem);
  	for(int i=0; i < dem; i++) {
  		a[i] = (int *)malloc(sizeof(int)*dem);
	}*/
	
	int mid = (dem + 1)/2; 
	switch(filterId){
		case '1' : // ----> Box Blur filter (same as assignment)
			/*
			int **boxBlur;
	  		boxBlur= (int **)malloc(sizeof(int *)*dem);
	  		for(int i=0; i < dem; i++) {
	  				boxBlur[i] = (int *)malloc(sizeof(int)*dem);
			}*/
			for (int i =0 ; i < dem; i++){
				for (int j = 0; j < dem ; j++){
					a[i][j] = 1;
				}
			}
    		//memcpy(a, boxBlur, sizeof (int) * dem * dem);
    		targetSum = dem * dem;
			break;
		case '2' : // ----> Sharpen filter
			/*
			int **sharpen;
	  		sharpen= (int **)malloc(sizeof(int *)*dem);
	  		for(int i=0; i < dem; i++) {
	  				sharpen[i] = (int *)malloc(sizeof(int)*dem);
			}*/
			for (int i =0 ; i < dem; i++){
				for (int j = 0; j < dem ; j++){
					if( i == mid && j == mid){
						a[i][j] = dem*2 -1;
					}else if( i == mid || j == mid){
						a[i][j] = -1;
					}else {
						//sharpen[i][j] = 0;
						a[i][j] = 0;
					}
				}
			}
    		//memcpy(a, sharpen, sizeof (int) * dem * dem);
    		targetSum = 1;
			break;
		case '3' : // ----> Sobel filter
			/*
			int **sobel;
	  		sobel= (int **)malloc(sizeof(int *)*dem);
	  		for(int i=0; i < dem; i++) {
	  				sobel[i] = (int *)malloc(sizeof(int)*dem);
			}*/

			int s = (dem - 1)/2; 
			for (int i =0 ; i < mid ; i++){
				for (int j = 0; j < mid - 1; j++){
					a[i][j] = s + i -j;
					a[i][dem-j-1] = -(s + i-j);
					
					a[dem - i - 1][j] = s + i -j;
					a[dem - i - 1][dem-j-1] = -(s + i-j);
					
				}
				a[i][mid-1] = 0;
				a[dem-i-1][mid-1] = 0;
			}
			
    		//memcpy(a, sobel, sizeof (int) * dem * dem);
    		targetSum = 1;
			break;
	}

	for(int y = 0; gridDim.y*y*120<=height; y++){
		for(int x = 0; gridDim.x*x*120<=width; x++){
			int offset_x = (blockIdx.x!= 0 || x!=0)? -8 : -4;
			int offset_y = (blockIdx.y!= 0 || y!=0)? -8 : -4;

			int index_x = blockDim.x*(blockIdx.x + gridDim.x*x) + threadIdx.x + offset_x*(blockIdx.x + 1 + gridDim.x*x);
			int index_y = blockDim.y*num*(blockIdx.y + gridDim.y*y) + threadIdx.y + offset_y*(blockIdx.y + 1 + gridDim.y*y);
			
			if(index_x <= width || index_y <= height){
				for(int i = 0 ; i< num ; i ++){
					if(width*(index_y + 8*i) + index_x <= width*height)
						s_data[threadIdx.y+8*i][threadIdx.x] = inputPixels[width*(index_y + 8*i) + index_x];
					__syncthreads();
				}

				for(int k = 0; k<num; k++){
					if (threadIdx.y + 8*k < 124 && threadIdx.y + 8*k >=4 && threadIdx.x < 124 && threadIdx.x >=4){
						float3 channel_value = make_float3(0, 0, 0);
						int localSum = targetSum;
						int location = width*(index_y + 8*k) + index_x;
						for(int i=-RADIUS;i<=RADIUS;i++){
							for(int j=-RADIUS; j<=RADIUS;j++){
								if(a[i+RADIUS][j+RADIUS] == 0) continue;
								if((location+i*width) && ((location+i*width)/width)<height && (location%width+j) && (location%width+j)<width){
									// Pixel value is multiplied by kernel value at location
					        		channel_value.x += (float)a[i+RADIUS][j+RADIUS] * static_cast<float>(s_data[threadIdx.y+8*k+i][threadIdx.x+j].x);
					        		channel_value.y += (float)a[i+RADIUS][j+RADIUS] * static_cast<float>(s_data[threadIdx.y+8*k+i][threadIdx.x+j].y);
					        		channel_value.z += (float)a[i+RADIUS][j+RADIUS] * static_cast<float>(s_data[threadIdx.y+8*k+i][threadIdx.x+j].z);
								} else {
									// If pixel is out of reach, no scaling
									localSum --;
								}
							}
						}

						if(location <=width*height){
							// Perform scaling
							outputPixels[location].x=channel_value.x/localSum;
							outputPixels[location].y=channel_value.y/localSum;
							outputPixels[location].z=channel_value.z/localSum;
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < dem; i++)
	{
    	int* currentIntPtr = a[i];
    	free(currentIntPtr);
	}
}

#endif // _IMAGEFILTER_KERNEL_H_
