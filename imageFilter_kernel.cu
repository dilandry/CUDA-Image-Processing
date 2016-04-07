#ifndef _IMAGEFILTER_KERNEL_H_
#define _IMAGEFILTER_KERNEL_H_

#define RADIUS 4
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
__global__ void imageFilterKernelPartC(uchar3* inputPixels, uchar3* outputPixels, int width, int height /*, other arguments */)
{
	__shared__ uchar3 s_data[128][128];
	int num = 16;

	for(int y = 0; gridDim.y*y*120<=height; y++){
		for(int x = 0; gridDim.x*x*120<=width; x++){
			int offset_x = (blockIdx.x!= 0 || x!=0)? -8 : -4;
			int offset_y = (blockIdx.y!= 0 || y!=0)? -8 : -4;

			int index_x = blockDim.x*(blockIdx.x + gridDim.x*x) + threadIdx.x + offset_x*(blockIdx.x + 1 + gridDim.x*x);
			int index_y = blockDim.y*num*(blockIdx.y + gridDim.y*y) + threadIdx.y + offset_y*(blockIdx.y + 1 + gridDim.y*y);
			
			if(index_x <= width || index_y <= height){
				for(int i = 0 ; i< num ; i ++){
					if(width*(index_y + 8*i) + index_x <= width*height)
						s_data[threadIdx.x][threadIdx.y+8*i] = inputPixels[width*(index_y + 8*i) + index_x];
					__syncthreads();
				}

				for(int k = 0; k<num; k++){
					if (threadIdx.y + 8*k < 124 && threadIdx.y + 8*k >=4 && threadIdx.x < 124 && threadIdx.x >=4){
						int3 sum = make_int3(0, 0, 0);
						int count = 0;
						int location = width*(index_y + 8*k) + index_x;
						for(int i=-RADIUS;i<=RADIUS;i++){
							for(int j=-RADIUS; j<=RADIUS;j++){
								if((location+i*width) && ((location+i*width)/width)<height && (location%width+j) && (location%width+j)<width){
									sum.x += s_data[threadIdx.x+j][threadIdx.y+8*k+i].x;
				        			sum.y += s_data[threadIdx.x+j][threadIdx.y+8*k+i].y;
				        			sum.z += s_data[threadIdx.x+j][threadIdx.y+8*k+i].z;
				        			count ++;
								}
							}
						}

						if(location <=width*height){
							outputPixels[location].x=sum.x/count;
							outputPixels[location].y=sum.y/count;
							outputPixels[location].z=sum.z/count;
						}
					}
				}
			}
		}
	}
}

#endif // _IMAGEFILTER_KERNEL_H_
