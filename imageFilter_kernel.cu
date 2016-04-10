#ifndef _IMAGEFILTER_KERNEL_H_
#define _IMAGEFILTER_KERNEL_H_

#define RADIUS 4
#define BLOCKS 12
#define THREADS 1024
#define tile_w 128
#define tile_h 128

#define CREATE_3X3_MORPH_KERNEL( name, seTargSum, \
                                            a00, a10, a20, \
                                            a01, a11, a21, \
                                            a02, a12, a22) ({ \
    int a[3][3] = {{a00, a10, a20}, {a01, a11, a21}, {a02, a12, a22}}; \
	int target_count = seTargSum; \
})
#define CREATE_9X9_MORPH_KERNEL( name, seTargSum, \
                                            a00, a10, a20, a30, a40, a50, a60, a70, a80, \
                                            a01, a11, a21, a31, a41, a51, a61, a71, a81, \
                                            a02, a12, a22, a32, a42, a52, a62, a72, a82, \
                                            a03, a13, a23, a33, a43, a53, a63, a73, a83, \
                                            a04, a14, a24, a34, a44, a54, a64, a74, a84, \
                                            a05, a15, a25, a35, a45, a55, a65, a75, a85, \
                                            a06, a16, a26, a36, a46, a56, a66, a76, a86, \
                                            a07, a17, a27, a37, a47, a57, a67, a77, a87, \
                                            a08, a18, a28, a38, a48, a58, a68, a78, a88) ({ \
    int a[9][9] = {{a00, a10, a20, a30, a40, a50, a60, a70, a80}, \
    			   {a01, a11, a21, a31, a41, a51, a61, a71, a81}, \
    			   {a02, a12, a22, a32, a42, a52, a62, a72, a82}, \
    			   {a03, a13, a23, a33, a43, a53, a63, a73, a83}, \
    			   {a04, a14, a24, a34, a44, a54, a64, a74, a84}, \
    			   {a05, a15, a25, a35, a45, a55, a65, a75, a85}, \
    			   {a06, a16, a26, a36, a46, a56, a66, a76, a86}, \
    			   {a07, a17, a27, a37, a47, a57, a67, a77, a87}, \
    			   {a08, a18, a28, a38, a48, a58, a68, a78, a88}}; \
	int target_count = seTargSum; \
})
 	//  \
 	// a[0][0]= a00; \
 	// a[1][0]= a10; \
 	// a[2][0]= a20; \
	// a[0][1]= a01; \
	// a[1][1]= a11; \
	// a[2][1]= a21; \
	// a[0][2]= a02; \
	// a[1][2]= a12; \
	// a[2][2]= a22;                              

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

	// CREATE_3X3_MORPH_KERNEL( BoxBlur, 9, \
 //                                            1, 1, 1, \
 //                                            1, 1, 1, \
 //                                            1, 1, 1);
	// CREATE_9X9_MORPH_KERNEL( BoxBlur, 81, \
	// 	                                  1, 1, 1, 1, 1, 1, 1, 1, 1,
	// 	                                  1, 1, 1, 1, 1, 1, 1, 1, 1,
	// 	                                  1, 1, 1, 1, 1, 1, 1, 1, 1,
	// 	                                  1, 1, 1, 1, 1, 1, 1, 1, 1,
	// 	                                  1, 1, 1, 1, 1, 1, 1, 1, 1,
	// 	                                  1, 1, 1, 1, 1, 1, 1, 1, 1,
	// 	                                  1, 1, 1, 1, 1, 1, 1, 1, 1,
	// 	                                  1, 1, 1, 1, 1, 1, 1, 1, 1,
	// 	                                  1, 1, 1, 1, 1, 1, 1, 1, 1);

	// Image kernel to apply
	int a[9][9] = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, \
    		   	   {1, 1, 1, 1, 1, 1, 1, 1, 1}, \
    		   	   {1, 1, 1, 1, 1, 1, 1, 1, 1}, \
    		   	   {1, 1, 1, 1, 1, 1, 1, 1, 1}, \
    		   	   {1, 1, 1, 1, 1, 1, 1, 1, 1}, \
    		   	   {1, 1, 1, 1, 1, 1, 1, 1, 1}, \
    		   	   {1, 1, 1, 1, 1, 1, 1, 1, 1}, \
    		   	   {1, 1, 1, 1, 1, 1, 1, 1, 1}, \
    		   	   {1, 1, 1, 1, 1, 1, 1, 1, 1}};
    int targetSum = 81;


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
						//DL--int count = 0;
						int localSum = targetSum;
						int location = width*(index_y + 8*k) + index_x;
						for(int i=-RADIUS;i<=RADIUS;i++){
							for(int j=-RADIUS; j<=RADIUS;j++){
								if(a[i+RADIUS][j+RADIUS] == 0) continue;
								if((location+i*width) && ((location+i*width)/width)<height && (location%width+j) && (location%width+j)<width){
									sum.x += s_data[threadIdx.x+j][threadIdx.y+8*k+i].x;
				        			sum.y += s_data[threadIdx.x+j][threadIdx.y+8*k+i].y;
				        			sum.z += s_data[threadIdx.x+j][threadIdx.y+8*k+i].z;
				        			//DL--count ++;
								} else {
									localSum --;
								}
							}
						}

						if(location <=width*height){
							outputPixels[location].x=sum.x/localSum;
							outputPixels[location].y=sum.y/localSum;
							outputPixels[location].z=sum.z/localSum;
						}
					}
				}
			}
		}
	}
}

#endif // _IMAGEFILTER_KERNEL_H_
