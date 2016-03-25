#ifndef _IMAGEFILTER_KERNEL_H_
#define _IMAGEFILTER_KERNEL_H_
#define getLinearFromXY(x, y, width, height) (x + (y*width))
#define getX(linear, width, height) ((linear) % width)
#define getY(linear, width, height) ((linear) / width)

__global__ void imageFilterKernelPartA(uchar3* inputPixels, uchar3* outputPixels, uint width, uint height /*, other arguments */)
{
	// Need to use 12 blocks of 1024 threads...
	// Develop a kernel in which threads directly access global memory for each pixel. 
	// The workload is distributed in a way that each thread is given a different coarse 
	// grained chunk of the output pixels to produce; therefore, the accesses to memory 
	// will not be coalesced.


	// Each thread will have to compute a specific number of pixels...
	signed int radius = 4;

	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int numThreads = blockDim.x;
	unsigned int numBlocks = gridDim.x;

	// Number of pixels each thread will process...
	unsigned int pixelsPerThread = ((width * height) / (numBlocks * numThreads)) +1;

	//for(signed int location = 0; location < width * height; location++){ // For each pixel
	for(int threadPixelNumber = 0; threadPixelNumber <= pixelsPerThread; threadPixelNumber++){
		// Threads process non-contiguous pixels throughout the whole image
		signed int location = pixelsPerThread * idx + threadPixelNumber;
		signed int x = location % width;
		signed int y = location / width;

		int3 sum;
		sum.x = 0;
		sum.y = 0;
		sum.z = 0;
		int count = 0;

		for (signed int i = -radius; i<=radius; i++){ // for every position on the row
			for (signed int j = -radius; j<=radius; j++){ // for every row
				// // Check if pixel is out of bounds...
				// if (location + i * width + j >= width * height) continue;
				// if (location + i * width + j < 0) continue;

				// sum.x += (int) inputPixels[location+i * width+j].x;
				// sum.y += (int) inputPixels[location+i * width+j].y;
				// sum.z += (int) inputPixels[location+i * width+j].z;

				// Check if pixel is out of bounds (improved version)...
				if (x + i < 0      || y + j < 0      ) continue;
				if (x + i >= width || y + j >= height) continue;

				sum.x += (int) inputPixels[getLinearFromXY((x+i), (y+j), width, height)].x;
				sum.y += (int) inputPixels[getLinearFromXY((x+i), (y+j), width, height)].y;
				sum.z += (int) inputPixels[getLinearFromXY((x+i), (y+j), width, height)].z;

				count++;
			}
		}

		outputPixels[location].x = sum.x / count;
		outputPixels[location].y = sum.y / count;
		outputPixels[location].z = sum.z / count;
	}

	//Assign IDs to threads
	//distribute work between threads
	//do the computation and store the output pixels in outputPixels

}
__global__ void imageFilterKernelPartB(uchar3* inputPixels, uchar3* outputPixels, uint width, uint height /*, other arguments */)
{
	// Develop a kernel in which threads directly access global memory for each pixel. 
	// The workload is distributed in a way that consecutive threads will produce consecutive
	// output pixels; therefore, the accesses will be coalesced.

	signed int radius = 4;

	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int numThreads = blockDim.x;
	unsigned int numBlocks = gridDim.x;

	// Number of pixels each thread will process...
	unsigned int pixelsPerThread = (width * height) / (numBlocks * numThreads);

	// Group pixels; they cannot be processed all at once as there are not enough threads
	for(int threadPixelNumber = 0; threadPixelNumber <= pixelsPerThread; threadPixelNumber++){
		// Threads process consecutive values
		signed int location = idx + threadPixelNumber * (numBlocks * numThreads);
		signed int x = location % width;
		signed int y = location / width;

		int3 sum;
		sum.x = 0;
		sum.y = 0;
		sum.z = 0;
		int count = 0;

		for (signed int i = -radius; i<=radius; i++){ // for every position on the row
			for (signed int j = -radius; j<=radius; j++){ // for every row
				// // Check if pixel is out of bounds...
				// if (location + i * width + j >= width * height) continue;
				// if (location + i * width + j < 0) continue;

				// sum.x += (int) inputPixels[location+i * width+j].x;
				// sum.y += (int) inputPixels[location+i * width+j].y;
				// sum.z += (int) inputPixels[location+i * width+j].z;

				// Check if pixel is out of bounds (improved version)...
				if (x + i < 0      || y + j < 0      ) continue;
				if (x + i >= width || y + j >= height) continue;

				sum.x += (int) inputPixels[getLinearFromXY((x+i), (y+j), width, height)].x;
				sum.y += (int) inputPixels[getLinearFromXY((x+i), (y+j), width, height)].y;
				sum.z += (int) inputPixels[getLinearFromXY((x+i), (y+j), width, height)].z;

				count++;
			}
		}

		outputPixels[location].x = sum.x / count;
		outputPixels[location].y = sum.y / count;
		outputPixels[location].z = sum.z / count;
	}

	//Assign IDs to threads
	//distribute work between threads
	//do the computation and store the output pixels in outputPixels

}
__global__ void imageFilterKernelPartC(uchar3* inputPixels, uchar3* outputPixels, uint width, uint height /*, other arguments */)
{
	// Develop a kernel in which threads first load tiles of data into shared memory and
	// access data from there.

	__shared__ uchar3 s_input[128][128];

	// Each thread will have to compute a specific number of pixels...
	signed int radius = 4;

	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int numThreads = blockDim.x;
	unsigned int numBlocks = gridDim.x;

	// Number of 120*120 tiles the image will be split into 
	unsigned int tilesInImage = (width / 120 +1) * (height / 120 +1);
	int numberOfPasses = tilesInImage / numBlocks;

	//for (int passNumber = 0; passNumber <= numberOfPasses; passNumber++) { // For every tile
	for (int passNumber = 0; passNumber <= numberOfPasses; passNumber++){
		unsigned int linearTileNumber = numBlocks * passNumber + blockIdx.x;
		//if (blockIdx.x == 11 && tid == 0) printf("(%i , numberOfPasses: %i, tilesInImage: %i, numBlocks: %i, numThreads:%i); \n", linearTileNumber, numberOfPasses, tilesInImage, numBlocks, numThreads);

		unsigned int blockXcoordinate = linearTileNumber % (width/120 +1/* removed a +1 here for the edge of image*/);
		unsigned int blockYcoordinate = linearTileNumber / (width/120 +1/* removed a +1 here for the edge of image*/);

		unsigned int blockXoffset = blockXcoordinate * 120 -1;
		unsigned int blockYoffset = blockYcoordinate * 120 -1;

		// Get threads to load into shared memory
		for (int i = 0; i < 16; i++){
			unsigned int innerX = (threadIdx.x + i*1024) % 128; //[0-127]
			unsigned int innerY = (threadIdx.x + i*1024) / 128; //[0-127]

			// if (tid == 1000 && blockIdx.x == 0) {
			// 	printf("(%i, %i, blockXoffset: %i, blockYoffset: %i)\n", innerX, innerY, blockXoffset, blockYoffset);
			// 	printf("-> linear -> (%i)", getLinearFromXY((blockXoffset + innerX), (blockYoffset + innerY), width, height));
			// }

			int x = blockXoffset + innerX - radius;
			int y = blockYoffset + innerY - radius;
			int linearCoord = getLinearFromXY(x, y, width, height);

			if (linearCoord > width*height) continue;
			if ((x < 0 || y < 0) || (x >= width || y >= height)) continue;

			s_input[innerX][innerY].x = inputPixels[linearCoord].x;
			s_input[innerX][innerY].y = inputPixels[linearCoord].y;
			s_input[innerX][innerY].z = inputPixels[linearCoord].z;
	
			// outputPixels[linearCoord].x = s_input[innerX][innerY].x;
			// outputPixels[linearCoord].y = s_input[innerX][innerY].y;
	 		// outputPixels[linearCoord].z = s_input[innerX][innerY].z;
			
		}

		// Do the actual blurring
		for (int i = 0; i < 16; i++){
			// 120 pixels of output
			signed int innerX = (threadIdx.x + i*1024) % 120 + (radius); //[0-127]
			signed int innerY = (threadIdx.x + i*1024) / 120 + (radius); //[0-127]

			// Coordinates of the pixel in the original image
			signed int x = (signed int)blockXoffset + innerX - radius;
			signed int y = (signed int)blockYoffset + innerY - radius;

			int linearCoord = getLinearFromXY(x, y, width, height);

			if (linearCoord > width*height) continue;
			if (x > blockXoffset + 120 || y > blockYoffset + 120) continue;
			if ((x < 0 || y < 0) || (x >= width || y >= height)) continue;

			int3 sum;
			sum.x = 0;
			sum.y = 0;
			sum.z = 0;
			int count = 0;
	
			for (signed int i = -radius; i<=radius; i++){ // for every position on the row
				for (signed int j = -radius; j<=radius; j++){ // for every row
					// Check if pixel is out of bounds (improved version)...
					if (x + i < 0      || y + j < 0      ) continue;
					if (x + i >= width || y + j >= height) continue;
					if (innerX + i >= 128 || innerY + j >= 128) continue;
	
					sum.x += (int) s_input[innerX + i][innerY + j].x;
					sum.y += (int) s_input[innerX + i][innerY + j].y;
					sum.z += (int) s_input[innerX + i][innerY + j].z;
	
					count++;
				}
			}

			outputPixels[linearCoord].x = sum.x / count;
			outputPixels[linearCoord].y = sum.y / count;
			outputPixels[linearCoord].z = sum.z / count;

		}
	}

	//Assign IDs to threads
	//distribute work between threads
	//do the computation and store the output pixels in outputPixels

}

#endif // _IMAGEFILTER_KERNEL_H_
