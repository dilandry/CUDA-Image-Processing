/*$Id: imageFilter.cu 2016-03-04 18:27:54 (author: Reza Mokhtari)$*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <ctype.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <sys/time.h>

#include "imageFilter_kernel.cu"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>
#include <stopwatch.h>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>
#include <stopwatch.h>
#include <cmath>

#include "cudaImageHost.h"
#include "cudaImageDevice.h.cu"
#include "cudaConvUtilities.h.cu"
#include "cudaConvolution.h.cu"
#include "cudaMorphology.h.cu"
#include "ImageWorkbench.h.cu"

#define IMG_DATA_OFFSET_POS 10
#define BITS_PER_PIXEL_POS 28

int swap;
void test_endianess();
void runMorphologyUnitTests(void);
void swap_bytes(char *bytes, int num_bytes);

int main(int argc, char *argv[]) 
{
	int i;
	int fd;
	unsigned char *fdata; //DL: Changed to unsigned
	struct stat finfo;
	char * inputfname;
	char * outputfname;

	if (argc < 4)
	{
		printf("USAGE: %s <bitmap input filename> <bitmap output file name> <part specifier>\n", argv[0]);
		exit(1);
	}

	inputfname = argv[1];
	outputfname = argv[2];
	char partId = argv[3][0];
	if(partId != 'a' && partId != 'b' && partId != 'c')
	{
		printf("Please provide a part specifier: a, b, or c\n");
		exit(1);
	}

	printf("Image filter: Running...\n");

	fd = open(inputfname, O_RDONLY);
	fstat(fd, &finfo);

	fdata = (unsigned char*) malloc(finfo.st_size); //DL: Changed to unsigned
	
	read (fd, fdata, finfo.st_size);

	if ((fdata[0] != 'B') || (fdata[1] != 'M')) 
	{
		printf("File is not a valid bitmap file. Terminating the program\n");
		exit(1);
	}

	test_endianess();     // will set the variable "swap"

	unsigned short *bitsperpixel = (unsigned short *)(&(fdata[BITS_PER_PIXEL_POS]));
	if (swap) 
	{
		printf("swapping\n");
		swap_bytes((char *)(bitsperpixel), sizeof(*bitsperpixel));
	}

 	// ensure its 3 bytes per pixel
	if (*bitsperpixel != 24) 
	{
		printf("Error: Invalid bitmap format - ");
		printf("This application only accepts 24-bit pictures. Exiting\n");
		exit(1);
	}

	unsigned short *data_pos = (unsigned short *)(&(fdata[IMG_DATA_OFFSET_POS]));
	if (swap) 
	{
		swap_bytes((char *)(data_pos), sizeof(*data_pos));
	}

	int imgdata_bytes = (int)finfo.st_size - (int)(*(data_pos));
	printf("This file has %d bytes of image data, %d pixels\n", imgdata_bytes, imgdata_bytes / 3);

	int width = *((int*)&fdata[18]);
	printf("Width: %d\n", width);
	int height = *((int*)&fdata[22]);
	printf("Height: %d\n", height);

	int fileSize = (int) finfo.st_size;	

	//p will point to the first pixel
	unsigned char* p = &(fdata[*data_pos]); //DL: Changed to unsigned

	//Set the number of blocks and threads
	//CDL--dim3 grid(1024, 1, 1/*x, y, x*/);
	//CDL--dim3 block(12, 1, 1/*x, y, x*/);
	dim3 grid(12, 1, 1/*x, y, x*/);
	dim3 block(1024, 1, 1/*x, y, x*/);

	unsigned char* d_inputPixels;
	cudaMalloc((void**) &d_inputPixels, width * height * 3);
	cudaMemcpy(d_inputPixels, p, width * height * 3, cudaMemcpyHostToDevice);
	
	unsigned char* d_outputPixels;
	cudaMalloc((void**) &d_outputPixels, width * height * 3);
	cudaMemset(d_outputPixels, 0, width * height * 3);

	struct timeval start_tv, end_tv;
	time_t sec;
	time_t ms;
	time_t diff;
	gettimeofday(&start_tv, NULL);


	if(partId == 'a')
	{
		imageFilterKernelPartA<<<grid, block>>>((uchar3*) d_inputPixels, (uchar3*) d_outputPixels, width, height /*, other arguments */); // Changed to uchar3
	} 
	else if(partId == 'b')
	{
		imageFilterKernelPartB<<<grid, block>>>((uchar3*) d_inputPixels, (uchar3*) d_outputPixels, width, height /*, other arguments */); // Changed to uchar3
	}
	else if(partId == 'c')
	{
		imageFilterKernelPartC<<<grid, block>>>((uchar3*) d_inputPixels, (uchar3*) d_outputPixels, width, height /*, other arguments */); // Changed to uchar3
	}

	cudaThreadSynchronize();

	gettimeofday(&end_tv, NULL);
	sec = end_tv.tv_sec - start_tv.tv_sec;
	ms = end_tv.tv_usec - start_tv.tv_usec;

	diff = sec * 1000000 + ms;

	printf("%10s:\t\t%fms\n", "Time elapsed", (double)((double)diff/1000.0));


	char* outputPixels = (char*) malloc(height * width * 3);
	cudaMemcpy(outputPixels, d_outputPixels, height * width * 3, cudaMemcpyDeviceToHost);

	memcpy(&(fdata[*data_pos]), outputPixels, height * width * 3);

	FILE *writeFile; 
	writeFile = fopen(outputfname,"w+");
	for(i = 0; i < fileSize; i++)
		fprintf(writeFile,"%c", fdata[i]);
	fclose(writeFile);

	runMorphologyUnitTests();

	return 0;
} 

void test_endianess() {
    unsigned int num = 0x12345678;
    char *low = (char *)(&(num));
    if (*low ==  0x78) {
        //dprintf("No need to swap\n");
        swap = 0;
    }
    else if (*low == 0x12) {
        //dprintf("Need to swap\n");
        swap = 1;
    }
    else {
        printf("Error: Invalid value found in memory\n");
        exit(1);
    } 
}

////////////////////////////////////////////////////////////////////////////////
void runMorphologyUnitTests()
{
   cout << endl << "Executing morphology unit tests (no workbench)..." << endl;

   /////////////////////////////////////////////////////////////////////////////
   // Allocate host memory and read in the test image from file
   /////////////////////////////////////////////////////////////////////////////
   unsigned int imgW  = 256;
   unsigned int imgH  = 256;
   unsigned int nPix  = imgW*imgH;
   string fn("salt256.txt");

   printf("\nTesting morphology operations on %dx%d mask.\n", imgW,imgH);
   cout << "Reading mask from " << fn.c_str() << endl << endl;

   cudaImageHost<int> imgIn(fn, imgW, imgH);
   cudaImageHost<int> imgOut(imgW, imgH);

   imgIn.writeFile("ImageIn.txt");


   // A very unique SE for checking coordinate systems
   int se17H = 17;
   int se17W = 17;
   cudaImageHost<int> se17("asymmPSF_17x17.txt", se17W, se17H);

   // Circular SE from utilities file
   int seCircD = 5;
   cudaImageHost<int> seCirc(seCircD, seCircD);
   int seCircNZ = createBinaryCircle(seCirc.getDataPtr(), seCircD); // return #non-zero

   // Display the two SEs
   cout << "Using the unique, 17x17 structuring element:" << endl;
   se17.printMask('.','0');
   cout << "Other tests using basic circular SE:" << endl;
   seCirc.printMask('.','0');

   // Allocate Device Memory
   cudaImageDevice<int> devIn(imgIn);
   cudaImageDevice<int> devPsf(se17);
   cudaImageDevice<int> devOut(imgW, imgH);

   cudaImageDevice<int>::calculateDeviceMemoryUsage(true);


   int bx = 8;
   int by = 32;
   int gx = imgW/bx;
   int gy = imgH/by;
   dim3 BLOCK1D( bx*by, 1, 1);
   dim3 GRID1D(  nPix/(bx*by), 1, 1);
   dim3 BLOCK2D( bx, by, 1);
   dim3 GRID2D(  gx, gy, 1);

   /////////////////////////////////////////////////////////////////////////////
   // TEST THE GENERIC/UNIVERSAL MORPHOLOGY OPS
   // Non-zero elts = 134, so use -133 for dilate
   Morph_Generic_Kernel<<<GRID2D,BLOCK2D>>>(devIn, devOut, imgW, imgH, 
                                                devPsf, se17H/2, se17W/2, -133);
   cutilCheckMsg("Kernel execution failed");  // Check if kernel exec failed
   devOut.copyToHost(imgOut);
   imgOut.writeFile("ImageDilate.txt");

   // Non-zero elts = 134, so use 134 for erod
   Morph_Generic_Kernel<<<GRID2D,BLOCK2D>>>(devIn, devOut, imgW, imgH, 
                                                devPsf, se17H/2, se17W/2, 134);
   cutilCheckMsg("Kernel execution failed");  // Check if kernel exec failed
   devOut.copyToHost(imgOut);
   imgOut.writeFile("ImageErode.txt");

   /////////////////////////////////////////////////////////////////////////////
   // We also need to verify that the 3x3 optimized functions work
   Morph3x3_Dilate_Kernel<<<GRID2D,BLOCK2D>>>(devIn, devOut, imgW, imgH);
   cutilCheckMsg("Kernel execution failed");  // Check if kernel exec failed
   devOut.copyToHost(imgOut);
   imgOut.writeFile("Image3x3_dilate.txt");

   Morph3x3_Erode4connect_Kernel<<<GRID2D,BLOCK2D>>>(devIn, devOut, imgW, imgH);
   cutilCheckMsg("Kernel execution failed");  // Check if kernel exec failed
   devOut.copyToHost(imgOut);
   imgOut.writeFile("Image3x3_erode.txt");

   Morph3x3_Thin8_Kernel<<<GRID2D,BLOCK2D>>>(devOut, devIn, imgW, imgH);
   cutilCheckMsg("Kernel execution failed");  // Check if kernel exec failed
   devIn.copyToHost(imgIn);
   imgIn.writeFile("Image3x3_erode_thin.txt");
   /////////////////////////////////////////////////////////////////////////////
}


void swap_bytes(char *bytes, int num_bytes) 
{
    int i;
    char tmp;
    
    for (i = 0; i < num_bytes/2; i++) {
        //dprintf("Swapping %d and %d\n", bytes[i], bytes[num_bytes - i - 1]);
        tmp = bytes[i];
        bytes[i] = bytes[num_bytes - i - 1];
        bytes[num_bytes - i - 1] = tmp;    
    }
}


