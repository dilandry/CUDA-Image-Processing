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

int swap_2;
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
	if (swap_2) 
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
	if (swap_2) 
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

	runWorkbenchUnitTests();

	return 0;
} 

void test_endianess() {
    unsigned int num = 0x12345678;
    char *low = (char *)(&(num));
    if (*low ==  0x78) {
        //dprintf("No need to swap\n");
        swap_2 = 0;
    }
    else if (*low == 0x12) {
        //dprintf("Need to swap\n");
        swap_2 = 1;
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
   unsigned int imgW  = 1280;
   unsigned int imgH  = 854;
   unsigned int nPix  = imgW*imgH;
   //CDL--string fn("salt256.txt");
   string fn("sample.bmp");

   printf("\nTesting morphology operations on %dx%d mask.\n", imgW,imgH);
   cout << "Reading mask from " << fn.c_str() << endl << endl;

   cudaImageHost<int> imgIn(fn, imgW, imgH);
   cudaImageHost<int> imgOut(imgW, imgH);

   imgIn.writeFile("ImageIn.txt");
   //CDL--imgIn.writeFile("sample.bmp");

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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void runWorkbenchUnitTests(void)
{
   cout << "****************************************";
   cout << "***************************************" << endl;
   cout << "***Testing ImageWorkbench basic operations" << endl << endl;

   // Read the salt image from file
   //CDL--cudaImageHost<int> imgIn("salt256.txt", 256, 256);
   cudaImageHost<int> imgIn("sample.bmp", 1280, 854);

   // Create a place to put the result
   cudaImageHost<int> imgOut(1280, 854);

   // A very unique SE for checking coordinate systems
   cudaImageHost<int> se17("asymmPSF_17x17.txt", 17, 17);

   // Circular SE from utilities file
   int seCircD = 11;
   cudaImageHost<int> seCirc(seCircD, seCircD);
   createBinaryCircle(seCirc.getDataPtr(), seCircD);

   // Check that rectangular SEs work, too
   int rectH = 5;
   int rectW = 9;
   cudaImageHost<int> seRect(rectH, rectW);
   for(int r=0; r<rectH; r++)
      for(int c=0; c<rectW; c++)
         seRect(r, c) = 1;


   // The SEs are added to the static, master SE list in ImageWorkbench, and
   // are used by giving the index into that list (returned by addStructElt())
   cout << "Adding unique SE to list" << endl;
   se17.printMask();
   int seIdxUnique17 = ImageWorkbench::addStructElt(se17);

   cout << "Adding circular SE to list" << endl;
   seCirc.printMask();
   int seIdxCircle11 = ImageWorkbench::addStructElt(seCirc);

   cout << "Adding rectangular SE to list" << endl;
   seRect.printMask();
   int seIdxRect9x5  = ImageWorkbench::addStructElt(seRect);
   
   cudaImageDevice<int>::calculateDeviceMemoryUsage(true);  // printToStdOut==true


   /////////////////////////////////////////////////////////////////////////////
   // Let's start testing ImageWorkbench
   /////////////////////////////////////////////////////////////////////////////
   // Create the workbench, which copies the image into device memory
   ImageWorkbench theIwb(imgIn);

   // Start by simply fetching the unmodified image (sanity check)
   cout << "Copying unaltered image back to host for verification" << endl;
   theIwb.copyBufferToHost(imgOut);
   imgOut.writeFile("Workbench1_In.txt");
   
   // Dilate by the circle
   cout << "Dilating with 11x11 circle" << endl;
   theIwb.Dilate(seIdxCircle11);
   theIwb.copyBufferToHost(imgOut);
   imgOut.writeFile("Workbench2_DilateCirc.txt");

   // We Erode the image now, but with the basic 3x3
   cout << "Performing simple 3x3 erode" << endl;
   theIwb.Erode();
   theIwb.copyBufferToHost(imgOut);
   imgOut.writeFile("Workbench3_Erode3.txt");

   // We Erode the image now, but with the basic 3x3
   cout << "Try a closing operation" << endl;
   theIwb.Close(seIdxRect9x5);
   theIwb.copyBufferToHost(imgOut);
   imgOut.writeFile("Workbench4_Close.txt");

   // We now test subtract by eroding an image w/ 3x3 and subtracting from original
   // Anytime we manually select src/dst for image operations, make sure we end up
   // with the final result in buffer A, or in buffer B with a a call to flipBuffers()
   // to make sure that our input/output locations are consistent
   ImageWorkbench iwb2(imgIn);
   cout << "Testing subtract kernel" << endl;
   
   iwb2.Dilate();
   iwb2.Dilate();
   iwb2.copyBufferToHost(imgOut);
   imgOut.writeFile("Workbench5a_dilated.txt");

   iwb2.Erode(A, 1);  // put result in buffer 1, don't flip
   iwb2.copyBufferToHost(1, imgOut);
   imgOut.writeFile("Workbench5b_erode.txt");
   
   iwb2.Subtract(1, A, A);
   iwb2.copyBufferToHost(imgOut);  // default is always the input buffer A
   imgOut.writeFile("Workbench5c_subtract.txt");

   cudaImageHost<int> cornerDetect(3,3);
   cornerDetect(0,0) = -1;  cornerDetect(1,0) = -1;  cornerDetect(2,0) = 0;
   cornerDetect(0,1) = -1;  cornerDetect(1,1) =  1;  cornerDetect(2,1) = 1;
   cornerDetect(0,2) =  0;  cornerDetect(1,2) =  1;  cornerDetect(2,2) = 0;
   int seIdxCD = ImageWorkbench::addStructElt(cornerDetect);
   iwb2.FindAndRemove(seIdxCD);
   iwb2.copyBufferToHost(imgOut);
   imgOut.writeFile("Workbench5d_findandrmv.txt");

   cout << endl << "Checking device memory usage so far: " << endl;
   cudaImageDevice<int>::calculateDeviceMemoryUsage(true);  // printToStdOut==true

   /////////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////
   // With a working workbench, we can finally SOLVE A MAZE !!
   cout << endl << "Time to solve a maze! " << endl << endl;
   cudaImageHost<int> mazeImg("elephantmaze.txt", 512, 512);
   ImageWorkbench iwbMaze(mazeImg);

   // Morph-close the image [for fun, not necessary], write it to file for ref
   iwbMaze.Close();  
   iwbMaze.copyBufferToHost(imgOut);
   imgOut.writeFile("MazeTxt1_In.txt");

   // Start thinning
   cout << "\tThinning sweep 2x" << endl;
   iwbMaze.ThinningSweep();
   iwbMaze.ThinningSweep();
   iwbMaze.copyBufferToHost(imgOut);
   imgOut.writeFile("MazeTxt2_Thin2x.txt");


   // Finish thinning by checking when the image is no longer changing
   cout << "\tThinning sweep til complete" << endl;
   int thinOps = 2;
   int diff=-1;
   while(diff != 0)
   {
      iwbMaze.ThinningSweep();
      diff = iwbMaze.CountChanged();
      thinOps++;
   }
   iwbMaze.copyBufferToHost(imgOut);
   imgOut.writeFile("MazeTxt3_ThinComplete.txt");

   cout << "\tPruning sweep 1-5" << endl;
   int pruneOps = 0;
   for(int i=0; i<5; i++)
   {
      iwbMaze.PruningSweep();
      pruneOps++;
   }
   iwbMaze.copyBufferToHost(imgOut);
   imgOut.writeFile("MazeTxt4_Prune5x.txt");

   cout << "\tPruning sweep 6-20" << endl;
   for(int i=0; i<15; i++)
   {
      iwbMaze.PruningSweep();
      pruneOps++;
   }
   iwbMaze.copyBufferToHost(imgOut);
   imgOut.writeFile("MazeTxt5_Prune20x.txt");

   diff=-1;
   cout << "\tPruning sweep until complete" << endl;
   while(diff != 0)
   {
      iwbMaze.PruningSweep();
      diff = iwbMaze.CountChanged();
      pruneOps++;
   }
   iwbMaze.copyBufferToHost(imgOut);
   imgOut.writeFile("MazeTxt6_PruneComplete.txt");

   int totalHomOps = 8*(thinOps + pruneOps);
   cout << "Finished the maze!  Total operations: " << endl
        << "\t" << thinOps  << " thinning sweeps and " << endl
        << "\t" << pruneOps << " pruning sweeps" << endl
        << "\tTotal of " << totalHomOps << " HitOrMiss operations and the same "
        << "number of subtract operations" << endl << endl;


   // Check to see how much device memory we're using right now
   cudaImageDevice<int>::calculateDeviceMemoryUsage(true);  // printToStdOut==true

   cout << "Finished IWB testing!" << endl;
   cout << "****************************************";
   cout << "***************************************" << endl;

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


