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
#include "cudaConvolution.cu"

#define IMG_DATA_OFFSET_POS 10
#define BITS_PER_PIXEL_POS 28

int swap2;
void blurCPU(uchar3*, int, int, char3*);
void test_endianess();
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
    printf("\t1: BoxBlur\n");
    printf("\t2: Sharpen\n");
    printf("\t3: Sobel\n");
    printf("\t4: Blur filter executed on CPU\n");
    exit(1);
  }

  inputfname = argv[1];
  outputfname = argv[2];
  char partId = argv[3][0];
  if(partId != '1' && partId != '2' && partId != '3' && partId != '4')
  {
    printf("Please provide a filter number\n");
    printf("\t1: BoxBlur\n");
    printf("\t2: Sharpen\n");
    printf("\t3: Sobel\n");
    printf("\t4: Blur filter executed on CPU\n");
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
  if (swap2) 
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
  if (swap2) 
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

  uchar3* p_uchar3 = (uchar3*) malloc(height * width *3);
  memcpy(p_uchar3, p, height * width * 3);

  // Allocate output pixel array
  char3* outputPixelsCPU = (char3*) malloc(height * width*3);

  //Set the number of blocks and threads
  //CDL--dim3 grid(1024, 1, 1/*x, y, x*/);
  //CDL--dim3 block(12, 1, 1/*x, y, x*/);
  dim3 grid(12, 1, 1/*x, y, x*/);
  dim3 block(1024, 1, 1/*x, y, x*/);
  
  unsigned char* d_outputPixels;
  cudaMalloc((void**) &d_outputPixels, width * height * 3);
  cudaMemset(d_outputPixels, 0, width * height * 3);

  struct timeval start_tv, end_tv;
  time_t sec;
  time_t ms;
  time_t diff;
  gettimeofday(&start_tv, NULL);

  // Blur on CPU, if that option was selected
  if (partId == '4'){
    blurCPU(p_uchar3, width, height, outputPixelsCPU);
  } else {
    unsigned char* d_inputPixels;
    cudaMalloc((void**) &d_inputPixels, width * height * 3);
    cudaMemcpy(d_inputPixels, p, width * height * 3, cudaMemcpyHostToDevice);

    block.x = 128;
    block.y = 8;
    grid.x = 4;
    grid.y = 3;

    imageFilterKernel<<<grid, block>>>((uchar3*) d_inputPixels, (uchar3*) d_outputPixels, width, height, partId /*, other arguments */); // Changed to uchar3
    //convolveBasicNb2<<<grid, block>>>((uchar3*) d_inputPixels, (uchar3*) d_outputPixels, width, height, 4, 4 /*, other arguments */); // Changed to uchar3
  }

  cudaThreadSynchronize();

  gettimeofday(&end_tv, NULL);
  sec = end_tv.tv_sec - start_tv.tv_sec;
  ms = end_tv.tv_usec - start_tv.tv_usec;

  diff = sec * 1000000 + ms;

  printf("%10s:\t\t%fms\n", "Time elapsed", (double)((double)diff/1000.0));


  char* outputPixels = (char*) malloc(height * width * 3);
  cudaMemcpy(outputPixels, d_outputPixels, height * width * 3, cudaMemcpyDeviceToHost);

  if (partId == '4'){
    memcpy(&(fdata[*data_pos]), outputPixelsCPU, height * width * 3);
  } else {
    memcpy(&(fdata[*data_pos]), outputPixels, height * width * 3);
  }

  FILE *writeFile; 
  writeFile = fopen(outputfname,"w+");
  for(i = 0; i < fileSize; i++)
    fprintf(writeFile,"%c", fdata[i]);
  fclose(writeFile);

  return 0;
} 

// This function applies the blur filter to the image using the CPU
void blurCPU(uchar3* p_uchar3, int width, int height, char3* outputPixelsCPU) {
  int radius = 4;

  // For each pixel...
  for(int i=0; i<height; i++){
    for(int j=0; j<width; j++){
      int3 sum = make_int3(0,0,0);
      int count = 0;

      // Apply blur...
      for (int i_off = -radius; i_off<=radius; i_off++){
        for (int j_off = -radius; j_off<=radius; j_off++){
          if (((i+i_off)*width + j+j_off) >= width*height 
            || ((i+i_off)*width + j+j_off) >= width*height) continue;
          if (((i+i_off)*width + j+j_off) < 0 
            || ((i+i_off)*width + j+j_off) < 0) continue;

          sum.x += p_uchar3[((i+i_off)*width + j+j_off)].x;
          sum.y += p_uchar3[((i+i_off)*width + j+j_off)].y;
          sum.z += p_uchar3[((i+i_off)*width + j+j_off)].z;
          count++;
        }
      }
      outputPixelsCPU[IDX_1D(i,j,width)] = make_char3(sum.x/count, sum.y/count, sum.z/count);
    }
  }
}

void test_endianess() {
    unsigned int num = 0x12345678;
    char *low = (char *)(&(num));
    if (*low ==  0x78) {
        //dprintf("No need to swap\n");
        swap2 = 0;
    }
    else if (*low == 0x12) {
        //dprintf("Need to swap\n");
        swap2 = 1;
    }
    else {
        printf("Error: Invalid value found in memory\n");
        exit(1);
    } 
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


