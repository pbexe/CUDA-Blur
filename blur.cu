#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <cuda.h>
#include <stdlib.h>

#define IMAGE_HEIGHT 521
#define IMAGE_WIDTH 428

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void blur(int *d_R, int *d_G, int *d_B, int *d_Rnew, int *d_Gnew, int *d_Bnew)
{
  // Get the X and y coords of the pixel for this thread
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  // Stop the thread if it is not part of the image
	if (x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT) {
      return;
  }

  // Apply the box blur
  if (y != 0 && y != (IMAGE_HEIGHT-1) && x != 0 && x != (IMAGE_WIDTH-1)){
    d_Rnew[(IMAGE_WIDTH*y) + x] = (d_R[(IMAGE_WIDTH*y+1) + x]+d_R[(IMAGE_WIDTH*y-1) + x]+d_R[(IMAGE_WIDTH*y) + x+1]+d_R[(IMAGE_WIDTH*y) + x-1])/4;
    d_Gnew[(IMAGE_WIDTH*y) + x] = (d_G[(IMAGE_WIDTH*y+1) + x]+d_G[(IMAGE_WIDTH*y-1) + x]+d_G[(IMAGE_WIDTH*y) + x+1]+d_G[(IMAGE_WIDTH*y) + x-1])/4;
    d_Bnew[(IMAGE_WIDTH*y) + x] = (d_B[(IMAGE_WIDTH*y+1) + x]+d_B[(IMAGE_WIDTH*y-1) + x]+d_B[(IMAGE_WIDTH*y) + x+1]+d_B[(IMAGE_WIDTH*y) + x-1])/4;
  }
  else if (y == 0 && x != 0 && x != (IMAGE_WIDTH-1)){
    d_Rnew[(IMAGE_WIDTH*y) + x] = (d_R[(IMAGE_WIDTH*y+1) + x]+d_R[(IMAGE_WIDTH*y) + x+1]+d_R[(IMAGE_WIDTH*y) + x-1])/3;
    d_Gnew[(IMAGE_WIDTH*y) + x] = (d_G[(IMAGE_WIDTH*y+1) + x]+d_G[(IMAGE_WIDTH*y) + x+1]+d_G[(IMAGE_WIDTH*y) + x-1])/3;
    d_Bnew[(IMAGE_WIDTH*y) + x] = (d_B[(IMAGE_WIDTH*y+1) + x]+d_B[(IMAGE_WIDTH*y) + x+1]+d_B[(IMAGE_WIDTH*y) + x-1])/3;
  }
  else if (y == (IMAGE_HEIGHT-1) && x != 0 && x != (IMAGE_WIDTH-1)){
    d_Rnew[(IMAGE_WIDTH*y) + x] = (d_R[(IMAGE_WIDTH*y-1) + x]+d_R[(IMAGE_WIDTH*y) + x+1]+d_R[(IMAGE_WIDTH*y) + x-1])/3;
    d_Gnew[(IMAGE_WIDTH*y) + x] = (d_G[(IMAGE_WIDTH*y-1) + x]+d_G[(IMAGE_WIDTH*y) + x+1]+d_G[(IMAGE_WIDTH*y) + x-1])/3;
    d_Bnew[(IMAGE_WIDTH*y) + x] = (d_B[(IMAGE_WIDTH*y-1) + x]+d_B[(IMAGE_WIDTH*y) + x+1]+d_B[(IMAGE_WIDTH*y) + x-1])/3;
  }
  else if (x == 0 && y != 0 && y != (IMAGE_HEIGHT-1)){
    d_Rnew[(IMAGE_WIDTH*y) + x] = (d_R[(IMAGE_WIDTH*y+1) + x]+d_R[(IMAGE_WIDTH*y-1) + x]+d_R[(IMAGE_WIDTH*y) + x+1])/3;
    d_Gnew[(IMAGE_WIDTH*y) + x] = (d_G[(IMAGE_WIDTH*y+1) + x]+d_G[(IMAGE_WIDTH*y-1) + x]+d_G[(IMAGE_WIDTH*y) + x+1])/3;
    d_Bnew[(IMAGE_WIDTH*y) + x] = (d_B[(IMAGE_WIDTH*y+1) + x]+d_B[(IMAGE_WIDTH*y-1) + x]+d_B[(IMAGE_WIDTH*y) + x+1])/3;
  }
  else if (x == (IMAGE_WIDTH-1) && y != 0 && y != (IMAGE_HEIGHT-1)){
    d_Rnew[(IMAGE_WIDTH*y) + x] = (d_R[(IMAGE_WIDTH*y+1) + x]+d_R[(IMAGE_WIDTH*y-1) + x]+d_R[(IMAGE_WIDTH*y) + x-1])/3;
    d_Gnew[(IMAGE_WIDTH*y) + x] = (d_G[(IMAGE_WIDTH*y+1) + x]+d_G[(IMAGE_WIDTH*y-1) + x]+d_G[(IMAGE_WIDTH*y) + x-1])/3;
    d_Bnew[(IMAGE_WIDTH*y) + x] = (d_B[(IMAGE_WIDTH*y+1) + x]+d_B[(IMAGE_WIDTH*y-1) + x]+d_B[(IMAGE_WIDTH*y) + x-1])/3;
  }
  else if (y==0 &&x==0){
    d_Rnew[(IMAGE_WIDTH*y) + x] = (d_R[(IMAGE_WIDTH*y) + x+1]+d_R[(IMAGE_WIDTH*y+1) + x])/2;
    d_Gnew[(IMAGE_WIDTH*y) + x] = (d_G[(IMAGE_WIDTH*y) + x+1]+d_G[(IMAGE_WIDTH*y+1) + x])/2;
    d_Bnew[(IMAGE_WIDTH*y) + x] = (d_B[(IMAGE_WIDTH*y) + x+1]+d_B[(IMAGE_WIDTH*y+1) + x])/2;
  }
  else if (y==0 &&x==(IMAGE_WIDTH-1)){
    d_Rnew[(IMAGE_WIDTH*y) + x] = (d_R[(IMAGE_WIDTH*y) + x-1]+d_R[(IMAGE_WIDTH*y+1) + x])/2;
    d_Gnew[(IMAGE_WIDTH*y) + x] = (d_G[(IMAGE_WIDTH*y) + x-1]+d_G[(IMAGE_WIDTH*y+1) + x])/2;
    d_Bnew[(IMAGE_WIDTH*y) + x] = (d_B[(IMAGE_WIDTH*y) + x-1]+d_B[(IMAGE_WIDTH*y+1) + x])/2;
  }
  else if (y==(IMAGE_HEIGHT-1) &&x==0){
    d_Rnew[(IMAGE_WIDTH*y) + x] = (d_R[(IMAGE_WIDTH*y) + x+1]+d_R[(IMAGE_WIDTH*y-1) + x])/2;
    d_Gnew[(IMAGE_WIDTH*y) + x] = (d_G[(IMAGE_WIDTH*y) + x+1]+d_G[(IMAGE_WIDTH*y-1) + x])/2;
    d_Bnew[(IMAGE_WIDTH*y) + x] = (d_B[(IMAGE_WIDTH*y) + x+1]+d_B[(IMAGE_WIDTH*y-1) + x])/2;
  }
  else if (y==(IMAGE_HEIGHT-1) &&x==(IMAGE_WIDTH-1)){
    d_Rnew[(IMAGE_WIDTH*y) + x] = (d_R[(IMAGE_WIDTH*y) + x-1]+d_R[(IMAGE_WIDTH*y-1) + x])/2;
    d_Gnew[(IMAGE_WIDTH*y) + x] = (d_G[(IMAGE_WIDTH*y) + x-1]+d_G[(IMAGE_WIDTH*y-1) + x])/2;
    d_Bnew[(IMAGE_WIDTH*y) + x] = (d_B[(IMAGE_WIDTH*y) + x-1]+d_B[(IMAGE_WIDTH*y-1) + x])/2;
  }

}

int main (int argc, const char * argv[]) {
  struct timeval tim;
	gettimeofday(&tim, NULL);
	static int const maxlen = 200, rowsize = 521, colsize = 428, linelen = 12;
	char str[maxlen], lines[5][maxlen];
	FILE *fp, *fout;
	int nlines = 0;
	unsigned int h1, h2, h3;
	char *sptr;
	int R[rowsize][colsize], G[rowsize][colsize], B[rowsize][colsize];
	int row = 0, col = 0, nblurs, lineno=0, k;

	fp = fopen("David.ps", "r");

	while(! feof(fp))
	{
		fscanf(fp, "\n%[^\n]", str);
		if (nlines < 5) {strcpy((char *)lines[nlines++],(char *)str);}
		else{
			for (sptr=&str[0];*sptr != '\0';sptr+=6){
				sscanf(sptr,"%2x",&h1);
				sscanf(sptr+2,"%2x",&h2);
				sscanf(sptr+4,"%2x",&h3);

				if (col==colsize){
					col = 0;
					row++;
				}
				if (row < rowsize) {
					R[row][col] = h1;
					G[row][col] = h2;
					B[row][col] = h3;
				}
				col++;
			}
		}
	}
	fclose(fp);
  // Number of blur iterations
	nblurs = atoi(argv[1]);
  // Start the timer
	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
  // The size of the 1D arrays for the GPU
  int size = sizeof(int) * IMAGE_WIDTH * IMAGE_HEIGHT;
  // Initialise the arrays to hold the flatened image
  int *flat_R, *flat_G, *flat_B;
  flat_R = (int *)malloc(size);
  flat_G = (int *)malloc(size);
  flat_B = (int *)malloc(size);
  // Create pointers to GPU array locations
  int *d_R, *d_G, *d_B, *d_Rnew, *d_Gnew, *d_Bnew;
  // Define how many threads per block
  int numBlocksY = ceil(IMAGE_HEIGHT/16.0);
  int numBlocksX = ceil(IMAGE_WIDTH/16.0);
  dim3 dimBlock(numBlocksX,numBlocksY);
  // Define how many blocks per grid
  dim3 dimGrid(16, 16);
  // Allocate GPU mem for the 1D arrays
  cudaMalloc((void **)&d_R, size);
  cudaMalloc((void **)&d_G, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_Rnew, size);
  cudaMalloc((void **)&d_Bnew, size);
  cudaMalloc((void **)&d_Gnew, size);
  // Pointers to handle the output
  int *h_R, *h_G, *h_B;
  h_R = (int *)malloc(size);
  h_G = (int *)malloc(size);
  h_B = (int *)malloc(size);
  // Start the blur loop
  for(k=0;k<nblurs;k++){
    // Flatten the 2D arrays to make them easier to handle with CUDA
  	for (int row=0;row<IMAGE_HEIGHT;row++){
  		for (int col=0;col<IMAGE_WIDTH;col++){
  			flat_R[IMAGE_WIDTH*row+col] = R[row][col];
  			flat_G[IMAGE_WIDTH*row+col] = G[row][col];
  			flat_B[IMAGE_WIDTH*row+col] = B[row][col];
  		}
  	}
    // Copy these arrays to the GPU
  	cudaMemcpy(d_R, flat_R, size, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_G, flat_G, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flat_B, size, cudaMemcpyHostToDevice);


    // Punch it Chewie
  	blur<<<dimGrid, dimBlock>>>(d_R, d_G, d_B, d_Rnew, d_Gnew, d_Bnew);

    // Copy the modified values out of the GPU
  	cudaMemcpy(h_R, d_Rnew, size, cudaMemcpyDeviceToHost);
  	cudaMemcpy(h_G, d_Gnew, size, cudaMemcpyDeviceToHost);
  	cudaMemcpy(h_B, d_Bnew, size, cudaMemcpyDeviceToHost);
    // Check for errors
  	gpuErrchk( cudaPeekAtLastError() );
  	gpuErrchk( cudaDeviceSynchronize() );
    // Convert the 1D arrays back into 2D
  	for (int row=0;row<IMAGE_HEIGHT;row++){
  		for (int col=0;col<IMAGE_WIDTH;col++){
  			R[row][col] = h_R[IMAGE_WIDTH*row+col];
  			G[row][col] = h_G[IMAGE_WIDTH*row+col];
  			B[row][col] = h_B[IMAGE_WIDTH*row+col];
  		}
  	}
  }
  // Free up the allocated memory
  cudaFree(d_R);
  cudaFree(d_G);
  cudaFree(d_B);
  cudaFree(d_Rnew);
  cudaFree(d_Gnew);
  cudaFree(d_Bnew);
  free(h_R);
  free(h_G);
  free(h_B);

	fout= fopen("DavidBlur.ps", "w");
	for (k=0;k<nlines;k++) fprintf(fout,"\n%s", lines[k]);
	fprintf(fout,"\n");
	for(row=0;row<rowsize;row++){
		for (col=0;col<colsize;col++){
			fprintf(fout,"%02x%02x%02x",R[row][col],G[row][col],B[row][col]);
			lineno++;
			if (lineno==linelen){
				fprintf(fout,"\n");
				lineno = 0;
			}
		}
	}
	fclose(fout);
  gettimeofday(&tim, NULL);
	double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed\n", t2-t1);
  return 0;
}
