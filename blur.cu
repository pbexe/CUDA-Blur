#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <cuda.h>

#define IMAGE_WIDTH 428
#define IMAGE_HEIGHT 521

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
void blur(int *d_R,int *d_G, int *d_B)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT) {
      return;
  }
  int myval = d_R[(IMAGE_WIDTH*x) + y];
  d_R[(IMAGE_WIDTH*x) + y] = d_R[(IMAGE_WIDTH*x) + y] / 2;
  d_G[(IMAGE_WIDTH*x) + y] = d_G[(IMAGE_WIDTH*x) + y] / 2;
  d_B[(IMAGE_WIDTH*x) + y] = d_B[(IMAGE_WIDTH*x) + y] / 2;
}

int main (int argc, const char * argv[]) {
	static int const maxlen = 200, rowsize = 521, colsize = 428, linelen = 12;
	char str[maxlen], lines[5][maxlen];
	FILE *fp, *fout;
	int nlines = 0;
	unsigned int h1, h2, h3;
	char *sptr;
	int R[rowsize][colsize], G[rowsize][colsize], B[rowsize][colsize];
	int Rnew[rowsize][colsize], Gnew[rowsize][colsize], Bnew[rowsize][colsize];
	int row = 0, col = 0, nblurs, lineno=0, k;
	struct timeval tim;
	gettimeofday(&tim, NULL);

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

	nblurs = 1;
	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);

	for(k=0;k<nblurs;k++){
		int flat_R[rowsize * colsize];
		int flat_G[rowsize * colsize];
		int flat_B[rowsize * colsize];
		for (int row=0;row<rowsize;row++){
			for (int col=0;col<colsize;col++){
				flat_R[colsize*row+col] = R[row][col];
				flat_G[colsize*row+col] = G[row][col];
				flat_B[colsize*row+col] = B[row][col];
			}
		}
		int *d_R, *d_G, *d_B;
		int size = sizeof(int) * colsize * rowsize;
		// printf("%d\n", size);
		cudaMalloc((void **)&d_R, size);
		cudaMemcpy(d_R, flat_R, size, cudaMemcpyHostToDevice);
		cudaMalloc((void **)&d_G, size);
		cudaMemcpy(d_G, flat_G, size, cudaMemcpyHostToDevice);
		cudaMalloc((void **)&d_B, size);
		cudaMemcpy(d_B, flat_B, size, cudaMemcpyHostToDevice);
		int numBlocksY = ceil(colsize/16.0);
		int numBlocksX = ceil(rowsize/16.0);
		dim3 dimBlock(numBlocksY,numBlocksX);
		dim3 dimGrid(16, 16);


    // Punch it chewie
		blur<<<dimGrid, dimBlock>>>(d_R, d_G, d_B);


		int *h_R, *h_G, *h_B;
		h_R = (int *)malloc(size);
		h_G = (int *)malloc(size);
		h_B = (int *)malloc(size);
		cudaMemcpy(h_R, d_R, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_G, d_G, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
		cudaFree(d_R);
		cudaFree(d_G);
		cudaFree(d_B);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		for (int row=0;row<rowsize;row++){
			for (int col=0;col<colsize;col++){
				R[row][col] = h_R[colsize*row+col];
				G[row][col] = h_G[colsize*row+col];
				B[row][col] = h_B[colsize*row+col];
			}
		}

		// for(row=0;row<rowsize;row++){
		// 	for (col=0;col<colsize;col++){
		// 	    R[row][col] = Rnew[row][col];
		// 	    G[row][col] = Gnew[row][col];
		// 	    B[row][col] = Bnew[row][col];
		// 	}
		// }
	}

	gettimeofday(&tim, NULL);
	double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed\n", t2-t1);

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
    return 0;
}
