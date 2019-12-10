#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <cuda.h>

#define IMAGE_WIDTH 521
#define IMAGE_HEIGHT 428

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
void blur(int *d_R,int *d_G, int *d_B, int *d_Rnew, int *d_Gnew, int *d_Bnew)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= IMAGE_HEIGHT || y >= IMAGE_WIDTH) {
      // printf("%s\n", "Pixel out of bounds");
      return;
  }
  int myval = d_R[(IMAGE_WIDTH*x) + y];
  // d_R[(IMAGE_WIDTH*x) + y] = d_R[(IMAGE_WIDTH*x) + y] / 2;
  // d_G[(IMAGE_WIDTH*x) + y] = d_G[(IMAGE_WIDTH*x) + y] / 3;
  // d_B[(IMAGE_WIDTH*x) + y] = d_B[(IMAGE_WIDTH*x) + y] / 4;
  // x and y could be the wrong way round
  // \[(\w\+?-?\d?)\]\[(\w\+?-?\d?)\]
  // [(IMAGE_WIDTH*$1) + $2]
  if (x != 0 && x != (IMAGE_WIDTH-1) && y != 0 && y != (IMAGE_HEIGHT-1)){
    d_Rnew[(IMAGE_WIDTH*x) + y] = (d_R[(IMAGE_WIDTH*x+1) + y]+d_R[(IMAGE_WIDTH*x-1) + y]+d_R[(IMAGE_WIDTH*x) + y+1]+d_R[(IMAGE_WIDTH*x) + y-1])/4;
    d_Gnew[(IMAGE_WIDTH*x) + y] = (d_G[(IMAGE_WIDTH*x+1) + y]+d_G[(IMAGE_WIDTH*x-1) + y]+d_G[(IMAGE_WIDTH*x) + y+1]+d_G[(IMAGE_WIDTH*x) + y-1])/4;
    d_Bnew[(IMAGE_WIDTH*x) + y] = (d_B[(IMAGE_WIDTH*x+1) + y]+d_B[(IMAGE_WIDTH*x-1) + y]+d_B[(IMAGE_WIDTH*x) + y+1]+d_B[(IMAGE_WIDTH*x) + y-1])/4;
  }
  else if (x == 0 && y != 0 && y != (IMAGE_HEIGHT-1)){
    d_Rnew[(IMAGE_WIDTH*x) + y] = (d_R[(IMAGE_WIDTH*x+1) + y]+d_R[(IMAGE_WIDTH*x) + y+1]+d_R[(IMAGE_WIDTH*x) + y-1])/3;
    d_Gnew[(IMAGE_WIDTH*x) + y] = (d_G[(IMAGE_WIDTH*x+1) + y]+d_G[(IMAGE_WIDTH*x) + y+1]+d_G[(IMAGE_WIDTH*x) + y-1])/3;
    d_Bnew[(IMAGE_WIDTH*x) + y] = (d_B[(IMAGE_WIDTH*x+1) + y]+d_B[(IMAGE_WIDTH*x) + y+1]+d_B[(IMAGE_WIDTH*x) + y-1])/3;
  }
  else if (x == (IMAGE_WIDTH-1) && y != 0 && y != (IMAGE_HEIGHT-1)){
    d_Rnew[(IMAGE_WIDTH*x) + y] = (d_R[(IMAGE_WIDTH*x-1) + y]+d_R[(IMAGE_WIDTH*x) + y+1]+d_R[(IMAGE_WIDTH*x) + y-1])/3;
    d_Gnew[(IMAGE_WIDTH*x) + y] = (d_G[(IMAGE_WIDTH*x-1) + y]+d_G[(IMAGE_WIDTH*x) + y+1]+d_G[(IMAGE_WIDTH*x) + y-1])/3;
    d_Bnew[(IMAGE_WIDTH*x) + y] = (d_B[(IMAGE_WIDTH*x-1) + y]+d_B[(IMAGE_WIDTH*x) + y+1]+d_B[(IMAGE_WIDTH*x) + y-1])/3;
  }
  else if (y == 0 && x != 0 && x != (IMAGE_WIDTH-1)){
    d_Rnew[(IMAGE_WIDTH*x) + y] = (d_R[(IMAGE_WIDTH*x+1) + y]+d_R[(IMAGE_WIDTH*x-1) + y]+d_R[(IMAGE_WIDTH*x) + y+1])/3;
    d_Gnew[(IMAGE_WIDTH*x) + y] = (d_G[(IMAGE_WIDTH*x+1) + y]+d_G[(IMAGE_WIDTH*x-1) + y]+d_G[(IMAGE_WIDTH*x) + y+1])/3;
    d_Bnew[(IMAGE_WIDTH*x) + y] = (d_B[(IMAGE_WIDTH*x+1) + y]+d_B[(IMAGE_WIDTH*x-1) + y]+d_B[(IMAGE_WIDTH*x) + y+1])/3;
  }
  else if (y == (IMAGE_HEIGHT-1) && x != 0 && x != (IMAGE_WIDTH-1)){
    d_Rnew[(IMAGE_WIDTH*x) + y] = (d_R[(IMAGE_WIDTH*x+1) + y]+d_R[(IMAGE_WIDTH*x-1) + y]+d_R[(IMAGE_WIDTH*x) + y-1])/3;
    d_Gnew[(IMAGE_WIDTH*x) + y] = (d_G[(IMAGE_WIDTH*x+1) + y]+d_G[(IMAGE_WIDTH*x-1) + y]+d_G[(IMAGE_WIDTH*x) + y-1])/3;
    d_Bnew[(IMAGE_WIDTH*x) + y] = (d_B[(IMAGE_WIDTH*x+1) + y]+d_B[(IMAGE_WIDTH*x-1) + y]+d_B[(IMAGE_WIDTH*x) + y-1])/3;
  }
  else if (x==0 &&y==0){
    d_Rnew[(IMAGE_WIDTH*x) + y] = (d_R[(IMAGE_WIDTH*x) + y+1]+d_R[(IMAGE_WIDTH*x+1) + y])/2;
    d_Gnew[(IMAGE_WIDTH*x) + y] = (d_G[(IMAGE_WIDTH*x) + y+1]+d_G[(IMAGE_WIDTH*x+1) + y])/2;
    d_Bnew[(IMAGE_WIDTH*x) + y] = (d_B[(IMAGE_WIDTH*x) + y+1]+d_B[(IMAGE_WIDTH*x+1) + y])/2;
  }
  else if (x==0 &&y==(IMAGE_HEIGHT-1)){
    d_Rnew[(IMAGE_WIDTH*x) + y] = (d_R[(IMAGE_WIDTH*x) + y-1]+d_R[(IMAGE_WIDTH*x+1) + y])/2;
    d_Gnew[(IMAGE_WIDTH*x) + y] = (d_G[(IMAGE_WIDTH*x) + y-1]+d_G[(IMAGE_WIDTH*x+1) + y])/2;
    d_Bnew[(IMAGE_WIDTH*x) + y] = (d_B[(IMAGE_WIDTH*x) + y-1]+d_B[(IMAGE_WIDTH*x+1) + y])/2;
  }
  else if (x==(IMAGE_WIDTH-1) &&y==0){
    d_Rnew[(IMAGE_WIDTH*x) + y] = (d_R[(IMAGE_WIDTH*x) + y+1]+d_R[(IMAGE_WIDTH*x-1) + y])/2;
    d_Gnew[(IMAGE_WIDTH*x) + y] = (d_G[(IMAGE_WIDTH*x) + y+1]+d_G[(IMAGE_WIDTH*x-1) + y])/2;
    d_Bnew[(IMAGE_WIDTH*x) + y] = (d_B[(IMAGE_WIDTH*x) + y+1]+d_B[(IMAGE_WIDTH*x-1) + y])/2;
  }
  else if (x==(IMAGE_WIDTH-1) &&y==(IMAGE_HEIGHT-1)){
    d_Rnew[(IMAGE_WIDTH*x) + y] = (d_R[(IMAGE_WIDTH*x) + y-1]+d_R[(IMAGE_WIDTH*x-1) + y])/2;
    d_Gnew[(IMAGE_WIDTH*x) + y] = (d_G[(IMAGE_WIDTH*x) + y-1]+d_G[(IMAGE_WIDTH*x-1) + y])/2;
    d_Bnew[(IMAGE_WIDTH*x) + y] = (d_B[(IMAGE_WIDTH*x) + y-1]+d_B[(IMAGE_WIDTH*x-1) + y])/2;
  }
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

	nblurs = 20;
	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);

	for(k=0;k<nblurs;k++){
		int flat_R[rowsize * colsize];
		int flat_G[rowsize * colsize];
		int flat_B[rowsize * colsize];
		for (int row=0;row<colsize;row++){
			for (int col=0;col<rowsize;col++){
				flat_R[rowsize*row+col] = R[col][row];
				flat_G[rowsize*row+col] = G[col][row];
				flat_B[rowsize*row+col] = B[col][row];
			}
		}
		int *d_R, *d_G, *d_B, *d_Rnew, *d_Gnew, *d_Bnew;
		int size = sizeof(int) * colsize * rowsize;
		// printf("%d\n", size);
		cudaMalloc((void **)&d_R, size);
		cudaMemcpy(d_R, flat_R, size, cudaMemcpyHostToDevice);
		cudaMalloc((void **)&d_G, size);
		cudaMemcpy(d_G, flat_G, size, cudaMemcpyHostToDevice);
		cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_Rnew, size);
    cudaMalloc((void **)&d_Gnew, size);
		cudaMalloc((void **)&d_Bnew, size);
		cudaMemcpy(d_B, flat_B, size, cudaMemcpyHostToDevice);
		int numBlocksY = ceil(colsize/16.0);
		int numBlocksX = ceil(rowsize/16.0);
		dim3 dimBlock(numBlocksY,numBlocksX);
		dim3 dimGrid(16, 16);

		blur<<<dimGrid, dimBlock>>>(d_R, d_G, d_B, d_Rnew, d_Gnew, d_Bnew);
		int *h_R, *h_G, *h_B;
		h_R = (int *)malloc(size);
    h_G = (int *)malloc(size);
		h_B = (int *)malloc(size);
		cudaMemcpy(h_R, d_Rnew, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_G, d_Gnew, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_B, d_Bnew, size, cudaMemcpyDeviceToHost);
		cudaFree(d_R);
		cudaFree(d_G);
		cudaFree(d_B);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		for (int row=0;row<colsize;row++){
			for (int col=0;col<rowsize;col++){
				R[col][row] = h_R[rowsize*row+col];
				G[col][row] = h_G[rowsize*row+col];
				B[col][row] = h_B[rowsize*row+col];
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
