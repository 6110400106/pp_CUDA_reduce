#include<stdio.h>
#include<math.h>

#define N 2048

//Interleave addressing kernel version
__global__ void interleaved_reduce(int* d_in, int* d_out) {
	int i = threadIdx.x;
	//int M = N/2;
	__shared__ int sB[N];		//shared-block memory
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("d_in[%d]: %d\n", id, d_in[id]);
	sB[i] = d_in[id];
	__syncthreads();
	// slight consideration in using
       	// s = s<<1 instead of s = s*2
	// for aesthetic purposes
	for(int s = 1; s < blockDim.x; s = s*2) {
		/*if(i < M) {
			printf("stride: %d and thread %d is active \n", s, i);
			//d_in[(2*s)*i] = d_in[(2*s)*i] + d_in[(2*s)*i+s];	
		}*/
		int index = 2 * s * id;
		if(index < blockDim.x) {
			//printf("stride: %d and thread %d is active \n", s, i);
			sB[index] += sB[index+s];
		}
		__syncthreads();
		//M = M/2;
	}
	if(i == 0)
		//d_out[0] = d_in[0];
		d_out[blockIdx.x] = sB[0];
}

//Contiguous addressing kernel version
__global__ void contiguous_reduce(int* d_in, int* d_out) {
	/*
	//What teacher taught me
	int i = threadIdx.x;
        int M = N/2;
        for(int s = M; s > 0; s=s>>1) {	// s=>>1 means right shift one bit
					// or means s = s/2
                if(i < M) {
                        printf("stride: %d and thread %d is active \n", s, i);
                        d_in[i] = d_in[i] + d_in[i+s];
                }
                M = M/2;
        }
        if(i == 0)
                d_out[0] = d_in[0];
	*/

	// What I implemented myself
	// parallel sum by using per-block shared memory
	int i = threadIdx.x;
	int id =  blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int sB[N];           //shared-block memory
	sB[i] = d_in[id];
        __syncthreads();
	//s=>>1 means right shift one bit
        // or means s = s/2
        for(int s = blockDim.x/2; s > 0; s=s>>1) {
                if(i < s) {
                        //printf("stride: %d and thread %d is active \n", s, i);
                        sB[i] += sB[i+s];
                }
            	__syncthreads();
        }
        if(i == 0)
                d_out[blockIdx.x] = sB[0];

}


int main() {
	int h_in[N];
	int h_out;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for(int i = 0; i < N; i++) {
		h_in[i] = i+1;	
	}
	
	int *d_in, *d_out;

	//Part 1: Memory transfer from host to device
	cudaMalloc((void**) &d_in, N*sizeof(int));
	cudaMalloc((void**) &d_out, sizeof(int));

	cudaMemcpy(d_in, &h_in, N*sizeof(int), cudaMemcpyHostToDevice);

	//Part 2: Execute kernel

	//Timed interleaved_reduce function
	/*cudaEventRecord(start);
	interleaved_reduce<<<1, 1024>>>(d_in, d_out);
	cudaEventRecord(stop);*/
	
	//Timed contiguos_reduce function
	cudaEventRecord(start);
	contiguous_reduce<<<1, 1024>>>(d_in, d_out);
	cudaEventRecord(stop);

	//Part 3: Memory transfer from device to host
	cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaFree(d_in);
	cudaFree(d_out);

	printf("Output: %d\n", h_out);
	printf("Time used: %f milliseconds\n", milliseconds);

	return -1;
}
