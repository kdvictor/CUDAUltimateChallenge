#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 10

__global__ void add(int *a, int *b, int *c)
{
	int tid = blockIdx.x;
	c[tid] = a[tid] + b[tid];
}

int main()
{
	int a[N], b[N], c[N];
	int *deva, *devb, *devc;
	//在device上分配内存
	cudaMalloc((void **)&deva, N * sizeof(int));
	cudaMalloc((void **)&devb, N * sizeof(int));
	cudaMalloc((void **)&devc, N * sizeof(int));

	//在host端为数组赋值
	for (int i = 0; i < N; ++i)
	{
		a[i] = -i;
		b[i] = i * i;
	}

	//将数组的值拷贝到device端
	cudaMemcpy(deva, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devb, b, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devc, c, N * sizeof(int), cudaMemcpyHostToDevice);

	//调用Kernel函数
	add <<<N,1 >> >(deva, devb, devc);

	//将结果从device拷到host
	cudaMemcpy(c, devc, N * sizeof(int), cudaMemcpyDeviceToHost);

	//输出结果
	for (size_t i = 0; i < N; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	//释放device分配的内存
	cudaFree(deva);
	cudaFree(devb);
	cudaFree(devc);

	return 0;
}