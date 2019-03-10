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
	//��device�Ϸ����ڴ�
	cudaMalloc((void **)&deva, N * sizeof(int));
	cudaMalloc((void **)&devb, N * sizeof(int));
	cudaMalloc((void **)&devc, N * sizeof(int));

	//��host��Ϊ���鸳ֵ
	for (int i = 0; i < N; ++i)
	{
		a[i] = -i;
		b[i] = i * i;
	}

	//�������ֵ������device��
	cudaMemcpy(deva, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devb, b, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devc, c, N * sizeof(int), cudaMemcpyHostToDevice);

	//����Kernel����
	add <<<N,1 >> >(deva, devb, devc);

	//�������device����host
	cudaMemcpy(c, devc, N * sizeof(int), cudaMemcpyDeviceToHost);

	//������
	for (size_t i = 0; i < N; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	//�ͷ�device������ڴ�
	cudaFree(deva);
	cudaFree(devb);
	cudaFree(devc);

	return 0;
}