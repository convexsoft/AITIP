#ifndef CSRD_H
#define CSRD_H

#include <cuda_runtime.h>

#include "csr.h"

typedef CSR CSRD;

CSRD* csrd_new(int m, int n, int nz) {
	CSRD* csr = (CSRD*)malloc(sizeof(CSRD));

	csr->m = m;
	csr->n = n;
	csr->nz = nz;

	cudaError_t err;

	err = cudaMalloc(&csr->vals, sizeof(double) * csr->nz);
	err = cudaMalloc(&csr->cols, sizeof(int) * csr->nz);
	err = cudaMalloc(&csr->rows, sizeof(int) * (csr->m + 1));

	if (err) {
		printf("csrd_new() failed to allocate device memory\n");
		printf("msg: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	return csr;
}

void csrd_free(CSRD* csr) {
	cudaFree(csr->vals);
	cudaFree(csr->cols);
	cudaFree(csr->rows);
	free(csr);
}

CSRD* csrd_from_csr(CSR* csr) {
	CSRD* d = csrd_new(csr->m, csr->n, csr->nz);

	cudaError_t err;

	err = cudaMemcpy(d->vals, csr->vals, sizeof(double) * d->nz,
	                 cudaMemcpyHostToDevice);
	err = cudaMemcpy(d->cols, csr->cols, sizeof(int) * d->nz,
	                 cudaMemcpyHostToDevice);
	err = cudaMemcpy(d->rows, csr->rows, sizeof(int) * (d->m + 1),
	                 cudaMemcpyHostToDevice);

	if (err) {
		printf("csrd_from_csr() failed to copy to device\n");
		printf("msg: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	return d;
}

CSR* csrd_to_csr(CSRD* d) {
	CSR* csr = csr_new(d->m, d->n, d->nz);

	cudaError_t err;

	err = cudaMemcpy(csr->vals, d->vals, sizeof(double) * d->nz,
	                 cudaMemcpyDeviceToHost);
	err = cudaMemcpy(csr->cols, d->cols, sizeof(int) * d->nz,
	                 cudaMemcpyDeviceToHost);
	err = cudaMemcpy(csr->rows, d->rows, sizeof(int) * (d->m + 1),
	                 cudaMemcpyDeviceToHost);

	if (err) {
		printf("csrd_to_csr() failed to copy to host\n");
		printf("msg: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	return csr;
}

void csrd_print(CSRD* d) {
	CSR* csr = csrd_to_csr(d);
	csr_print(csr);
	csr_free(csr);
}

#endif
