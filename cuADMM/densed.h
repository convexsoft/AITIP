#ifndef DENSED_H
#define DENSED_H

#include <cuda_runtime.h>

#include "dense.h"

typedef Vec Vd;

Vd* vd_new(int size) {
	Vd* v = (Vd*)malloc(sizeof(Vd));
	v->size = size;

	cudaError_t err;
	err = cudaMalloc(&v->vals, sizeof(double) * size);
	if (err) {
		printf("vd_new() failed to allocate device memory\n");
		printf("msg: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	return v;
}

void vd_memcpy(Vd* v1, Vd* v2) {
	assert(v1->size == v2->size);
	cudaError_t err;
	err = cudaMemcpy(v1->vals, v2->vals, sizeof(double) * v1->size,
	                 cudaMemcpyDeviceToDevice);
	if (err) {
		printf("vd_memcpy() failed to copy memory\n");
		printf("msg: %s\n", cudaGetErrorString(err));
		exit(1);
	}
}

void vd_free(Vd* d) {
	cudaFree(d->vals);
	free(d);
}

Vd* vd_from_vec(Vec* v) {
	Vd* d = vd_new(v->size);

	cudaError_t err;
	err = cudaMemcpy(d->vals, v->vals, sizeof(double) * v->size,
	                 cudaMemcpyHostToDevice);
	if (err) {
		printf("vd_from_vec() failed to copy to device\n");
		printf("msg: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	return d;
}

Vec* vd_to_vec(Vd* d) {
	Vec* v = vec_new(d->size);

	cudaError_t err;
	err = cudaMemcpy(v->vals, d->vals, sizeof(double) * d->size,
	                 cudaMemcpyDeviceToHost);
	if (err) {
		printf("vd_to_vec() failed to copy to device\n");
		printf("msg: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	return v;
}

void vd_print(Vd* d) {
	Vec* v = vd_to_vec(d);
	vec_print(v);
	vec_free(v);
}

#endif
