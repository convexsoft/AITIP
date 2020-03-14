#ifndef DENSE_H
#define DENSE_H

#include <stdarg.h>
#include <stdlib.h>

#include "util.h"

typedef struct {
	int size;
	double* vals;
} Vec;

Vec* vec_new(int size) {
	Vec* v = (Vec*)malloc(sizeof(Vec));
	v->size = size;
	v->vals = (double*)malloc(sizeof(double) * size);
	return v;
}

Vec* vec_new_from_ary(int size, double* ary) {
	Vec* out = vec_new(size);
	memcpy(out->vals, ary, sizeof(double) * size);
	return out;
}

void vec_free(Vec* v) {
	free(v->vals);
	free(v);
}

void vec_print(Vec* v, const char* name = "") { ppA(v->vals, v->size, name); }

Vec* vec_copy(Vec* v) {
	Vec* out = vec_new(v->size);
	memcpy(out->vals, v->vals, sizeof(double) * v->size);
	return out;
}

Vec* vec_vstack(Vec* v1, Vec* v2) {
	Vec* out = vec_new(v1->size + v2->size);
	memcpy(out->vals, v1->vals, sizeof(double) * v1->size);
	memcpy(out->vals + v1->size, v2->vals, sizeof(double) * v2->size);
	return out;
}

void vec_fill_val(Vec* v, double val) { std::fill_n(v->vals, v->size, val); }

Vec* vec_zeros(int size) {
	Vec* out = vec_new(size);
	vec_fill_val(out, 0);
	return out;
}

Vec* vec_ones(int size) {
	Vec* out = vec_new(size);
	vec_fill_val(out, 1);
	return out;
}

#endif
