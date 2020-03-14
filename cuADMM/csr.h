#ifndef CSR_H
#define CSR_H

#include <stdlib.h>

#include "util.h"

typedef struct {
	int m;
	int n;
	int nz;
	double* vals;
	int* cols;
	int* rows;
} CSR;

void csr_print(CSR* csr) {
	printf("m: %d\n", csr->m);
	printf("n: %d\n", csr->n);
	printf("nz: %d\n", csr->nz);
	ppA(csr->vals, csr->nz, "vals");
	ppA<int>(csr->cols, csr->nz, "cols");
	ppA<int>(csr->rows, csr->m + 1, "rows");
}

CSR* csr_new(int m, int n, int nz) {
	CSR* csr = (CSR*)malloc(sizeof(CSR));

	csr->m = m;
	csr->n = n;
	csr->nz = nz;
	csr->vals = (double*)malloc(sizeof(double) * csr->nz);
	csr->cols = (int*)malloc(sizeof(int) * csr->nz);
	csr->rows = (int*)malloc(sizeof(int) * (csr->m + 1));

	return csr;
}

CSR* csr_from_sm(SM sm) {
	sm.makeCompressed();

	CSR* csr = csr_new(sm.rows(), sm.cols(), sm.nonZeros());

	memcpy(csr->vals, sm.valuePtr(), sizeof(double) * csr->nz);
	memcpy(csr->cols, sm.innerIndexPtr(), sizeof(int) * csr->nz);
	memcpy(csr->rows, sm.outerIndexPtr(), sizeof(int) * (csr->m + 1));

	return csr;
}

CSR* csr_identity(int n) {
	CSR* out = csr_new(n, n, n);
	for (int i = 0; i < n; i++) {
		out->vals[i] = 1;
		out->cols[i] = i;
		out->rows[i] = i;
	}
	out->rows[n] = n;
	return out;
}

CSR* csr_copy(CSR* csr) {
	CSR* n = csr_new(csr->m, csr->n, csr->nz);

	memcpy(n->vals, csr->vals, sizeof(double) * n->nz);
	memcpy(n->cols, csr->cols, sizeof(int) * n->nz);
	memcpy(n->rows, csr->rows, sizeof(int) * (n->m + 1));

	return n;
}

void csr_reorder(CSR* csr) {
	// the returned cols array from some mkl routines do not guarantee ordering.
	// 		i.e., the cols indices in the same row are not in the increasing
	// order. 		in order to input to pardiso, we need to reorder the cols and vals
	// array.

	int row_b = 0;
	int row_e = 0;
	int row_nz = 0;

	double* new_vals = (double*)malloc(sizeof(double) * csr->nz);
	int* new_cols = (int*)malloc(sizeof(int) * csr->nz);

	for (int i = 0; i < csr->m; i++) {
		row_b = csr->rows[i];
		row_e = csr->rows[i + 1];
		row_nz = row_e - row_b;

		int perm[row_nz];
		for (int j = 0; j < row_nz; j++) {
			perm[j] = row_b + j;
		}
		std::vector<int> permv(perm, perm + row_nz);
		std::sort(permv.begin(), permv.end(), [&csr](const int a, const int b) {
			int av = csr->cols[a];
			int bv = csr->cols[b];
			return av < bv;
		});

		for (int j = 0; j < row_nz; j++) {
			new_vals[row_b + j] = csr->vals[permv[j]];
			new_cols[row_b + j] = csr->cols[permv[j]];
		}
	}

	memcpy(csr->vals, new_vals, sizeof(double) * csr->nz);
	memcpy(csr->cols, new_cols, sizeof(int) * csr->nz);

	free(new_vals);
	free(new_cols);
}

CSR* csr_vstack(CSR* m1, CSR* m2) {
	assert(m1->n == m2->n);
	CSR* out = csr_new(m1->m + m2->m, m1->n, m1->nz + m2->nz);

	memcpy(out->vals, m1->vals, sizeof(double) * m1->nz);
	memcpy(out->vals + m1->nz, m2->vals, sizeof(double) * m2->nz);
	memcpy(out->cols, m1->cols, sizeof(int) * m1->nz);
	memcpy(out->cols + m1->nz, m2->cols, sizeof(int) * m2->nz);
	memcpy(out->rows, m1->rows, sizeof(int) * (m1->m + 1));

	int* rows_cp = (int*)malloc(sizeof(int) * (m2->m));
	int last = m1->rows[m1->m];
	for (int i = 0; i < m2->m; i++) {
		rows_cp[i] = last + m2->rows[i + 1];
	}
	memcpy(out->rows + m1->m + 1, rows_cp, sizeof(int) * m2->m);

	free(rows_cp);
	return out;
}

void csr_free(CSR* csr) {
	free(csr->vals);
	free(csr->cols);
	free(csr->rows);
	free(csr);
}

#endif
