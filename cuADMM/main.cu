#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverSp.h>
#include <cusolverDn.h>
#include <gurobi_c.h>
#include "util.h"
#include "csr.h"
#include "csrd.h"
#include "densed.h"
#include "Timer.h"

#define h2d cudaMemcpyHostToDevice
#define d2h cudaMemcpyDeviceToHost
#define d2d cudaMemcpyDeviceToDevice

__global__ void bound(double* in, int len, bool pos) {
	int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
		double v = in[i];
		if (pos && v < 0) v = 0;
		if (!pos && v > 0) v = 0;

		in[i] = v;
    }
}

__global__ void fill(double* out, int len, double v) {
	int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
		out[i] = v;
    }
}

double* random_sparse_b(int n, double sparsity=0.3) {
	srand(time(NULL));
	int k = get_k(n);
	double* b = (double*)malloc(sizeof(double) * k);
	for (int i = 0; i < k; i++) {
		double v = 0;
		double k = (double)rand()/RAND_MAX;
		if (k < sparsity) {
			v = (double)r(-5, 5);
		}
		b[i] = v;
	}
	return b;
}

extern "C" {
	void get_random_b(int n, double* b) {
		double* rb = random_sparse_b(n);
		memcpy(b, rb, sizeof(double) * get_k(n));
	}
	void getElements(int input, int* element_count, int* elements) {
		int n = ceil(log(input)/log(2));
		int count = 0;
		for (int i = 0; i < n + 1; i++) {
			int p = (int)pow(2, i);
			if ((p & input) == p) {
				elements[count] = p;
				count++;
			}
		}
		element_count[0] = count;
	}
	void getSubsets(int input, int* subset_count, int* subsets) {
		int count = 0;
		for (int i = 1; i < input + 1; i++) {
			if ((i & input) == i) {
				subsets[count] = i;
				count++;
			}
		}
		subset_count[0] = count;
	}
	void buildCommonAry(int n, int* klst, int* common_count_p, int* common) {
		int k = get_k(n);
		int common_count = 0;
		for (int idx = k; idx > 0; idx--) {
			int* ss = (int*)malloc(sizeof(int) * k);
			int ss_count;
			getSubsets(idx, &ss_count, ss);

			int target = klst[ss[0] - 1];
			if (target == 0) {
				continue;
			}

			int shouldContinue = 0;
			for (int i = 0; i < ss_count; i++) {
				if (klst[ss[i] - 1] != target) {
					shouldContinue = 1;
					break;
				}
			}
			if (shouldContinue) continue;
			
			common[common_count] = idx;
			common_count++;
			for (int i = 0; i < ss_count; i++) {
				klst[ss[i] - 1] = 0;
			}
		}
		common_count_p[0] = common_count;
	}
	void Arow(int n, int rowNum, double* outRow) {
		SM A = make_A(n);

		for (IT it(A, rowNum); it; ++it) {
			outRow[it.col()] = it.value();
		}
	}

	// 0: normal
	// 1: admm time up
	// 2: crossover time up
	int admm(int n, int l, double* ba, double* E, 
			double* outObj=NULL, double* outLmb=NULL,
			int crossover=1,
			double maxTime=1024,
			double eps=1e-8,
			int threads=0,
			const char* logfile=NULL) {

		if (logfile)
			freopen(logfile, "a+", stdout);

		Timer timer;

		double rho = 2;

		int m = get_m(n);
		int k = get_k(n);

		printf("n: %d\n", n);
		printf("m: %d, k: %d, l: %d\n", m, k, l);
		printf("setting up problem\n");

		int status;
		cublasHandle_t handle;
		cusparseHandle_t s_handle;
		cusolverSpHandle_t solver_handle;
		cusolverDnHandle_t dense_handle;

		status = cublasCreate(&handle);
		if (status) {
			printf("failed to initialize cublas. code: %d\n", status);
			exit(1);
		}
		status = cusparseCreate(&s_handle);
		if (status) {
			printf("failed to initialize cusparse. code: %d\n", status);
			exit(1);
		}
		status = cusolverSpCreate(&solver_handle);
		if (status) {
			printf("failed to initialize cusolver sparse. code: %d\n", status);
			exit(1);
		}
		status = cusolverDnCreate(&dense_handle);
		if (status) {
			printf("failed to initialize cusolver dense. code: %d\n", status);
			exit(1);
		}

		CSR* Bh;
		{
			SM b = make_B(n);
			Bh = csr_from_sm(b);
		}
		CSRD* B = csrd_from_csr(Bh);

		double one = 1;
		double zero = 0;
		double none = -1;

		/*cusparseSetPointerMode(s_handle, CUSPARSE_POINTER_MODE_HOST);*/

		cusparseMatDescr_t descr;
		cusparseCreateMatDescr(&descr);
		cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

		int* BTB_rows;
		int BTB_nz;
		cudaMalloc(&BTB_rows, sizeof(int) * (k + 1));
		status = cusparseXcsrgemmNnz(s_handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
				B->n, B->n, B->m,
				descr, B->nz, B->rows, B->cols,
				descr, B->nz, B->rows, B->cols,
				descr, BTB_rows, &BTB_nz);

		CSRD* BTB = csrd_new(k, k, BTB_nz);
		cudaMemcpy(BTB->rows, BTB_rows, sizeof(int) * (k + 1), d2d);
		status = cusparseDcsrgemm(s_handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
				B->n, B->n, B->m,
				descr, B->nz, B->vals, B->rows, B->cols,
				descr, B->nz, B->vals, B->rows, B->cols,
				descr, BTB->vals, BTB->rows, BTB->cols);

		if (status) {
			printf("failed to compute BTB. code: %d\n", status);
			exit(1);
		}

		SM Em;
		if (E) {
			Em = SM(Eigen::Map<M>(E, l, k).sparseView());
			CSR* Ecsr = csr_from_sm(Em);
			CSR* BEh = csr_vstack(Bh, Ecsr);
			csr_free(Bh);
			Bh = BEh;
			csrd_free(B);
			B = csrd_from_csr(Bh);
		}

		// inverse
		timer.start("inverse");


		// now BTB_dense contains the factorization
		// TODO: save and load factorization

		int err;

		double* BTB_dense;
		cudaMalloc(&BTB_dense, sizeof(double) * k * k);
		int* info;
		cudaMalloc(&info, sizeof(int));

		char fn[100];
		char facdir[100];

		get_fac_dir(facdir);
		sprintf(fn, "%s/%d.data", facdir, n);

		printf("searching for factorization file at %s\n", fn);

		if (access(fn, F_OK) != -1) {
			printf("found factorization data. loading...\n");
			double* hBTB = (double*)malloc(sizeof(double) * k * k);
			FILE* fp = fopen(fn, "rb");
			fread(hBTB, sizeof(double), k * k, fp);
			fclose(fp);
			
			cudaMemcpy(BTB_dense, hBTB, sizeof(double) * k * k, h2d);
			free(hBTB);
		}
		else {
			// convert computed BTB to dense
			cusparseDcsr2dense(s_handle, k, k, descr, BTB->vals, BTB->rows, BTB->cols, BTB_dense, k);

			int workspace_size;
			err = cusolverDnDpotrf_bufferSize(dense_handle, CUBLAS_FILL_MODE_LOWER, k, BTB_dense, k, &workspace_size);

			double* workspace;
			cudaMalloc(&workspace, sizeof(double) * workspace_size);

			printf("starting Cholesky factorization\n");
			err = cusolverDnDpotrf(dense_handle, CUBLAS_FILL_MODE_LOWER, k, BTB_dense, k, workspace, workspace_size, info);
			if (err) {
				printf("factorization failed. code: %d\n", err);
				exit(1);
			}

			printf("saving factorization to file\n");
			double* fac = (double*)malloc(sizeof(double) * k * k);
			cudaMemcpy(fac, BTB_dense, sizeof(double) * k * k, d2h);
			FILE* fp = fopen(fn, "wb+");
			fwrite(fac, sizeof(double), k * k, fp);
			fclose(fp);
		}


		double* BTBi;
		if (E) {
			printf("found E. inversing BTB using factorization\n");
			M identity = M::Identity(k, k);
			double* I;
			cudaMalloc(&I, sizeof(double) * k * k);
			cudaMemcpy(I, identity.data(), sizeof(double) * k * k, h2d);
			err = cusolverDnDpotrs(dense_handle, CUBLAS_FILL_MODE_LOWER, k, k, BTB_dense, k, I, k, info);

			if (err) {
				printf("inversion failed. code: %d\n", err);
				exit(1);
			}

			BTBi = I;

			printf("Adding user-constraints to BTBi\n");
			Vd* out = vd_new(k);
			Vd* e = vd_new(k);
			Vd* A = vd_new(k * k);
			Vd* C = vd_new(k * k);
			for (int i = 0; i < l; i++) {
				printf("%d/%d\n", i+1, l);
				V ev = Em.row(i);
				cudaMemcpy(e->vals, ev.data(), sizeof(double) * k, h2d);
				// out = BTBi @ e
				cublasDsymv(handle, CUBLAS_FILL_MODE_LOWER, k, &one, BTBi, k, e->vals, 1, &zero, out->vals, 1);
				// A = out @ out.T
				cudaMemset(A->vals, 0, sizeof(double) * k * k);
				cublasDger(handle, k, k, &one, out->vals, 1, out->vals, 1, A->vals, k);
				// res = out.T @ e
				double res;
				cublasDdot(handle, k, out->vals, 1, e->vals, 1, &res);
				// BTBi -= A / (res + 1)
				double alpha = (double)-1/(res + 1);
				cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, k, &alpha, A->vals, k, &one, BTBi, k, C->vals, k);
				cudaMemcpy(BTBi, C->vals, sizeof(double) * k * k, d2d);
				
			}
			vd_free(out);
			vd_free(e);
			vd_free(A);
			vd_free(C);
		}
		else {
			printf("No user specified constraint. No need to inverse\n");
		}
		timer.end("inverse");

		int csize = 2 * (m + k) + l;
		Vec* bh = vec_new_from_ary(k, ba);
		Vec* ch;
		{
			Vec* v1 = vec_zeros(m + k);
			Vec* v2 = vec_ones(m + k);
			Vec* v3 = vec_zeros(l);

			Vec* mid = vec_vstack(v1, v2);
			ch = vec_vstack(mid, v3);

			vec_free(v1);
			vec_free(v2);
			vec_free(v3);
			vec_free(mid);
		}
		Vec* xh = vec_zeros(k);
		Vec* yh = vec_copy(ch);
		Vec* lmbh = vec_zeros(csize);

		Vd* b = vd_from_vec(bh);
		Vd* c = vd_from_vec(ch);
		Vd* x = vd_from_vec(xh);
		Vd* y = vd_from_vec(yh);
		Vd* lmb = vd_from_vec(lmbh);

		vec_free(bh);
		vec_free(ch);
		vec_free(xh);
		vec_free(yh);
		vec_free(lmbh);

		Vd* c1 = vd_new(csize);
		Vd* c2 = vd_new(csize);
		Vd* k1 = vd_new(k);
		Vd* k2 = vd_new(k);

		double* oy;
		cudaMalloc(&oy, sizeof(double) * csize);

		int it = 0;
		double s, r;

		int timeReached = 0;

		double xt = 0;
		double yt = 0;
		double lt = 0;
		double srt = 0;

		printf("ADMM main loop starting\n");
		timer.start("admm");
		while (true) {
			it ++;

			timer.start("x");
			// ==========
			// x-update
			// ==========

			// V xinner = lmb + rho * y - rho * c;
			// V xr = b + B.transpose() * xinner;
			// x = ((double)-1/rho) * solveSystem(L, xr);

			{
				double nrho = -rho;
				// c1 = lmb
				cudaMemcpy(c1->vals, lmb->vals, sizeof(double) * csize, d2d);
				// c1 = rho * y + c1
				cublasDaxpy(handle, csize, &rho, y->vals, 1, c1->vals, 1);
				// c1 = - rho * c + c1
				cublasDaxpy(handle, csize, &nrho, c->vals, 1, c1->vals, 1);
				// k1 = b
				cudaMemcpy(k1->vals, b->vals, sizeof(double) * k, d2d);
				// k1 = B.T * c1 + k1
				cusparseDcsrmv(s_handle, CUSPARSE_OPERATION_TRANSPOSE, csize, k, B->nz, &one, descr, B->vals, B->rows, B->cols, c1->vals, &one, k1->vals);

				if (E && BTBi) {
					// k2 = BTBi * k1
					cublasDsymv(handle, CUBLAS_FILL_MODE_LOWER, k, &one, BTBi, k, k1->vals, 1, &zero, k2->vals, 1);
				}
				else {
					// k2 = solve(BTB, k1)
					vd_memcpy(k2, k1);
					err = cusolverDnDpotrs(dense_handle, CUBLAS_FILL_MODE_LOWER, k, 1, BTB_dense, k, k2->vals, k, info);
					if (err) {
						printf("x-update system solving failed. code: %d\n", err);
						exit(1);
					}
				}

				// k1 = 0
				cudaMemset(k1->vals, 0, sizeof(double) * k);
				// k1 = -1/rho * k2 + k1
				double alpha = -1/rho;
				cublasDaxpy(handle, k, &alpha, k2->vals, 1, k1->vals, 1);
				// x = k1
				cudaMemcpy(x->vals, k1->vals, sizeof(double) * k, d2d);
			}
			xt += timer.end("x", false);

			timer.start("y");
			// ==========
			// y-update
			// ==========

			//V oy = y;
			//V yinner = c - B * x - ((double)1/rho) * lmb;
			//V ya = yinner.head(m + k).cwiseMin(0);
			//V yb = yinner.tail(m + k + l).head(m + k).cwiseMax(0);
			//V yy(csize);
			//yy << ya, yb, V::Zero(l);
			//y = yy;

			{
				// oy = y
				cudaMemcpy(oy, y->vals, sizeof(double) * csize, d2d);
				// c1 = c
				cudaMemcpy(c1->vals, c->vals, sizeof(double) * csize, d2d);
				// c1 = -B * x + c1
				cusparseDcsrmv(s_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, csize, k, B->nz, &none, descr, B->vals, B->rows, B->cols, x->vals, &one, c1->vals);
				// c1 = -1/rho * lmb + c1
				double alpha = -1/rho;
				cublasDaxpy(handle, csize, &alpha, lmb->vals, 1, c1->vals, 1);

				/* Compute execution configuration */
				dim3 dimBlock(256);
				int threadBlocks = (m + k + (dimBlock.x - 1)) / dimBlock.x;
				if (threadBlocks > 65520) threadBlocks = 65520;
				dim3 dimGrid(threadBlocks);

				bound<<<dimGrid,dimBlock>>>(c1->vals, m + k, false);
				bound<<<dimGrid,dimBlock>>>(c1->vals + m + k, m + k, true);
				cudaMemset(c1->vals + 2 * (m + k), 0, sizeof(double) * l);

				cudaMemcpy(y->vals, c1->vals, sizeof(double) * csize, d2d);
			}
			yt += timer.end("y", false);

			timer.start("lmb");
			// ==========
			// lmb-update
			// ==========

			//V ll = lmb + rho * (B * x + y - c);
			//lmb = ll;

			{
				// c1 = y
				cudaMemcpy(c1->vals, y->vals, sizeof(double) * csize, d2d);
				// c1 = B * x + c1
				cusparseDcsrmv(s_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, csize, k, B->nz, &one, descr, B->vals, B->rows, B->cols, x->vals, &one, c1->vals);
				// c1 = -c + c1
				cublasDaxpy(handle, csize, &none, c->vals, 1, c1->vals, 1);
				// c2 = lmb
				cudaMemcpy(c2->vals, lmb->vals, sizeof(double) * csize, d2d);
				// c2 = rho * c1 + c2
				cublasDaxpy(handle, csize, &rho, c1->vals, 1, c2->vals, 1);
				
				cudaMemcpy(lmb->vals, c2->vals, sizeof(double) * csize, d2d);
			}
			lt += timer.end("lmb", false);

			timer.start("sr");
			// ==========
			// s & r update
			// ==========

			//s = (rho * B.transpose() * (oy - y)).norm();
			//r = (B * x + y - c).norm();

			{
				// oy = -y + oy
				cublasDaxpy(handle, csize, &none, y->vals, 1, oy, 1);
				// k1 = rho * B.T * oy + 0 * k1
				cusparseDcsrmv(s_handle, CUSPARSE_OPERATION_TRANSPOSE, csize, k, B->nz, &rho, descr, B->vals, B->rows, B->cols, oy, &zero, k1->vals);
				// s = norm(k1)
				cublasDnrm2(handle, k, k1->vals, 1, &s);
				// oy = y
				cudaMemcpy(oy, y->vals, sizeof(double) * csize, d2d);
				// oy = B * x + oy
				cusparseDcsrmv(s_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, csize, k, B->nz, &one, descr, B->vals, B->rows, B->cols, x->vals, &one, oy);
				// oy = -c + oy
				cublasDaxpy(handle, csize, &none, c->vals, 1, oy, 1);
				// r = norm(oy)
				cublasDnrm2(handle, csize, oy, 1, &r);
			}
			srt += timer.end("sr", false);

			int print_step = 100;
			if (it % print_step == 0) {
				printf("iteration: %d\n", it);
				printf("r: %g, s: %g\n", r, s);
				printf("time used:\n");
				printf("\tx-update: %g\n", xt/print_step);
				printf("\ty-update: %g\n", yt/print_step);
				printf("\tlmb-update: %g\n", lt/print_step);
				printf("\ts&r-update: %g\n", srt/print_step);
				xt = 0;
				yt = 0;
				lt = 0;
				srt = 0;

				double obj;
				cublasDdot(handle, k, b->vals, 1, x->vals, 1, &obj);
				printf("obj: %g\n", obj);
				printf("==========\n");

				if (maxTime && timer.duration("admm") > maxTime) {
					printf("maximum admm time %g reached. terminating\n", maxTime);
					timeReached = 1;
					break;
				}
			}

			if (r < eps && s < eps) {
				break;
			}
		}

		cudaFree(c1);
		cudaFree(c2);
		cudaFree(k1);
		cudaFree(k2);
		cudaFree(oy);

		csr_free(Bh);

		csrd_free(B);
		csrd_free(BTB);

		printf("ADMM finished\n");
		double tm = timer.end("admm", false);

		printf("\nresult:\n");
		printf("iteration: %d\n", it);
		printf("r: %g, s: %g\n", r, s);
		double obj;
		cublasDdot(handle, k, b->vals, 1, x->vals, 1, &obj);
		printf("obj: %g\n", obj);
		printf("ADMM time: %g\n", tm);

		cudaFree(BTB_dense);

		if (outObj && !crossover) {
			*outObj = obj;

			cublasDscal(handle, m + k, &none, lmb->vals, 1);
			lmbh = vd_to_vec(lmb);
			memcpy(outLmb, lmbh->vals, sizeof(double) * csize);
			vec_free(lmbh);
		}

		if (crossover && !timeReached) {
			GRBenv *env;
			GRBmodel *model;
			GRBloadenv(&env, "crossover.log");

			printf("setting up problem\n");
			double *lb = (double*)malloc(sizeof(double) * k);
			for (int i = 0; i < k; i++) {
				lb[i] = -GRB_INFINITY;
			}
			GRBnewmodel(env, &model, "crossover", k, ba, lb, NULL, NULL, NULL);

			SM A = make_A(n);

			printf("adding constraints\n");
			for (int i = 0; i < m + k; i++) {
				int size = A.row(i).nonZeros();
				double *vals = (double*)malloc(sizeof(double) * size);
				int* idx = (int*)malloc(sizeof(int) * size);
				int j = 0;
				for (IT it(A, i); it; ++it) {
					vals[j] = it.value();
					idx[j] = it.col();
					j++;
				}
				GRBaddconstr(model, size, idx, vals, GRB_GREATER_EQUAL, 0, NULL);

				free(vals);
				free(idx);
			}
			for (int i = 0; i < m + k; i++) {
				int size = A.row(i).nonZeros();
				double *vals = (double*)malloc(sizeof(double) * size);
				int* idx = (int*)malloc(sizeof(int) * size);
				int j = 0;
				for (IT it(A, i); it; ++it) {
					vals[j] = it.value();
					idx[j] = it.col();
					j++;
				}
				GRBaddconstr(model, size, idx, vals, GRB_LESS_EQUAL, 1, NULL);

				free(vals);
				free(idx);
			}
			if (E) {
				for (int i = 0; i < l; i++) {
					int size = Em.row(i).nonZeros();
					double *vals = (double*)malloc(sizeof(double) * size);
					int* idx = (int*)malloc(sizeof(int) * size);
					int j = 0;
					for (IT it(Em, i); it; ++it) {
						vals[j] = it.value();
						idx[j] = it.col();
						j++;
					}
					GRBaddconstr(model, size, idx, vals, GRB_EQUAL, 0, NULL);

					free(vals);
					free(idx);
				}
			}

			GRBenv *modelEnv = GRBgetenv(model);
			GRBsetintparam(modelEnv, GRB_INT_PAR_METHOD, 1);
			GRBsetintparam(modelEnv, GRB_INT_PAR_UPDATEMODE, 0);
			GRBsetdblparam(modelEnv, GRB_DBL_PAR_TIMELIMIT, maxTime - timer.duration("admm"));

			GRBupdatemodel(model);

			xh = vd_to_vec(x);
			printf("setting pstart\n");
			for (int i = 0; i < k; i++) {
				GRBsetdblattrelement(model, GRB_DBL_ATTR_PSTART, i, xh->vals[i]);
			}
			vec_free(xh);

			printf("setting dstart\n");
			cublasDscal(handle, 2 * (m + k), &none, lmb->vals, 1);
			lmbh = vd_to_vec(lmb);
			for (int i = 0; i < csize; i++) {
				GRBsetdblattrelement(model, GRB_DBL_ATTR_DSTART, i, lmbh->vals[i]);
			}
			vec_free(lmbh);

			printf("optimizing\n");
			int err = GRBoptimize(model);
			if (err) {
				printf("%s\n", GRBgeterrormsg(env));
				exit(1);
			}
			// check status to see if time out
			int status;
			GRBgetintattr(model, GRB_INT_ATTR_STATUS, &status);
			if (status == GRB_TIME_LIMIT) {
				timeReached = 2;
			}

			if (outObj) {
				for (int i = 0; i < csize; i++) {
					double v;
					GRBgetdblattrelement(model, GRB_DBL_ATTR_PI, i, &v);
					if (i >= 2*(m + k)) {
						v *= -1;
					}
					outLmb[i] = v;
				}
				double v;
				GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &v);
				*outObj = v;
			}
		}

		if (logfile)
			freopen ("/dev/tty", "a", stdout);
		
		return timeReached;
	}
}

int main(int argc, char** argv) {
	if (argc > 1) {
		int n = atoi(argv[1]);
		double* b = random_sparse_b(n, 0.15);
		return admm(n, 0, b, NULL, NULL, NULL, 0, 1024, 1e-5);
	}
	int n = 2;
	double b[3] = {-1, -2, 2};
	double e[6] = {1, 1, 1, 0, -1, 0};
	return admm(n, 0, b, NULL, NULL, NULL, 0, 1024, 1e-8);
	/*return admm(n, 2, b, e, NULL, NULL, 0, 1024, 1e-8);*/
}
