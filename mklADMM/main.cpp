#include <gurobi_c.h>
#include <mkl.h>
#include <mkl_pardiso.h>
#include <mkl_spblas.h>
#include <mkl_types.h>
#include <unistd.h>

#include "Timer.h"
#include "csr.h"
#include "dense.h"
#include "util.h"

void bound(double* ary, int size, int keep_pos) {
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		double v = ary[i];
		if (keep_pos && v < 0) v = 0;
		if (!keep_pos && v > 0) v = 0;
		ary[i] = v;
	}
}

double* random_sparse_b(int n, double sparsity = 0.3) {
	srand(time(NULL));
	int k = get_k(n);
	double* b = (double*)malloc(sizeof(double) * k);
	for (int i = 0; i < k; i++) {
		double v = 0;
		double k = (double)rand() / RAND_MAX;
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
	int n = ceil(log(input) / log(2));
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
int admm(int n, int l, double* ba, double* E, double* outObj = NULL,
         double* outLmb = NULL, int crossover = 1, double maxTime = 1024,
         double eps = 1e-8, int threads = 0, const char* logfile = NULL) {
	if (logfile) freopen(logfile, "a+", stdout);

	if (threads <= 0) threads = mkl_get_max_threads();
	mkl_set_num_threads(threads);
	omp_set_num_threads(threads);
	printf("Maximum threads set to %d\n", threads);

	Timer timer;

	double rho = 2;

	int m = get_m(n);
	int k = get_k(n);

	printf("n: %d\n", n);
	printf("m: %d, k: %d, l: %d\n", m, k, l);
	printf("setting up problem\n");

	CSR* B;
	{
		SM b = make_B(n);
		B = csr_from_sm(b);
	}

	sparse_status_t status;
	sparse_matrix_t Bh = csr_get_handle(B);
	sparse_matrix_t BTBh = NULL;

	status = mkl_sparse_spmm(SPARSE_OPERATION_TRANSPOSE, Bh, Bh, &BTBh);
	if (status) {
		printf("failed spmm. %d\n", status);
		exit(1);
	}
	mkl_sparse_destroy(Bh);

	SM Em;
	if (E) {
		Em = SM(Eigen::Map<M>(E, l, k).sparseView());
		CSR* Ecsr = csr_from_sm(Em);
		CSR* BE = csr_vstack(B, Ecsr);
		csr_free(B);
		B = csr_copy(BE);
		csr_free(BE);
		csr_free(Ecsr);
	}

	CSR* BTB = csr_from_handle(BTBh);
	mkl_sparse_destroy(BTBh);
	csr_reorder(BTB);

	// convert BTB to dense
	int job[6];
	job[0] = 1;  // csr to dense
	job[1] = 0;  // 0-base
	job[2] = 0;
	job[3] = 0;  // lower triangular
	int info;

	timer.start("invert");
	double* dBTB = (double*)malloc(sizeof(double) * k * k);

	char fn[100];
	char facdir[100];

	get_fac_dir(facdir);
	sprintf(fn, "%s/%d.data", facdir, n);

	printf("searching for factorization file at %s\n", fn);

	if (access(fn, F_OK) != -1) {
		printf("found factorization data. loading...\n");
		FILE* fp = fopen(fn, "rb");
		size_t read_size = fread(dBTB, sizeof(double), k * k, fp);
		fclose(fp);
		if (read_size != k * k) {
			printf("failed to load factorization\n");
			exit(1);
		}
	} else {
		mkl_ddnscsr(job, &k, &k, dBTB, &k, BTB->vals, BTB->cols, BTB->rows,
		            &info);
		if (info) {
			printf("BTB csr to dense failed. info: %d\n", info);
			exit(1);
		}
		printf("starting factorization\n");
		LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', k, dBTB, k);
		printf("saving factorization to file\n");
		FILE* fp = fopen(fn, "wb+");
		fwrite(dBTB, sizeof(double), k * k, fp);
		fclose(fp);
	}

	double* BTBi = NULL;

	if (E) {
		M I = M::Identity(k, k);
		BTBi = (double*)malloc(sizeof(double) * k * k);
		memcpy(BTBi, I.data(), sizeof(double) * k * k);

		LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', k, k, dBTB, k, BTBi, k);

		printf("Adding user-constraints to BTBi\n");
		double* out = (double*)malloc(sizeof(double) * k);
		double* A = (double*)malloc(sizeof(double) * k * k);
		V denom = V::Ones(k);
		for (int i = 0; i < l; i++) {
			printf("%d/%d\n", i + 1, l);

			V e = Em.row(i);
			memset(out, 0, sizeof(double) * k);
			memset(A, 0, sizeof(double) * k * k);
			// out = BTBi @ e
			cblas_dgemv(CblasRowMajor, CblasNoTrans, k, k, 1, BTBi, k, e.data(),
			            1, 0, out, 1);
			// A = out @ out.T
			cblas_dger(CblasRowMajor, k, k, 1, out, 1, out, 1, A, k);
			// res = out.T @ e
			double res = cblas_ddot(k, out, 1, e.data(), 1);

			// BTBi -= A / (res + 1)
			for (int i = 0; i < k * k; i++) {
				BTBi[i] -= A[i] / (res + 1);
			}
		}
		free(out);
		free(A);
	} else {
		printf("No E found. No need to inverse\n");
	}
	timer.end("invert");

	int csize = 2 * (m + k) + l;
	Vec* b = vec_new_from_ary(k, ba);
	Vec* c;
	{
		Vec* v1 = vec_zeros(m + k);
		Vec* v2 = vec_ones(m + k);
		Vec* v3 = vec_zeros(l);

		Vec* mid = vec_vstack(v1, v2);
		c = vec_vstack(mid, v3);

		vec_free(v1);
		vec_free(v2);
		vec_free(v3);
		vec_free(mid);
	}

	Vec* x = vec_zeros(k);
	Vec* y = vec_copy(c);
	Vec* lmb = vec_zeros(csize);

	int it = 0;
	double s, r;

	int timeReached = 0;
	double xt = 0;
	double yt = 0;
	double lt = 0;
	double srt = 0;

	double* c1 = (double*)malloc(sizeof(double) * csize);
	double* c2 = (double*)malloc(sizeof(double) * csize);
	double* k1 = (double*)malloc(sizeof(double) * k);
	double* k2 = (double*)malloc(sizeof(double) * k);

	double* oy = (double*)malloc(sizeof(double) * csize);

	printf("ADMM main loop starting\n");
	timer.start("admm");
	while (true) {
		it++;

		timer.start("x");
		// ==========
		// x-update
		// ==========

		// V xinner = lmb + rho * y - rho * c;
		// V xr = b + B.transpose() * xinner;
		// x = ((double)-1/rho) * solveSystem(L, xr);

		{
			// c1 = lmb
			// c1 = rho * y + c1
			memcpy(c1, lmb->vals, sizeof(double) * csize);
			cblas_daxpy(csize, rho, y->vals, 1, c1, 1);
			// c1 = - rho * c + c1
			cblas_daxpy(csize, -rho, c->vals, 1, c1, 1);
			// k1 = B.T * c1
			struct matrix_descr desc;
			desc.type = SPARSE_MATRIX_TYPE_GENERAL;
			Bh = csr_get_handle(B);
			mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1, Bh, desc, c1, 0, k1);
			mkl_sparse_destroy(Bh);
			// k1 = b + k1
			cblas_daxpy(k, 1, b->vals, 1, k1, 1);

			if (BTBi) {
				// k1 = -1/rho * BTBi * k1 + 0 * k2;
				cblas_dsymv(CblasRowMajor, CblasUpper, k, (double)-1 / rho,
				            BTBi, k, k1, 1, 0, k2, 1);
				// x = k2
				memcpy(x->vals, k2, sizeof(double) * k);
			} else {
				// No E, so just use factorization to solve system
				LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', k, 1, dBTB, k, k1, 1);
				// k2 = 0
				std::fill_n(k2, k, 0);
				// k2 = -1/rho * k1 + k2
				cblas_daxpy(k, (double)-1 / rho, k1, 1, k2, 1);
				// x = k2
				memcpy(x->vals, k2, sizeof(double) * k);
			}
		}
		xt += timer.end("x", false);

		timer.start("y");
		// ==========
		// y-update
		// ==========

		// V oy = y;
		// V yinner = c - B * x - ((double)1/rho) * lmb;
		// V ya = yinner.head(m + k).cwiseMin(0);
		// V yb = yinner.tail(m + k + l).head(m + k).cwiseMax(0);
		// V yy(csize);
		// yy << ya, yb, V::Zero(l);
		// y = yy;

		memcpy(oy, y->vals, sizeof(double) * csize);

		{
			// c1 = lmb
			// c1 = -B * x - 1/rho * c1
			memcpy(c1, lmb->vals, sizeof(double) * csize);
			struct matrix_descr desc;
			desc.type = SPARSE_MATRIX_TYPE_GENERAL;
			Bh = csr_get_handle(B);
			mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1, Bh, desc,
			                x->vals, (double)-1 / rho, c1);
			mkl_sparse_destroy(Bh);
			// c1 = c + c1
			cblas_daxpy(csize, 1, c->vals, 1, c1, 1);

			bound(c1, m + k, 0);
			bound(c1 + m + k, m + k, 1);
			std::fill_n(c1 + 2 * (m + k), l, 0);

			memcpy(y->vals, c1, sizeof(double) * csize);
		}
		yt += timer.end("y", false);

		timer.start("l");
		// ==========
		// lmb-update
		// ==========

		// V ll = lmb + rho * (B * x + y - c);
		// lmb = ll;

		{
			// c1 = y
			// c1 = B * x + c1
			memcpy(c1, y->vals, sizeof(double) * csize);

			struct matrix_descr desc;
			desc.type = SPARSE_MATRIX_TYPE_GENERAL;
			Bh = csr_get_handle(B);
			mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, Bh, desc,
			                x->vals, 1, c1);
			mkl_sparse_destroy(Bh);

			// c1 = -c + c1
			cblas_daxpy(csize, -1, c->vals, 1, c1, 1);

			// c2 = lmb
			// c2 = rho * c1 + c2
			memcpy(c2, lmb->vals, sizeof(double) * csize);
			cblas_daxpy(csize, rho, c1, 1, c2, 1);

			memcpy(lmb->vals, c2, sizeof(double) * csize);
		}
		lt += timer.end("l", false);

		timer.start("sr");
		// ==========
		// s & r update
		// ==========

		// s = (rho * B.transpose() * (oy - y)).norm();
		// r = (B * x + y - c).norm();

		{
			cblas_daxpy(csize, -1, y->vals, 1, oy, 1);

			// k1 = rho * B.T * oy + 0 * k1
			struct matrix_descr desc;
			desc.type = SPARSE_MATRIX_TYPE_GENERAL;
			Bh = csr_get_handle(B);
			mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, rho, Bh, desc, oy, 0,
			                k1);

			// s = norm(k1)
			s = cblas_dnrm2(k, k1, 1);

			// oy = y
			// oy = -c + oy
			memcpy(oy, y->vals, sizeof(double) * csize);
			cblas_daxpy(csize, -1, c->vals, 1, oy, 1);

			// oy = B * x + oy
			mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, Bh, desc,
			                x->vals, 1, oy);
			mkl_sparse_destroy(Bh);

			// r = norm(oy)
			r = cblas_dnrm2(csize, oy, 1);
		}
		srt += timer.end("sr", false);

		int print_step = 100;
		if (it % print_step == 0) {
			printf("iteration: %d\n", it);
			printf("r: %g, s: %g\n", r, s);
			printf("time used:\n");
			printf("\tx-update: %g\n", xt / print_step);
			printf("\ty-update: %g\n", yt / print_step);
			printf("\tlmb-update: %g\n", lt / print_step);
			printf("\ts&r-update: %g\n", srt / print_step);
			xt = 0;
			yt = 0;
			lt = 0;
			srt = 0;

			double obj = cblas_ddot(k, b->vals, 1, x->vals, 1);
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

	free(c1);
	free(c2);
	free(k1);
	free(k2);
	free(oy);

	printf("ADMM finished\n");
	double tm = timer.end("admm", false);

	printf("\nresult:\n");
	printf("iteration: %d\n", it);
	printf("r=%g, s=%g\n", r, s);
	double obj = cblas_ddot(k, b->vals, 1, x->vals, 1);
	printf("obj: %g\n", obj);
	printf("ADMM time: %g\n", tm);

	csr_free(B);
	csr_free(BTB);
	if (BTBi) free(BTBi);

	if (outObj && !crossover) {
		*outObj = obj;

		cblas_dscal(m + k, -1, lmb->vals, 1);
		memcpy(outLmb, lmb->vals, sizeof(double) * csize);
	}

	if (crossover && !timeReached) {
		GRBenv* env;
		GRBmodel* model;
		GRBloadenv(&env, "crossover.log");

		printf("setting up problem\n");
		double* lb = (double*)malloc(sizeof(double) * k);
		for (int i = 0; i < k; i++) {
			lb[i] = -GRB_INFINITY;
		}
		GRBnewmodel(env, &model, "crossover", k, ba, lb, NULL, NULL, NULL);

		SM A = make_A(n);

		printf("adding constraints\n");
		for (int i = 0; i < m + k; i++) {
			int size = A.row(i).nonZeros();
			double* vals = (double*)malloc(sizeof(double) * size);
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
			double* vals = (double*)malloc(sizeof(double) * size);
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
				double* vals = (double*)malloc(sizeof(double) * size);
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

		GRBenv* modelEnv = GRBgetenv(model);
		GRBsetintparam(modelEnv, GRB_INT_PAR_METHOD, 1);
		GRBsetintparam(modelEnv, GRB_INT_PAR_UPDATEMODE, 0);
		GRBsetdblparam(modelEnv, GRB_DBL_PAR_TIMELIMIT,
		               maxTime - timer.duration("admm"));

		GRBupdatemodel(model);

		printf("setting pstart\n");
		for (int i = 0; i < k; i++) {
			GRBsetdblattrelement(model, GRB_DBL_ATTR_PSTART, i, x->vals[i]);
		}
		printf("setting dstart\n");
		cblas_dscal(2 * (m + k), -1, lmb->vals, 1);
		for (int i = 0; i < csize; i++) {
			GRBsetdblattrelement(model, GRB_DBL_ATTR_DSTART, i, lmb->vals[i]);
		}

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
				if (i >= 2 * (m + k)) {
					v *= -1;
				}
				outLmb[i] = v;
			}
			double v;
			GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &v);
			*outObj = v;
		}
	}

	vec_free(b);
	vec_free(c);
	vec_free(x);
	vec_free(y);
	vec_free(lmb);

	if (logfile) freopen("/dev/tty", "a", stdout);

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
	return admm(n, 2, b, e, NULL, NULL, 0, 1024, 1e-8);
	// return admm(n, 0, b, NULL, NULL, NULL, 0, 1024, 1e-8);
}
