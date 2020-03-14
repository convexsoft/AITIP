#include "util.h"

#include <float.h>
#include <stdlib.h>
#include <unistd.h>

#include <fstream>
#include <random>

int get_m(int n) { return (1L << (n - 2)) * n * (n - 1) / 2 + n; }

int get_k(int n) { return (1 << n) - 1; }

int D_nz(int n) {
	int n2 = n;
	int n3 = n * (n - 1) / 2;
	int n4 = n3 * ((1 << (n - 2)) - 1);
	return 2 * n2 + 3 * n3 + 4 * n4;
}

int r(int min, int max) {
	return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

SM make_A(int n) {
	int m = get_m(n);
	int k = get_k(n);

	std::vector<T> tList;

	tList.reserve(D_nz(n) + k);

	int idx, i, j, s1, s2, mask, temp;

	/*row k will contain the quantity H(X1,X2,....,XN) -
	 * H(X1,X2,...,X(k-1),X(k+1),...,XN), where k=idx*/
	for (idx = 0; idx < n; idx++) {
		s1 = 1L << idx; /*s1 is 2^(idx-1)*/  // idx representing the bit number

		tList.emplace_back(T(idx, k - 1, 1));
		/* H(X1,...,XN)*/ /*in QSopt, matrix-indices start at 0*/
		tList.emplace_back(T(idx, k - s1 - 1, -1));
		/* H( {X1,...,XN} minus {k} ) */  // All bits except bit idx
	}

	/* this loop is for I(X_{i+1}; X_{j+1} | X_K), where K is a subset of {
	 * {1,...,number_vars} minus {i+1, j+1}} */
	for (i = 0; i < (n - 1); i++) {
		for (j = i + 1; j < n; j++) {
			s1 = 1L << i; /*the single variable X_k has coordinate (1L << k-1)
			                    in the joint entropy space (k =
			                    1,...,number_vars)*/
			s2 = 1L << j;
			mask = s1 | s2; /*this is the idx of (X_(i+1),X_(j+1))*/

			/*first row ( for K = {} ):*/          // H(X_(i+1)) + H(X_(j+1)) >=
			                                       // H(X_(i+1),X_(j+1))
			tList.emplace_back(T(idx, s1 - 1, 1)); /*H(X_(i+1))*/
			tList.emplace_back(T(idx, s2 - 1, 1)); /*H(X_(j+1))*/
			tList.emplace_back(T(idx, mask - 1, -1)); /* H(X_(i+1),X_(j+1)) */

			idx++;

			/*additional rows for all non-empty K:*/
			for (temp = 1; temp <= k;
			     temp++) { /* temp runs through all possible subsets of (X_1,
				              ..., X_N)*/
				if (!(temp & mask)) { /*if K != (i+1,j+1)*/

					tList.emplace_back(
					    T(idx, (s1 | temp) - 1, 1)); /* H(X_(i+1), X_K) */
					tList.emplace_back(
					    T(idx, (s2 | temp) - 1, 1)); /* H(X_(j+1), X_K) */
					tList.emplace_back(T(idx, temp - 1, -1)); /* H(X_K) */
					tList.emplace_back(T(idx, (s1 | s2 | temp) - 1,
					                     -1)); /* H(X_(i+1), X_(j+1), X_K) */

					idx++;
				}
			}
		}
	}

	for (i = 0; i < k; i++) {
		tList.emplace_back(T(idx, i, 1));
		idx++;
	}

	SM A(m + k, k);
	A.setFromTriplets(tList.begin(), tList.end());

	return A;
}

SM make_B(int n) {
	int m = get_m(n);
	int k = get_k(n);

	std::vector<T> tList;

	tList.reserve(2 * (D_nz(n) + k));

	int idx, i, j, s1, s2, mask, temp;

	/*row k will contain the quantity H(X1,X2,....,XN) -
	 * H(X1,X2,...,X(k-1),X(k+1),...,XN), where k=idx*/
	for (idx = 0; idx < n; idx++) {
		s1 = 1L << idx; /*s1 is 2^(idx-1)*/  // idx representing the bit number

		tList.emplace_back(T(idx, k - 1, 1));
		/* H(X1,...,XN)*/ /*in QSopt, matrix-indices start at 0*/
		tList.emplace_back(T(idx, k - s1 - 1, -1));
		/* H( {X1,...,XN} minus {k} ) */  // All bits except bit idx
	}

	/* this loop is for I(X_{i+1}; X_{j+1} | X_K), where K is a subset of {
	 * {1,...,number_vars} minus {i+1, j+1}} */
	for (i = 0; i < (n - 1); i++) {
		for (j = i + 1; j < n; j++) {
			s1 = 1L << i; /*the single variable X_k has coordinate (1L << k-1)
			                    in the joint entropy space (k =
			                    1,...,number_vars)*/
			s2 = 1L << j;
			mask = s1 | s2; /*this is the idx of (X_(i+1),X_(j+1))*/

			/*first row ( for K = {} ):*/          // H(X_(i+1)) + H(X_(j+1)) >=
			                                       // H(X_(i+1),X_(j+1))
			tList.emplace_back(T(idx, s1 - 1, 1)); /*H(X_(i+1))*/
			tList.emplace_back(T(idx, s2 - 1, 1)); /*H(X_(j+1))*/
			tList.emplace_back(T(idx, mask - 1, -1)); /* H(X_(i+1),X_(j+1)) */

			idx++;

			/*additional rows for all non-empty K:*/
			for (temp = 1; temp <= k;
			     temp++) { /* temp runs through all possible subsets of (X_1,
				              ..., X_N)*/
				if (!(temp & mask)) { /*if K != (i+1,j+1)*/

					tList.emplace_back(
					    T(idx, (s1 | temp) - 1, 1)); /* H(X_(i+1), X_K) */
					tList.emplace_back(
					    T(idx, (s2 | temp) - 1, 1)); /* H(X_(j+1), X_K) */
					tList.emplace_back(T(idx, temp - 1, -1)); /* H(X_K) */
					tList.emplace_back(T(idx, (s1 | s2 | temp) - 1,
					                     -1)); /* H(X_(i+1), X_(j+1), X_K) */

					idx++;
				}
			}
		}
	}

	for (i = 0; i < k; i++) {
		tList.emplace_back(T(idx, i, 1));
		idx++;
	}

	/*row k will contain the quantity H(X1,X2,....,XN) -
	 * H(X1,X2,...,X(k-1),X(k+1),...,XN), where k=idx*/
	for (i = 0; i < n; i++) {
		s1 = 1L << i; /*s1 is 2^(idx-1)*/  // idx representing the bit number

		tList.emplace_back(T(idx, k - 1, 1));
		/* H(X1,...,XN)*/ /*in QSopt, matrix-indices start at 0*/
		tList.emplace_back(T(idx, k - s1 - 1, -1));
		/* H( {X1,...,XN} minus {k} ) */  // All bits except bit idx
		idx++;
	}

	/* this loop is for I(X_{i+1}; X_{j+1} | X_K), where K is a subset of {
	 * {1,...,number_vars} minus {i+1, j+1}} */
	for (i = 0; i < (n - 1); i++) {
		for (j = i + 1; j < n; j++) {
			s1 = 1L << i; /*the single variable X_k has coordinate (1L << k-1)
			                    in the joint entropy space (k =
			                    1,...,number_vars)*/
			s2 = 1L << j;
			mask = s1 | s2; /*this is the idx of (X_(i+1),X_(j+1))*/

			/*first row ( for K = {} ):*/          // H(X_(i+1)) + H(X_(j+1)) >=
			                                       // H(X_(i+1),X_(j+1))
			tList.emplace_back(T(idx, s1 - 1, 1)); /*H(X_(i+1))*/
			tList.emplace_back(T(idx, s2 - 1, 1)); /*H(X_(j+1))*/
			tList.emplace_back(T(idx, mask - 1, -1)); /* H(X_(i+1),X_(j+1)) */

			idx++;

			/*additional rows for all non-empty K:*/
			for (temp = 1; temp <= k;
			     temp++) { /* temp runs through all possible subsets of (X_1,
				              ..., X_N)*/
				if (!(temp & mask)) { /*if K != (i+1,j+1)*/

					tList.emplace_back(
					    T(idx, (s1 | temp) - 1, 1)); /* H(X_(i+1), X_K) */
					tList.emplace_back(
					    T(idx, (s2 | temp) - 1, 1)); /* H(X_(j+1), X_K) */
					tList.emplace_back(T(idx, temp - 1, -1)); /* H(X_K) */
					tList.emplace_back(T(idx, (s1 | s2 | temp) - 1,
					                     -1)); /* H(X_(i+1), X_(j+1), X_K) */

					idx++;
				}
			}
		}
	}

	for (i = 0; i < k; i++) {
		tList.emplace_back(T(idx, i, 1));
		idx++;
	}

	SM B(2 * (m + k), k);
	B.setFromTriplets(tList.begin(), tList.end());

	return B;
}

void get_fac_dir(char* dest) {
#ifndef FAC_DIR
#error "-DFAC_DIR not set. Please check the Makefile."
#endif
	strcpy(dest, STR(FAC_DIR));
}
