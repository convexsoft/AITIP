#include <stdio.h>
#include <time.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

#ifndef ITIP_EIGEN_UTIL_H
#define ITIP_EIGEN_UTIL_H

typedef Eigen::VectorXd V;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    M;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SM;
typedef Eigen::SparseMatrix<double> CSM;
typedef Eigen::Triplet<double> T;
typedef SM::InnerIterator IT;
typedef CSM::InnerIterator CIT;

#define XSTR(x) #x
#define STR(x) XSTR(x)

int get_k(int n);
int get_m(int n);

int r(int min, int max);

SM make_A(int n);
SM make_B(int n);

template <typename T = double>
void ppA(T* ary, int len, std::string name = "", int maxlen = 20) {
	if (name != "") std::cout << name + " = ";
	printf("[");
	if (len < maxlen) {
		for (int i = 0; i < len; i++) {
			std::cout << " ";
			std::cout << ary[i];
			if (i < len - 1) std::cout << ",";
			std::cout << " ";
		}
	} else {
		for (int i = 0; i < 5; i++) {
			std::cout << " ";
			std::cout << ary[i];
			if (i < 4) std::cout << ",";
			std::cout << " ";
		}
		printf(" ... ");
		for (int i = 0; i < 5; i++) {
			std::cout << " ";
			std::cout << ary[len - 5 + i];
			if (i < 4) std::cout << ",";
			std::cout << " ";
		}
	}
	printf("]\n");
}

void get_fac_dir(char* dest);

#endif  // ITIP_EIGEN_UTIL_H
