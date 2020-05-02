#ifndef HEAT_VECTOR_H
#define HEAT_VECTOR_H

#include <Eigen/SparseCholesky>
#include <vector>

namespace igl {

template <typename Scalar>
struct HeatLogData {
	int j;
	std::complex<Scalar> x, w;
	HeatLogData(int j, std::complex<Scalar> x, std::complex<Scalar> w): j(j), x(x), w(w) {}
};
template <typename Scalar>
struct HeatVectorData {
	Eigen::Matrix<Scalar, Eigen::Dynamic, 3> e0, e1;
	std::vector<std::vector<std::pair<int, Scalar>>> neighbors;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> scal_solver, poisson_solver;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<std::complex<Scalar>>> vec_solver;
    bool use_intrinsic_delaunay = false;
	std::vector<std::vector<HeatLogData<Scalar>>> log_data;
};

template <typename DerivedV, typename Scalar>
void complex_to_vector(
	const Eigen::MatrixBase<DerivedV> &V,
	const HeatVectorData<Scalar> &data,
	int vertex,
	const std::complex<Scalar> &c,
	Eigen::Matrix<Scalar, 1, 3> &vec);

template <typename Scalar>
void vector_to_complex(
	const HeatVectorData<Scalar> &data,
	int vertex,
	const Eigen::Matrix<Scalar, 1, 3> &vec,
	std::complex<Scalar> &c);

// Precompute factorized solvers for computing a fast approximation of
// parallel transport along geodesics on a mesh (V,F). [Sharp et al. 2019]
//
// Inputs:
//   V  #V by dim list of mesh vertex positions
//   F  #F by 3 list of mesh face indices into V
// Outputs:
//   data  precomputation data (see heat_vector_solve)
template <typename DerivedV, typename DerivedF, typename Scalar>
bool heat_vector_precompute(
	const Eigen::MatrixBase<DerivedV> &V,
	const Eigen::MatrixBase<DerivedF> &F,
	bool precompute_log,
	HeatVectorData<Scalar> &data);
// Inputs:
//   t  "heat" parameter (smaller --> more accurate, less stable)
template <typename DerivedV, typename DerivedF, typename Scalar>
bool heat_vector_precompute(
	const Eigen::MatrixBase<DerivedV> &V,
	const Eigen::MatrixBase<DerivedF> &F,
	bool precompute_log,
	const Scalar t,
	HeatVectorData<Scalar> &data);

// Compute fast approximate parallel transport along geodesics using precomputed
// data from a set of selected source vertices (Omega)
//
// Inputs: 
//   data	precomputation data (see heat_vector_precompute)
//   Omega  #Omega list of indices into V of source vertices
//   X		#Omega list of vectors in the source Omega
// Outputs:
//   D		#V list of vectors transported from Omega
template <typename Scalar, typename DerivedOmega, typename DerivedX>
void heat_vector_solve(
	const HeatVectorData<Scalar> &data,
	const Eigen::MatrixBase<DerivedOmega> &Omega,
	const Eigen::MatrixBase<DerivedX> &X,
	Eigen::PlainObjectBase<DerivedX> &res);

template <typename Scalar, typename DerivedOmega>
void heat_voronoi_solve(
	const HeatVectorData<Scalar> &data,
	const Eigen::MatrixBase<DerivedOmega> &Omega,
	Eigen::VectorXi &res);

template <typename Scalar>
void heat_R_solve(
	const HeatVectorData<Scalar> &data,
	int vertex,
	Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> &res);

template <typename Scalar>
void heat_log_solve(
	const HeatVectorData<Scalar> &data,
	int vertex,
	Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> &res);

template <typename Scalar, typename DerivedOmega, typename DerivedCoord>
void heat_log_solve(
	const HeatVectorData<Scalar> &data,
	const Eigen::MatrixBase<DerivedOmega> &Omega,
	const Eigen::MatrixBase<DerivedCoord> &coord,
	Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> &res);

}

#endif