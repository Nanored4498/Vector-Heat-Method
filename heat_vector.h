#ifndef HEAT_VECTOR_H
#define HEAT_VECTOR_H

#include <igl/heat_geodesics.h>

namespace igl {

template <typename Scalar>
struct HeatVectorData {
	HeatGeodesicsData<Scalar> scalar_data;
};

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
	HeatVectorData<Scalar> &data);
// Inputs:
//   t  "heat" parameter (smaller --> more accurate, less stable)
template <typename DerivedV, typename DerivedF, typename Scalar>
bool heat_vector_precompute(
	const Eigen::MatrixBase<DerivedV> &V,
	const Eigen::MatrixBase<DerivedF> &F,
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
template < typename Scalar, typename DerivedOmega, typename DerivedX>
void heat_vector_solve(
	const HeatVectorData<Scalar> &data,
	const Eigen::MatrixBase<DerivedOmega> &Omega,
	const Eigen::MatrixBase<DerivedX> &X,
	Eigen::PlainObjectBase<DerivedX> &D);

}

#endif