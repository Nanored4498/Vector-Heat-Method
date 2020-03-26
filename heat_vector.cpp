#include "heat_vector.h"

template <typename DerivedV, typename DerivedF, typename Scalar>
bool igl::heat_vector_precompute(
	const Eigen::MatrixBase<DerivedV> &V,
	const Eigen::MatrixBase<DerivedF> &F,
	HeatVectorData<Scalar> &data) {

  // default t value
  const Scalar h = avg_edge_length(V,F);
  const Scalar t = h*h;
  return heat_vector_precompute(V,F,t,data);

}


template <typename DerivedV, typename DerivedF, typename Scalar>
bool igl::heat_vector_precompute(
	const Eigen::MatrixBase<DerivedV> &V,
	const Eigen::MatrixBase<DerivedF> &F,
	const Scalar t,
	HeatVectorData<Scalar> &data) {

	return heat_geodesics_precompute(V, F, t, data.scalar_data);

}


template < typename Scalar, typename DerivedOmega, typename DerivedX>
void igl::heat_vector_solve(
	const HeatVectorData<Scalar> &data,
	const Eigen::MatrixBase<DerivedOmega> &Omega,
	const Eigen::MatrixBase<DerivedX> &X,
	Eigen::PlainObjectBase<DerivedX> &D) {



}