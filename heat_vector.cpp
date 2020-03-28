#include "heat_vector.h"

#include <igl/avg_edge_length.h>
#include <igl/squared_edge_lengths.h>
#include <igl/per_face_normals.h>

#include <map>
#include <iostream>

template <typename DerivedV, typename Scalar>
void igl::complex_to_vector(
	const Eigen::MatrixBase<DerivedV> &V,
	const HeatVectorData<Scalar> &data,
	int vertex,
	const std::complex<Scalar> &c,
	Eigen::Matrix<Scalar, 1, 3> &vec) {

	Scalar r = std::abs(c), t = std::arg(c);
	if(t < 0) t += 2*M_PI;
	int k = 1;
	while(k < data.neighbors[vertex].size() && t > data.neighbors[vertex][k].second) ++k;
	Scalar t0 = data.neighbors[vertex][k-1].second;
	Scalar t1 = k < data.neighbors[vertex].size() ? data.neighbors[vertex][k].second : 2*M_PI;
	int j = data.neighbors[vertex][k-1].first;
	k = data.neighbors[vertex][k % data.neighbors[vertex].size()].first;

	Eigen::Matrix<Scalar, 1, 3> e1 = V.row(j) - V.row(vertex), e2 = V.row(k) - V.row(vertex);
	e1.normalize();
	e2.normalize();
	Scalar x = (t - t0) / (t1 - t0) * M_PI_2;
	vec = e1 * std::cos(x) + e2 * std::sin(x);
	vec.normalize();
	vec *= r;

}

template <typename Scalar>
void igl::vector_to_complex(
	const HeatVectorData<Scalar> &data,
	int vertex,
	const Eigen::Matrix<Scalar, 1, 3> &vec,
	std::complex<Scalar> &c) {

	c.real(data.e0.row(vertex).dot(vec));
	c.imag(data.e1.row(vertex).dot(vec));

}

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
		
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 3> MatrixX3S;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXS;
	typedef std::complex<Scalar> Complex;

	assert(F.cols() == 3 && "Only triangles are supported");
	int n = V.rows(), m = F.rows(), d = V.cols();

	// Squared edge length
	MatrixX3S l2;
	igl::squared_edge_lengths(V, F, l2);

	// Face corner angles
	MatrixX3S theta(m, 3);
	VectorXS sumTheta = VectorXS::Zero(n);
	std::vector<std::map<int, std::pair<int, int>>> edge_to_face(n);
	for(int i = 0; i < m; ++i) {
		Scalar sl2 = l2.row(i).sum();
		Scalar pl2 = l2.row(i).prod();
		for(int j = 0; j < 3; ++j) {
			theta(i, j) = std::acos(.5 * (sl2 - 2*l2(i, j)) / std::sqrt(pl2 / l2(i, j)));
			sumTheta(F(i, j)) += theta(i, j);
			edge_to_face[F(i, j)][F(i, (j+1)%3)] = {i, j};
		}
	}

	// Neighborhoods, vertex tangent plance basis and phi
	MatrixX3S face_normals(m, 3);
	igl::per_face_normals(V, F, face_normals);
	data.e0.resize(n, 3);
	data.e1 = MatrixX3S::Zero(n, 3);
	data.neighbors.clear();
	data.neighbors.resize(n);
	std::vector<std::map<int, Scalar>> phi_map(n);
	int num_triplets = n;
	for(int i = 0; i < n; ++i) {
		assert(!edge_to_face[i].empty() && "Vertices need to appear in at least one face !");
		int j0 = (*edge_to_face[i].lower_bound(0)).first;
		for(std::pair<int, std::pair<int, int>> j : edge_to_face[i]) {
			if(!edge_to_face[j.first].count(i))
				j0 = j.first;
			data.e1.row(i) += face_normals.row(j.second.first);
		}
		// Tangent plane basis
		data.e1.row(i).normalize();
		data.e0.row(i) = V.row(j0) - V.row(i);
		data.e0.row(i) -= data.e1.row(i).dot(data.e0.row(i)) * data.e1.row(i);
		data.e0.row(i).normalize();
		data.e1.row(i) = data.e1.row(i).cross(data.e0.row(i));
		// Phi
		Scalar phi = 0;
		Scalar factor = 2 * M_PI / sumTheta(i);
		int j = j0;
		do {
			data.neighbors[i].emplace_back(j, phi);
			phi_map[i][j] = phi;
			if(!edge_to_face[i].count(j)) break;
			std::pair<int, int> &face = edge_to_face[i][j];
			phi += theta(face.first, face.second) * factor;
			j = F(face.first, (face.second+2)%3);
		} while(j != j0);
		num_triplets += data.neighbors[i].size();
	}

	// Build connexion Laplacian
	Eigen::SparseMatrix<Complex> connexionLaplacian(n, n);
	std::vector<Eigen::Triplet<Complex>> IJV;
	IJV.reserve(num_triplets);
	const auto cotan = [](Scalar x) -> Scalar { return std::cos(x) / std::sin(x); };
	const auto angle_to_complex = [](Scalar x) -> Complex { return Complex(std::cos(x), std::sin(x)); };
	for(int i = 0; i < n; ++i) {
		Complex L_ii = 0;
		Complex last = 0, first = 0;
		int ne = data.neighbors[i].size(), k;
		bool is_on_border = !edge_to_face[data.neighbors[i][0].first].count(i);
		if(is_on_border) --ne;
		for(int e = 0; e < ne; ++e) {
			int j = data.neighbors[i][e].first;
			k = data.neighbors[i][(e+1) % data.neighbors[i].size()].first;
			std::pair<int, int> &face = edge_to_face[i][j];
			Scalar rho_ij = phi_map[j][i] + M_PI - phi_map[i][j];
			Scalar rho_ik = phi_map[k][i] + M_PI - phi_map[i][k];
			Scalar b = cotan(theta(face.first, (face.second+1)%3));
			Scalar c = cotan(theta(face.first, (face.second+2)%3));
			L_ii -= b + c;
			if(e == 0 && !is_on_border) first = c * angle_to_complex(rho_ij);
			else IJV.emplace_back(i, j, 0.5 * (last + c * angle_to_complex(rho_ij)));
			last = b * angle_to_complex(rho_ik);
		}
		IJV.emplace_back(i, k, 0.5*(first + last));
		IJV.emplace_back(i, i, 0.5*L_ii);
	}
	connexionLaplacian.setFromTriplets(IJV.begin(), IJV.end());

	return heat_geodesics_precompute(V, F, t, data.scalar_data);

}


template < typename Scalar, typename DerivedOmega, typename DerivedX>
void igl::heat_vector_solve(
	const HeatVectorData<Scalar> &data,
	const Eigen::MatrixBase<DerivedOmega> &Omega,
	const Eigen::MatrixBase<DerivedX> &X,
	Eigen::PlainObjectBase<DerivedX> &D) {



}

// Explicit template instantiation
template void igl::complex_to_vector<Eigen::Matrix<double, -1, -1, 0, -1, -1>, double>(
	Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> const&,
	HeatVectorData<double> const&,
	int,
	const std::complex<double> &,
	Eigen::Matrix<double, 1, 3> &);
template void igl::vector_to_complex<double>(
	HeatVectorData<double> const&,
	int,
	const Eigen::Matrix<double, 1, 3> &,
	std::complex<double> &);
template bool igl::heat_vector_precompute<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, double>(
	Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> const&,
	Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const&,
	double,
	igl::HeatVectorData<double> &);
template bool igl::heat_vector_precompute<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, double>(
	Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> const&,
	Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const&,
	igl::HeatVectorData<double> &);
template void igl::heat_vector_solve<double, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>>(
	igl::HeatVectorData<double> const&,
	Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const&,
	Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1>> const&,
	Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1>> &);