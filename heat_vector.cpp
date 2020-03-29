#include "heat_vector.h"

#include <igl/avg_edge_length.h>
#include <igl/edge_lengths.h>
#include <igl/squared_edge_lengths.h>
#include <igl/intrinsic_delaunay_triangulation.h>
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

template <typename T>
void add_complex_triplet(std::vector<Eigen::Triplet<T>> &IJV, int i, int j, const std::complex<T> &c) {
	int i2 = i<<1, j2 = j<<1;
	IJV.emplace_back(i2, j2, c.real());
	IJV.emplace_back(i2, j2+1, -c.imag());
	IJV.emplace_back(i2+1, j2, c.imag());
	IJV.emplace_back(i2+1, j2+1, c.real());
}

template <typename DerivedV, typename DerivedF, typename Scalar>
bool igl::heat_vector_precompute(
	const Eigen::MatrixBase<DerivedV> &V,
	const Eigen::MatrixBase<DerivedF> &F0,
	const Scalar t,
	HeatVectorData<Scalar> &data) {
		
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 3> MatrixX3S;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXS;
	typedef std::complex<Scalar> Complex;

	assert(F.cols() == 3 && "Only triangles are supported");
	int n = V.rows(), m = F0.rows(), d = V.cols();

	// Squared edge length
	MatrixX3S l2;
	DerivedF F_intrinsic;
	if(data.use_intrinsic_delaunay) {
		MatrixX3S l;
		igl::edge_lengths(V, F0, l);
		igl::intrinsic_delaunay_triangulation(l, F0, l2, F_intrinsic);
		l2 = l2.array().square().eval();
	} else igl::squared_edge_lengths(V, F0, l2);
	const DerivedF &F = data.use_intrinsic_delaunay ? F_intrinsic : F0;

	// Face corner angles and face areas
	MatrixX3S theta(m, 3);
	VectorXS sumTheta = VectorXS::Zero(n);
	VectorXS faceArea(m);
	std::vector<std::map<int, std::pair<int, int>>> edge_to_face(n);
	for(int i = 0; i < m; ++i) {
		Scalar dblA = std::sqrt(l2(i, 1) * l2(i, 2));
		theta(i, 0) = std::acos(.5 * (l2(i, 1) + l2(i, 2) - l2(i, 0)) / dblA);
		theta(i, 1) = std::acos(.5 * (l2(i, 0) + l2(i, 2) - l2(i, 1)) / std::sqrt(l2(i, 0) * l2(i, 2)));
		theta(i, 2) = M_PI - theta(i, 0) - theta(i, 1);
		faceArea(i) = 0.5 * std::sin(theta(i, 0)) * dblA;
		for(int j = 0; j < 3; ++j) {
			sumTheta(F(i, j)) += theta(i, j);
			edge_to_face[F(i, j)][F(i, (j+1)%3)] = {i, j};
		}
	}

	// Neighborhoods, vertex tangent plance basis, mass matrix and phi
	MatrixX3S face_normals(m, 3);
	igl::per_face_normals(V, F, face_normals);
	data.e0.resize(n, 3);
	data.e1 = MatrixX3S::Zero(n, 3);
	data.neighbors.clear();
	data.neighbors.resize(n);
	std::vector<std::map<int, Scalar>> phi_map(n);
	Eigen::SparseMatrix<Scalar> M(n, n), cM(2*n, 2*n);
	std::vector<Eigen::Triplet<Scalar>> IJV, cIJV;
	IJV.reserve(n);
	cIJV.reserve(2*n);
	int num_triplets = n;
	for(int i = 0; i < n; ++i) {
		assert(!edge_to_face[i].empty() && "Vertices need to appear in at least one face !");
		Scalar M_i = 0;
		int j0 = (*edge_to_face[i].lower_bound(0)).first;
		for(std::pair<int, std::pair<int, int>> j : edge_to_face[i]) {
			if(!edge_to_face[j.first].count(i))
				j0 = j.first;
			M_i += faceArea[j.second.first];
			data.e1.row(i) += face_normals.row(j.second.first);
		}
		// Mass matrix coeff
		IJV.emplace_back(i, i, M_i/3);
		cIJV.emplace_back(2*i, 2*i, IJV.back().value());
		cIJV.emplace_back(2*i+1, 2*i+1, IJV.back().value());
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
	M.setFromTriplets(IJV.begin(), IJV.end());
	cM.setFromTriplets(cIJV.begin(), cIJV.end());

	// Build connexion Laplacian
	Eigen::SparseMatrix<Scalar> L(n, n), cL(2*n, 2*n);
	IJV.clear();
	IJV.reserve(num_triplets);
	cIJV.clear();
	cIJV.reserve(4*num_triplets-2*n);
	const auto cotan = [](Scalar x) -> Scalar { return std::cos(x) / std::sin(x); };
	const auto add_non_diag = [&IJV, &cIJV, &phi_map](int i, int j, Scalar s)->void {
		IJV.emplace_back(i, j, s);
		IJV.emplace_back(j, i, s);
		Scalar rho_ij = phi_map[i][j] - phi_map[j][i] + M_PI;
		Complex co = s * std::polar(1., rho_ij);
		add_complex_triplet(cIJV, i, j, co);
		add_complex_triplet(cIJV, j, i, std::conj(co));
	};
	for(int i = 0; i < n; ++i) {
		Scalar L_ii = 0;
		Scalar last = 0, first = 0;
		int ne = data.neighbors[i].size(), k;
		bool is_on_border = !edge_to_face[data.neighbors[i][0].first].count(i);
		if(is_on_border) --ne;
		for(int e = 0; e < ne; ++e) {
			int j = data.neighbors[i][e].first;
			k = data.neighbors[i][(e+1) % data.neighbors[i].size()].first;
			std::pair<int, int> &face = edge_to_face[i][j];
			Scalar b = cotan(theta(face.first, (face.second+1)%3));
			Scalar c = cotan(theta(face.first, (face.second+2)%3));
			L_ii -= b + c;
			if(e == 0 && !is_on_border) first = c;
			else if(i < j) add_non_diag(i, j, 0.5*(last + c));
			last = b;
		}
		if(i < k) add_non_diag(i, k, 0.5*(last + first));
		IJV.emplace_back(i, i, 0.5*L_ii);
		cIJV.emplace_back(2*i, 2*i, IJV.back().value());
		cIJV.emplace_back(2*i+1, 2*i+1, IJV.back().value());
	}
	L.setFromTriplets(IJV.begin(), IJV.end());
	cL.setFromTriplets(cIJV.begin(), cIJV.end());

	// Solvers
	data.scal_solver.compute(M - t*L);
	if(data.scal_solver.info() != Eigen::Success) return false;
	data.vec_solver.compute(cM - t*cL);
	if(data.vec_solver.info() != Eigen::Success) return false;

	return true;

}


template <typename Scalar, typename DerivedOmega, typename DerivedX>
void igl::heat_vector_solve(
	const HeatVectorData<Scalar> &data,
	const Eigen::MatrixBase<DerivedOmega> &Omega,
	const Eigen::MatrixBase<DerivedX> &X,
	Eigen::PlainObjectBase<DerivedX> &res) {

	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXS;

	int n = data.neighbors.size();

	VectorXS Y0 = VectorXS::Zero(2*n);
	for(int i = 0; i < Omega.size(); ++i) {
		int x = 2*Omega(i);
		Y0(x) = X(i).real();
		Y0(x+1) = X(i).imag();
	}
	VectorXS Y = data.vec_solver.solve(Y0);

	VectorXS u0 = VectorXS::Zero(n), phi0 = VectorXS::Zero(n);
	for(int i = 0; i < Omega.size(); ++i) {
		int x = Omega(i);
		u0(x) = std::abs(X(i));
		phi0(x) = 1.0;
	}
	VectorXS u = data.scal_solver.solve(u0);
	VectorXS phi = data.scal_solver.solve(phi0);

	res.resize(n, 1);
	for(int i = 0; i < n; ++i) {
		res(i) = std::complex<Scalar>(Y(2*i), Y(2*i+1));
		res(i) /= std::abs(res(i));
		res(i) *= u(i) / phi(i);
	}

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
template void igl::heat_vector_solve<double, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<std::complex<double>, -1, 1, 0, -1, 1>>(
	igl::HeatVectorData<double> const&,
	Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const&,
	Eigen::MatrixBase<Eigen::Matrix<std::complex<double>, -1, 1, 0, -1, 1>> const&,
	Eigen::PlainObjectBase<Eigen::Matrix<std::complex<double>, -1, 1, 0, -1, 1>> &);