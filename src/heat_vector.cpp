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
	bool precompute_log,
	HeatVectorData<Scalar> &data) {

  // default t value
  const Scalar h = avg_edge_length(V,F);
  const Scalar t = h*h;
  return heat_vector_precompute(V,F,precompute_log,t,data);

}

template <typename DerivedV, typename DerivedF, typename Scalar>
bool igl::heat_vector_precompute(
	const Eigen::MatrixBase<DerivedV> &V,
	const Eigen::MatrixBase<DerivedF> &F0,
	bool precompute_log,
	const Scalar t,
	HeatVectorData<Scalar> &data) {
		
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 3> MatrixX3S;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXS;
	typedef std::complex<Scalar> Complex;

	int n = V.rows(), m = F0.rows(), d = V.cols();

	// Squared edge length
	MatrixX3S l2;
	igl::squared_edge_lengths(V, F0, l2);

	// Face corner angles and face areas
	MatrixX3S theta(m, 3);
	VectorXS sumTheta = VectorXS(n);
	VectorXS faceArea(m);
	std::vector<std::map<int, std::pair<int, int>>> edge_to_face(n);
	const auto compute_theta = [&](const DerivedF &F)->void {
		for(int i = 0; i < n; ++i)
			edge_to_face[i].clear(), sumTheta(i) = 0;
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
	};
	compute_theta(F0);

	// Compute phi for the vertex i starting at the neighbor j0
	const auto compute_phi = [&](int i, int j0, Scalar phi, const DerivedF &F, const std::function<void(int, Scalar)> &fun)->void {
		Scalar factor = 2 * M_PI / sumTheta(i);
		int j = j0;
		do {
			fun(j, phi);
			if(!edge_to_face[i].count(j)) break;
			std::pair<int, int> &face = edge_to_face[i][j];
			phi += theta(face.first, face.second) * factor;
			j = F(face.first, (face.second+2)%3);
		} while(j != j0);
	};

	// Neighborhoods and vertex tangent plance basis
	data.neighbors.clear();
	data.neighbors.resize(n);
	MatrixX3S face_normals(m, 3);
	igl::per_face_normals(V, F0, face_normals);
	data.e0.resize(n, 3);
	data.e1 = MatrixX3S::Zero(n, 3);
	for(int i = 0; i < n; ++i) {
		// First neighbor
		int j0 = (*edge_to_face[i].lower_bound(0)).first;
		for(const std::pair<int, std::pair<int, int>> &j : edge_to_face[i]) {
			if(!edge_to_face[j.first].count(i)) j0 = j.first;
			data.e1.row(i) += face_normals.row(j.second.first);
		}
		// Tangent plane basis
		data.e1.row(i).normalize();
		data.e0.row(i) = V.row(j0) - V.row(i);
		data.e0.row(i) -= data.e1.row(i).dot(data.e0.row(i)) * data.e1.row(i);
		data.e0.row(i).normalize();
		data.e1.row(i) = data.e1.row(i).cross(data.e0.row(i));
		// Neighborhood
		compute_phi(i, j0, 0, F0, [i, &data](int j, Scalar phi) { data.neighbors[i].emplace_back(j, phi); });
	}

	// Phi
	std::vector<std::map<int, Scalar>> phi_map(n);
	DerivedF F_intrinsic;
	if(data.use_intrinsic_delaunay) {
		MatrixX3S l = l2.array().sqrt().eval();
		igl::intrinsic_delaunay_triangulation(l, F0, l2, F_intrinsic);
		l2 = l2.array().square().eval();
		compute_theta(F_intrinsic);
		for(int i = 0; i < n; ++i) {
			int j0 = data.neighbors[i][0].first;
			Scalar phi = 0;
			if(!edge_to_face[i].count(j0)) {
				j0 = (*edge_to_face[i].lower_bound(0)).first;
				Complex c;
				Eigen::Matrix<Scalar, 1, 3> vec = V.row(j0)-V.row(i);
				igl::vector_to_complex(data, i, vec, c);
				phi = std::arg(c);
			}
			compute_phi(i, j0, phi, F_intrinsic,
					[i, &phi_map](int j, Scalar phi) { phi_map[i][j] = phi; });
		}
	} else {
		for(int i = 0; i < n; ++i)
			for(const std::pair<int, Scalar> &j : data.neighbors[i])
				phi_map[i][j.first] = j.second;
	}
	int num_triplets = n;
	for(int i = 0; i < n; ++i) num_triplets += phi_map[i].size();
	const DerivedF &F = data.use_intrinsic_delaunay ? F_intrinsic : F0;

	// Build connexion Laplacian
	std::vector<Eigen::Triplet<Scalar>> IJV;
	IJV.reserve(num_triplets);
	std::vector<Eigen::Triplet<Complex>> cIJV;
	cIJV.reserve(num_triplets);
	const auto cotan = [](Scalar x) -> Scalar { return std::cos(x) / std::sin(x); };
	const auto add_non_diag = [&IJV, &cIJV, &phi_map](int i, int j, Scalar s)->void {
		IJV.emplace_back(i, j, s);
		Scalar rho_ij = phi_map[i][j] - phi_map[j][i] + M_PI;
		cIJV.emplace_back(i, j, s * std::polar(1., rho_ij));
	};
	for(int i = 0; i < n; ++i) {
		Scalar M_i = 0, L_i = 0;
		Scalar last = 0, first = 0;
		int j0 = data.neighbors[i][0].first;
		if(!edge_to_face[i].count(j0)) j0 = (*edge_to_face[i].lower_bound(0)).first;
		int j=j0, k;
		if(!edge_to_face[j0].count(i)) j0 = data.neighbors[i].back().first;
		do {
			std::pair<int, int> &face = edge_to_face[i][j];
			k = F(face.first, (face.second+2)%3);
			Scalar b = cotan(theta(face.first, (face.second+1)%3));
			Scalar c = cotan(theta(face.first, (face.second+2)%3));
			M_i += faceArea(face.first);
			L_i += b + c;
			if(j == j0) first = c;
			else add_non_diag(i, j, -t*0.5*(last + c));
			last = b;
			j = k;
		} while(j != j0);
		add_non_diag(i, k, -t*0.5*(last + first));
		IJV.emplace_back(i, i, M_i/3 + t*0.5*L_i);
		cIJV.emplace_back(i, i, IJV.back().value());
	}
	Eigen::SparseMatrix<Scalar> Q(n, n);
	Q.setFromTriplets(IJV.begin(), IJV.end());
	Eigen::SparseMatrix<Complex> cQ(n, n);
	cQ.setFromTriplets(cIJV.begin(), cIJV.end());

	// Solvers
	data.scal_solver.compute(Q);
	if(data.scal_solver.info() != Eigen::Success) return false;
	data.vec_solver.compute(cQ);
	if(data.vec_solver.info() != Eigen::Success) return false;

	if(!precompute_log) return true;

	data.x_log.clear();
	data.x_log.resize(n);
	l2 = l2.array().sqrt().eval();
	for(int i = 0; i < n; ++i) {
		Complex x_i = 0;
		Complex last=0, first=0;
		Scalar theta_factor = 2 * M_PI / sumTheta(i);
		int j0 = data.neighbors[i][0].first;
		if(!edge_to_face[i].count(j0)) j0 = (*edge_to_face[i].lower_bound(0)).first;
		int j=j0, k;
		if(!edge_to_face[j0].count(i)) j0 = data.neighbors[i].back().first;
		do {
			std::pair<int, int> &face = edge_to_face[i][j];
			k = F(face.first, (face.second+2)%3);
			Scalar l_ik = l2(face.first, (face.second+1)%3);
			Scalar l_ij = l2(face.first, (face.second+2)%3);
			Scalar A = 4 * faceArea(face.first);
			Scalar alpha = theta(face.first, face.second) * theta_factor;
			Scalar sin_alpha = std::sin(alpha), cos_alpha = std::cos(alpha);
			Complex c = Complex(alpha*sin_alpha, sin_alpha - alpha*cos_alpha) / A;
			Complex add_j = l_ik * c;
			Complex add_i = Complex(-l_ij*sin_alpha*sin_alpha, l_ij*(cos_alpha*sin_alpha-alpha)) / A - add_j;
			x_i += std::polar(1., phi_map[i][j]) * add_i;
			if(j == j0) first = add_j;
			else data.x_log[i].emplace_back(j, - std::polar(1., phi_map[j][i]) * (last+add_j));
			last = l_ij * std::conj(c);
			j = k;
		} while(j != j0);
		data.x_log[i].emplace_back(k, - std::polar(1., phi_map[k][i]) * (last+first));
		data.x_log[i].emplace_back(i, x_i);
	}

	return true;

}


template <typename Scalar, typename DerivedOmega, typename DerivedX>
void igl::heat_vector_solve(
	const HeatVectorData<Scalar> &data,
	const Eigen::MatrixBase<DerivedOmega> &Omega,
	const Eigen::MatrixBase<DerivedX> &X,
	Eigen::PlainObjectBase<DerivedX> &res) {

	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXS;
	typedef Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> VectorXC;

	int n = data.neighbors.size();

	VectorXC Y0 = VectorXC::Zero(n);
	for(int i = 0; i < Omega.size(); ++i)
		Y0(Omega(i)) = X(i);
	res = data.vec_solver.solve(Y0);

	VectorXS u0 = VectorXS::Zero(n), phi0 = VectorXS::Zero(n);
	for(int i = 0; i < Omega.size(); ++i) {
		int x = Omega(i);
		u0(x) = std::abs(X(i));
		phi0(x) = 1.0;
	}
	VectorXS u = data.scal_solver.solve(u0);
	VectorXS phi = data.scal_solver.solve(phi0);

	for(int i = 0; i < n; ++i)
		res(i) *= u(i) / phi(i) / std::abs(res(i));

}

template <typename Scalar, typename DerivedOmega>
void igl::heat_voronoi_solve(
	const HeatVectorData<Scalar> &data,
	const Eigen::MatrixBase<DerivedOmega> &Omega,
	Eigen::VectorXi &res) {

	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXS;

	int n = data.neighbors.size();

	VectorXS u0 = VectorXS::Zero(n), phi0 = VectorXS::Zero(n);
	std::cout << "test " << Omega.size() << " " << Omega(0) << std::endl;
	for(int i = 0; i < Omega.size(); ++i) {
		int x = Omega(i);
		u0(x) = i;
		phi0(x) = 1.0;
	}
	VectorXS u = data.scal_solver.solve(u0);
	VectorXS phi = data.scal_solver.solve(phi0);
	res.resize(n);
	for(int i = 0; i < n; ++i)
		res(i) = std::max(0, std::min(int(Omega.size())-1, int(0.5+u(i)/phi(i))));
	for(int i = 0; i < n; ++i) {
		int count = 0;
		for(const std::pair<int, Scalar> &p : data.neighbors[i])
			if(res(p.first) == res(i)) ++count;
		if(count < 0.7*data.neighbors[i].size()) res(i) = -1;
	}

}

template <typename Scalar>
void igl::heat_log_solve(
	const HeatVectorData<Scalar> &data,
	int vertex,
	Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> &res) {

	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXS;
	typedef std::complex<Scalar> Complex;
	typedef Eigen::Matrix<Complex, Eigen::Dynamic, 1> VectorXC;

	int n = data.neighbors.size();

	VectorXC H;
	igl::heat_vector_solve(data, (Eigen::VectorXi(1) << vertex).finished(),  (VectorXC(1) << 1.).finished(), H);

	Eigen::VectorXi Omega(data.x_log[vertex].size());
	VectorXC x(Omega.size());
	int i = 0;
	for(const std::pair<int, Complex> &p : data.x_log[vertex])
		x[i] = p.second, Omega(i++) = p.first;
	igl::heat_vector_solve(data, Omega, x, res);
	for(i = 0; i < n; ++i) res(i) /= abs(res(i));
	res(vertex) = 0;

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
	bool,
	double,
	igl::HeatVectorData<double> &);
template bool igl::heat_vector_precompute<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, double>(
	Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> const&,
	Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const&,
	bool,
	igl::HeatVectorData<double> &);
template void igl::heat_vector_solve<double, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<std::complex<double>, -1, 1, 0, -1, 1>>(
	igl::HeatVectorData<double> const&,
	Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const&,
	Eigen::MatrixBase<Eigen::Matrix<std::complex<double>, -1, 1, 0, -1, 1>> const&,
	Eigen::PlainObjectBase<Eigen::Matrix<std::complex<double>, -1, 1, 0, -1, 1>> &);
template void igl::heat_voronoi_solve<double, Eigen::Matrix<int, -1, 1, 0, -1, 1>>(
	igl::HeatVectorData<double> const&,
	Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const&,
	Eigen::VectorXi &);
template void igl::heat_log_solve<double>(
	igl::HeatVectorData<double> const&,
	int,
	Eigen::Matrix<std::complex<double>, -1, 1> &);