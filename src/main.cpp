#include <igl/read_triangle_mesh.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/avg_edge_length.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/heat_geodesics.h>
#include <igl/opengl/create_shader_program.h>
#include <igl/opengl/destroy_shader_program.h>
#include <igl/png/writePNG.h>
#include <GLFW/glfw3.h>

#include "heat_vector.h"
#include "my_draw_buffer.h"

#include <vector>
#include <iostream>
#include <fstream>

#define COMPUTE_PARALLEL_TRANSPORT 0
#define COMPUTE_VORONOI 1
#define COMPUTE_H_R 2
#define COMPUTE_LOG 3
#define COMPUTE_NUMBER 4

void load_standard_shader(igl::opengl::glfw::Viewer &viewer, int data_id) {
	viewer.data(data_id).meshgl.is_initialized = false;
	viewer.data(data_id).meshgl.init();
}

void load_my_shader(igl::opengl::glfw::Viewer &viewer, int data_id, std::string frag) {
	load_standard_shader(viewer, data_id);
	std::ifstream f("../src/shader.vert");
	std::string mesh_vertex_shader_string((std::istreambuf_iterator<char>(f)), (std::istreambuf_iterator<char>()));
	f.close();
	f.open(frag);
	std::string mesh_fragment_shader_string((std::istreambuf_iterator<char>(f)), (std::istreambuf_iterator<char>()));
	f.close();
	igl::opengl::destroy_shader_program(viewer.data(data_id).meshgl.shader_mesh);
	igl::opengl::create_shader_program(
		mesh_vertex_shader_string,
		mesh_fragment_shader_string,
		{},
		viewer.data(data_id).meshgl.shader_mesh);
}

int main(int argc, char *argv[]) {

	// Load Mesh
	Eigen::MatrixXi F;
	Eigen::MatrixXd V;
	igl::read_triangle_mesh(argc>1?argv[1]: "../meshes/bunny.obj", V, F);

	// Precomputation for vector heat method [In build]
	igl::HeatVectorData<double> hvm_data;
	igl::HeatGeodesicsData<double> geod_data;
	double avg_l = igl::avg_edge_length(V,F);
	double t = std::pow(avg_l, 2);
	const auto precompute = [&]() {
		if(!igl::heat_vector_precompute(V, F, true, t, hvm_data)) {
			std::cerr << "Error: heat_vector_precompute failed." << std::endl;
			exit(EXIT_FAILURE);
		};
		if(!igl::heat_geodesics_precompute(V, F, t, geod_data)) {
			std::cerr << "Error: heat_geodesic_precompute failed." << std::endl;
			exit(EXIT_FAILURE);
		};
	};
	precompute();

	int mode = COMPUTE_PARALLEL_TRANSPORT;

	// Show mesh
	igl::opengl::glfw::Viewer viewer;
	int mesh_id = viewer.data().id;
	viewer.data(mesh_id).set_mesh(V, F);
	viewer.data(mesh_id).set_colors(Eigen::RowVector3d(0.8, 0.3, 0.1));
	viewer.data(mesh_id).show_lines = false;
	viewer.launch_init(true, false, "vector heat method");
	load_my_shader(viewer, mesh_id, "../src/parallel.frag");

	// Initial vector field
	viewer.append_mesh();
	int X_id = viewer.data_list.back().id;
	viewer.data(X_id).line_width = 4;
	viewer.data(X_id).point_size = 2*viewer.data(X_id).line_width;
	Eigen::RowVector3d X_color(0.3, 0.1, 0.5);
	Eigen::VectorXcd X;
	std::vector<int> X_ind_in_data(V.rows(), -1);
	Eigen::VectorXi Omega;

	// Final field
	Eigen::VectorXcd barX;
	Eigen::VectorXd D;
	Eigen::VectorXi cluster;
	viewer.append_mesh();
	int res_id = viewer.data_list.back().id;
	viewer.data(res_id).line_width = 2.8;
	viewer.data(res_id).point_size = 2*viewer.data(res_id).line_width;
	Eigen::MatrixX3d res(V.rows(), 3), res_col;
	const auto barX_to_res = [&]()->void {
		Eigen::RowVector3d tmp_vec;
		for(int i = 0; i < V.rows(); ++i) {
			igl::complex_to_vector(V, hvm_data, i, barX(i), tmp_vec);
			res.row(i) = tmp_vec;
		}
	};

	// Some information on clicks
	int fid;
	Eigen::Vector3f bc;
	const auto get_vertex = [&]()->int {
		double x = viewer.current_mouse_x;
		double y = viewer.core().viewport(3) - viewer.current_mouse_y;
		viewer.selected_data_index = mesh_id;
		if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core().view,
				viewer.core().proj, viewer.core().viewport, V, F, fid, bc)) {
			const Eigen::RowVector3d pos = V.row(F(fid,0))*bc(0) + V.row(F(fid,1))*bc(1) + V.row(F(fid,2))*bc(2);
			int c = 0;
			Eigen::Vector3d(
				(V.row(F(fid,0))-pos).squaredNorm(),
				(V.row(F(fid,1))-pos).squaredNorm(),
				(V.row(F(fid,2))-pos).squaredNorm()).minCoeff(&c);
			return F(fid, c);
		}
		return -1;
	};
	const auto get_pos = [&](Eigen::RowVector3d &pos)->bool{
		double x = viewer.current_mouse_x;
		double y = viewer.core().viewport(3) - viewer.current_mouse_y;
		viewer.selected_data_index = mesh_id;
		if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core().view,
				viewer.core().proj, viewer.core().viewport, V, F, fid, bc)) {
			pos = V.row(F(fid,0))*bc(0) + V.row(F(fid,1))*bc(1) + V.row(F(fid,2))*bc(2);
			return true;
		}
		return false;
	};

	int clicked_vertex = -1;
	viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int but, int modif)->bool {
		if(but != GLFW_MOUSE_BUTTON_LEFT || (modif & GLFW_MOD_SHIFT)) {
			clicked_vertex = -1;
			return false;
		}
		clicked_vertex = get_vertex();
		return clicked_vertex >= 0;
	};

	viewer.callback_mouse_move = [&clicked_vertex](igl::opengl::glfw::Viewer& viewer, int, int)->bool {
		return clicked_vertex >= 0;
	};

	Eigen::RowVector3d pos;
	viewer.callback_mouse_up = [&](igl::opengl::glfw::Viewer& viewer, int but, int modif)->bool {
		if(but != GLFW_MOUSE_BUTTON_LEFT || (modif & GLFW_MOD_SHIFT)) return false;
		if(clicked_vertex >= 0 && get_pos(pos)) {
			Eigen::RowVector3d vec = pos - V.row(clicked_vertex);
			Eigen::RowVector3d v = V.row(clicked_vertex) + 0.03*avg_l*hvm_data.e0.row(clicked_vertex).cross(hvm_data.e1.row(clicked_vertex));
			if(mode == COMPUTE_H_R || mode == COMPUTE_LOG) {
				Omega.resize(1);
				Omega(0) = clicked_vertex;
				viewer.data(X_id).clear();
				viewer.data(X_id).add_points(mode == COMPUTE_H_R ? v : pos, X_color);
				return false;
			}
			if(X_ind_in_data[clicked_vertex] < 0) {
				X_ind_in_data[clicked_vertex] = viewer.data(X_id).points.rows();
				viewer.data(X_id).add_points(v, X_color);
				Omega.conservativeResize(Omega.size()+1);
				Omega(Omega.size()-1) = clicked_vertex;
			}
			if(mode != COMPUTE_PARALLEL_TRANSPORT) return false;
			std::complex<double> c;
			igl::vector_to_complex(hvm_data, clicked_vertex, vec, c);
			igl::complex_to_vector(V, hvm_data, clicked_vertex, c, vec);
			Eigen::RowVector3d dest = v + vec;
			int ind = X_ind_in_data[clicked_vertex];
			if(ind >= X.size()) {
				X.conservativeResize(Omega.size());
				viewer.data(X_id).add_edges(v, dest, X_color);
			} else {
				for(int i = 0; i < 3; ++i) viewer.data(X_id).lines(ind, 3+i) = dest[i];
				viewer.data(X_id).dirty |= igl::opengl::MeshGL::DIRTY_OVERLAY_LINES;
			}
			X(ind) = c;
		}
		clicked_vertex = -1;
		return false;
	};

	std::cout << "Usage:\n"
		"  [click]     Click on mesh then drag an release the button to add a new vector in X\n"
		"              If shift is pressed, then a rotation will be done instead of adding a vector\n"
		"  [BACKSPACE] When mouse button is pressed, remove vector at selected vertex\n"
		"  N,n         Switch mode (parallel transport/log map/...)\n"
		"  [SPACE]     Compute the parallel transport/log map/...\n"
		"  -/+         Decrease/increase t by factor of 10.0\n"
		"  D,d         Toggle using intrinsic Delaunay discrete differential operators\n"
		"  S,s         Take a screenshot\n"
		"\n";

	// For screenshots
	int screen_number = 0;
	std::stringstream file_name;
	Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R, G, B, A;

	viewer.callback_key_down = [&](igl::opengl::glfw::Viewer&, unsigned int key, int mod)->bool {
		switch(key) {
		case GLFW_KEY_BACKSPACE:
			if(clicked_vertex >= 0) {
				if(X_ind_in_data[clicked_vertex] >= 0) {
					int ind = X_ind_in_data[clicked_vertex], last = Omega(Omega.size()-1);
					Omega(ind) = last;
					if(X.size() > 0) X(ind) = X(X.size()-1);
					X_ind_in_data[last] = ind;
					Omega.conservativeResize(Omega.size()-1);
					X.conservativeResize(Omega.size());
					viewer.data(X_id).lines.row(ind) = viewer.data(X_id).lines.row(Omega.size());
					viewer.data(X_id).lines.conservativeResize(Omega.size(), Eigen::NoChange);
					viewer.data(X_id).points.row(ind) = viewer.data(X_id).points.row(Omega.size());
					viewer.data(X_id).points.conservativeResize(Omega.size(), Eigen::NoChange);
					viewer.data(X_id).dirty |= igl::opengl::MeshGL::DIRTY_OVERLAY_LINES | igl::opengl::MeshGL::DIRTY_OVERLAY_POINTS;
					X_ind_in_data[clicked_vertex] = -1;
				}
				clicked_vertex = -1;
			}
			break;
		case GLFW_KEY_SPACE:
			viewer.data(res_id).clear();
			viewer.data(mesh_id).set_colors(Eigen::RowVector3d(0.8, 0.2, 0.1));
			if(Omega.size() == 0) break;
			switch(mode) {
			case COMPUTE_PARALLEL_TRANSPORT:
				igl::heat_vector_solve(hvm_data, Omega, X, barX);
				barX_to_res();
				igl::heat_geodesics_solve(geod_data, Omega, D);
				res_col = (D / D.maxCoeff()).replicate(1, 3);
				viewer.data(mesh_id).set_colors(res_col);
				res_col.array() *= 0.66;
				res_col.array() += 0.15;
				viewer.data(res_id).add_edges(V, V + res, res_col);
				viewer.data(res_id).set_points(V, res_col);
				break;
			case COMPUTE_VORONOI:
				igl::heat_voronoi_solve(hvm_data, Omega, cluster);
				res_col = Eigen::MatrixX3d::Random(Omega.size(), 3);
				res_col.array() *= 0.5;
				res_col.array() += 0.5;
				for(int i = 0; i < V.rows(); ++i)
					res.row(i) = cluster(i) >= 0 ? res_col.row(cluster(i)) : Eigen::RowVector3d(0.03, 0.03, 0.03);
				viewer.data(mesh_id).set_colors(res);
				break;
			case COMPUTE_H_R:
				igl::heat_R_solve(hvm_data, Omega(0), barX);
				barX *= 0.6*avg_l;
				barX_to_res();
				igl::heat_geodesics_solve(geod_data, Omega, D);
				res_col = (D / D.maxCoeff()).replicate(1, 3);
				viewer.data(mesh_id).set_colors(res_col);
				res_col.array() *= 0.5;
				res_col.array() += 0.3;
				res_col.col(0) *= 0.4;
				res_col.col(2) *= 0.2;
				viewer.data(res_id).add_edges(V, V + res, res_col);
				igl::heat_vector_solve(hvm_data, Omega, (Eigen::VectorXcd(1) << 1.).finished(), barX);
				barX *= 0.6*avg_l;
				barX_to_res();
				res_col = (D / D.maxCoeff()).replicate(1, 3);
				res_col.array() *= 0.45;
				res_col.array() += 0.5;
				res_col.col(0) *= 0.2;
				res_col.col(1) *= 0.3;
				viewer.data(res_id).add_edges(V, V + res, res_col);
				break;
			case COMPUTE_LOG:
				igl::heat_log_solve(hvm_data, F.row(fid), bc, barX);
				barX /= barX.cwiseAbs().colwise().maxCoeff().coeff(0);
				res_col.resize(V.rows(), 3);
				for(int i = 0; i < V.rows(); ++i)
					res_col.row(i) = Eigen::RowVector3d(barX(i).real(), barX(i).imag(), std::abs(barX(i)));
				viewer.data(mesh_id).set_colors(res_col);
				break;
			}
			break;
		case 'D':
		case 'd':
			hvm_data.use_intrinsic_delaunay = !hvm_data.use_intrinsic_delaunay;
			geod_data.use_intrinsic_delaunay = hvm_data.use_intrinsic_delaunay;
			if(!hvm_data.use_intrinsic_delaunay) std::cout << "not ";
			std::cout << "using intrinsic delaunay..." << std::endl;
			precompute();
			break;
		case GLFW_KEY_KP_SUBTRACT:
		case GLFW_KEY_KP_ADD:
			t *= (key==GLFW_KEY_KP_ADD ? 10.0 : 0.1);
			precompute();
			std::cout << "t: " << t << std::endl;
			break;
		case 'S':
		case 's':
			R.resize((int) viewer.core().viewport(2), (int) viewer.core().viewport(3));
			G.resize((int) viewer.core().viewport(2), (int) viewer.core().viewport(3));
			B.resize((int) viewer.core().viewport(2), (int) viewer.core().viewport(3));
			A.resize((int) viewer.core().viewport(2), (int) viewer.core().viewport(3));
			igl::my_draw_buffer(viewer, false, R, G, B, A);
			file_name.str("");
			file_name << "../screen_" << (screen_number++) << ".png";
			std::cout << file_name.str() << std::endl;
			igl::png::writePNG(R, G, B, A, file_name.str());
			break;
		case 'N':
		case 'n':
			mode = (mode+1) % COMPUTE_NUMBER;
			viewer.data(mesh_id).set_colors(Eigen::RowVector3d(0.8, 0.2, 0.1));
			viewer.data(X_id).clear();
			viewer.data(res_id).clear();
			for(int i = 0; i < Omega.size(); ++i) X_ind_in_data[Omega(i)] = -1;
			Omega.resize(0);
			X.resize(0);
			switch(mode) {
			case COMPUTE_PARALLEL_TRANSPORT:
				std::cout << "Mode: Parallel transport" << std::endl;
				load_my_shader(viewer, mesh_id, "../src/parallel.frag");
				break;
			case COMPUTE_VORONOI:
				std::cout << "Mode: Voronoi" << std::endl;
				load_standard_shader(viewer, mesh_id);
				break;
			case COMPUTE_H_R:
				std::cout << "Mode: H and R fields" << std::endl;
				load_my_shader(viewer, mesh_id, "../src/parallel.frag");
				break;
			case COMPUTE_LOG:
				std::cout << "Mode: Log" << std::endl;
				load_my_shader(viewer, mesh_id, "../src/log.frag");
				break;
			}
		default:
			return false;
		}
		return true;
	};

	viewer.launch_rendering(true);
	viewer.launch_shut();

}