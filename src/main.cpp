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

int main(int argc, char *argv[]) {

	// Load Mesh
	Eigen::MatrixXi F;
	Eigen::MatrixXd V;
	igl::read_triangle_mesh(argc>1?argv[1]: "../meshes/bunny.obj",V,F);

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

	// Show mesh
	igl::opengl::glfw::Viewer viewer;
	int mesh_id = viewer.data().id;
	viewer.data(mesh_id).set_mesh(V, F);
	viewer.data(mesh_id).set_colors(Eigen::RowVector3d(0.8, 0.3, 0.1));
	viewer.data(mesh_id).show_lines = false;
	viewer.launch_init(true, false, "vector heat method");

	// Initial vector field
	viewer.append_mesh();
	int X_id = viewer.data_list.back().id;
	viewer.data(X_id).line_width = 4;
	viewer.data(X_id).point_size = 2*viewer.data(X_id).line_width;
	Eigen::RowVector3d X_color(0.3, 0.1, 0.5);
	std::vector<std::complex<double>> X(V.rows(), 0);
	std::vector<int> X_ind_in_data(V.rows(), -1);
	Eigen::VectorXi Omega;

	// Final field
	Eigen::VectorXcd X_omega, barX;
	Eigen::VectorXd D;
	Eigen::RowVector3d tmp_vec;
	viewer.append_mesh();
	int res_id = viewer.data_list.back().id;
	viewer.data(res_id).line_width = 2.8;
	viewer.data(res_id).point_size = 2*viewer.data(res_id).line_width;
	Eigen::MatrixX3d res(V.rows(), 3), res_col;
	std::ofstream writer;

	// Some information on clicks
	const auto get_vertex = [&]()->int {
		double x = viewer.current_mouse_x;
		double y = viewer.core().viewport(3) - viewer.current_mouse_y;
		viewer.selected_data_index = mesh_id;
		int fid;
		Eigen::Vector3f bc;
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
		int fid;
		Eigen::Vector3f bc;
		if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core().view,
				viewer.core().proj, viewer.core().viewport, V, F, fid, bc)) {
			pos = V.row(F(fid,0))*bc(0) + V.row(F(fid,1))*bc(1) + V.row(F(fid,2))*bc(2);
			return true;
		}
		return false;
	};

	int clicked_vertex = -1;
	viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int but, int modif)->bool {
		if(but != GLFW_MOUSE_BUTTON_LEFT || (modif & GLFW_MOD_SHIFT)) return false;
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
			igl::vector_to_complex(hvm_data, clicked_vertex, vec, X[clicked_vertex]);
			igl::complex_to_vector(V, hvm_data, clicked_vertex, X[clicked_vertex], vec);
			Eigen::RowVector3d v = V.row(clicked_vertex) + 0.03*avg_l*hvm_data.e0.row(clicked_vertex).cross(hvm_data.e1.row(clicked_vertex));
			if(X_ind_in_data[clicked_vertex] < 0) {
				X_ind_in_data[clicked_vertex] = viewer.data(X_id).lines.rows();
				viewer.data(X_id).add_edges(v, v + vec, X_color);
				viewer.data(X_id).add_points(v, X_color);
				Omega.conservativeResize(Omega.size()+1);
				Omega(Omega.size()-1) = clicked_vertex;
			} else {
				int ind = X_ind_in_data[clicked_vertex];
				Eigen::RowVector3d dest = v + vec;
				for(int i = 0; i < 3; ++i) viewer.data(X_id).lines(ind, 3+i) = dest[i];
				viewer.data(X_id).dirty |= igl::opengl::MeshGL::DIRTY_OVERLAY_LINES;
			}
		}
		clicked_vertex = -1;
		return false;
	};

	std::cout << "Usage:\n"
		"  [click]     Click on mesh then drag an release the button to add a new vector in X\n"
		"              If shift is pressed, then a rotation will be done instead of adding a vector\n"
		"  [BACKSPACE] When mouse button is pressed, remove vector at selected vertex\n"
		"  [SPACE]     Compute the parallel transport\n"
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
					X_ind_in_data[last] = ind;
					Omega.conservativeResize(Omega.size()-1);
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
			X_omega.resize(Omega.size(), 1);
			for(int i = 0; i < Omega.size(); ++i) X_omega(i) = X[Omega(i)];
			igl::heat_log_solve(hvm_data, Omega(0), barX);
			// igl::heat_vector_solve(hvm_data, Omega, X_omega, barX);
			writer.open("../field.obj");
			for(int i = 0; i < V.rows(); ++i) {
				igl::complex_to_vector(V, hvm_data, i, barX(i), tmp_vec);
				writer << "v " << tmp_vec(0) << " " << tmp_vec(1) << " " << tmp_vec(2) << "\n";
				res.row(i) = tmp_vec;
			}
			writer.close();
			igl::heat_geodesics_solve(geod_data, Omega, D);
			res_col = (D / D.maxCoeff()).replicate(1, 3);
			viewer.data(mesh_id).set_colors(res_col);
			res_col.array() *= 0.66;
			res_col.array() += 0.15;
			viewer.data(res_id).add_edges(V, V + res, res_col);
			viewer.data(res_id).set_points(V, res_col);

			// igl::heat_voronoi_solve(hvm_data, Omega, D);
			// res_col = Eigen::MatrixX3d::Random(Omega.size(), 3);
			// for(int i = 0; i < V.rows(); ++i)
			// 	res.row(i) = res_col.row(std::max(0, std::min(int(Omega.size())-1, int(0.5+D(i)))));
			// viewer.data(mesh_id).set_colors(res);
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
		default:
			return false;
		}
		return true;
	};

	std::ifstream f("../src/shader.vert");
	std::string mesh_vertex_shader_string((std::istreambuf_iterator<char>(f)), (std::istreambuf_iterator<char>()));
	f.close();
	f.open("../src/shader.frag");
	std::string mesh_fragment_shader_string((std::istreambuf_iterator<char>(f)), (std::istreambuf_iterator<char>()));
	f.close();
	viewer.data(mesh_id).meshgl.init();
	igl::opengl::destroy_shader_program(viewer.data(mesh_id).meshgl.shader_mesh);
	igl::opengl::create_shader_program(
		mesh_vertex_shader_string,
		mesh_fragment_shader_string,
		{},
		viewer.data(mesh_id).meshgl.shader_mesh);

	viewer.launch_rendering(true);
	viewer.launch_shut();

}