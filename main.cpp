#include <igl/read_triangle_mesh.h>
#include <igl/triangulated_grid.h>
#include <igl/heat_geodesics.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/avg_edge_length.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/create_shader_program.h>
#include <igl/opengl/destroy_shader_program.h>

#include <iostream>
#include <fstream>

int main(int argc, char *argv[]) {
	// Create the peak height field
	Eigen::MatrixXi F;
	Eigen::MatrixXd V;
	igl::read_triangle_mesh( argc>1?argv[1]: "../meshes/bunny.obj",V,F);

	// Precomputation
	igl::HeatGeodesicsData<double> data;
	double t = std::pow(igl::avg_edge_length(V,F),2);
	const auto precompute = [&]() {
		if(!igl::heat_geodesics_precompute(V,F,t,data)) {
			std::cerr << "Error: heat_geodesics_precompute failed." << std::endl;
			exit(EXIT_FAILURE);
		};
	};
	precompute();

	// Initialize white
	Eigen::MatrixXd C = Eigen::MatrixXd::Constant(V.rows(),3,1); // Color of vertices
	Eigen::VectorXd D(V.rows()); // Distances
	igl::opengl::glfw::Viewer viewer; // Viewer
	const auto update = [&]()->bool {
		int fid; // ID of the clicked face
		Eigen::Vector3f bc; // Barycentric coordinates of the click on the face

		// Cast a ray in the view direction starting from the mouse position
		double x = viewer.current_mouse_x;
		double y = viewer.core().viewport(3) - viewer.current_mouse_y;
		if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core().view,
				viewer.core().proj, viewer.core().viewport, V, F, fid, bc)) {
			Eigen::Vector3i f = F.row(fid); // Face
			const Eigen::RowVector3d m3 = // 3d position of hit
				V.row(f(0))*bc(0) + V.row(F(1))*bc(1) + V.row(F(2))*bc(2);
			int cid = 0; // Nearest corner ID
			Eigen::Vector3d(
				(V.row(f(0))-m3).squaredNorm(),
				(V.row(f(1))-m3).squaredNorm(),
				(V.row(f(2))-m3).squaredNorm()).minCoeff(&cid);
			const int vid = f(cid); // Nearest vertex ID
			igl::heat_geodesics_solve(data, (Eigen::VectorXi(1,1) << vid).finished(), D);
			viewer.data().set_colors((D/D.maxCoeff()).replicate(1,3));
			return true;
		}
		return false;
	};

	bool down_on_mesh = false;  // If the user clicked the mesh

	viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool {
		if(update()) {
			down_on_mesh = true;
			return true;
		}
		return false;
	};

	viewer.callback_mouse_move = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool {
		if(down_on_mesh) {
			update();
			return true;
		}
		return false;
	};

	viewer.callback_mouse_up = [&down_on_mesh](igl::opengl::glfw::Viewer& viewer, int, int)->bool {
		down_on_mesh = false;
		return false;
	};

	std::cout << "Usage:\n"
		"  [click]  Click on shape to pick new geodesic distance source\n"
		"  ,/.      Decrease/increase t by factor of 10.0\n"
		"  D,d      Toggle using intrinsic Delaunay discrete differential operatorsé\n"
		"\n";

  viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer&, unsigned int key, int mod)->bool {
	switch(key) {
	default:
		return false;
	case 'D':
	case 'd':
		data.use_intrinsic_delaunay = !data.use_intrinsic_delaunay;
		if(!data.use_intrinsic_delaunay) std::cout << "not ";
		std::cout << "using intrinsic delaunay..." << std::endl;
		precompute();
		update();
		break;
	case '.':
	case ',':
		t *= (key=='.' ? 10.0 : 0.1);
		precompute();
		update();
		std::cout << "t: " << t << std::endl;
		break;
	}
	return true;
  };

  // Show mesh
  viewer.data().set_mesh(V, F);
  viewer.data().set_colors(C);
  viewer.data().show_lines = false;
  viewer.launch_init(true, false, "vector heat method");

  viewer.data().meshgl.init();
  igl::opengl::destroy_shader_program(viewer.data().meshgl.shader_mesh);

  {
	std::ifstream f("../shader.vert");
	std::string mesh_vertex_shader_string((std::istreambuf_iterator<char>(f)), (std::istreambuf_iterator<char>()));
	f.close();
	f.open("../shader.frag");
	std::string mesh_fragment_shader_string((std::istreambuf_iterator<char>(f)), (std::istreambuf_iterator<char>()));
	f.close();
	
	igl::opengl::create_shader_program(
	  mesh_vertex_shader_string,
	  mesh_fragment_shader_string,
	  {},
	  viewer.data().meshgl.shader_mesh);
  }

  viewer.launch_rendering(true);
  viewer.launch_shut();

}