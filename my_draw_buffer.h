#ifndef MY_DRAW_BUFFER_H
#define MY_DRAW_BUFFER_H

#include <igl/opengl/glfw/Viewer.h>

namespace igl {

void my_draw_buffer(igl::opengl::glfw::Viewer &viewer,
	bool update_matrices,
	Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> &R,
	Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> &G,
	Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> &B,
	Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> &A);

}

#endif