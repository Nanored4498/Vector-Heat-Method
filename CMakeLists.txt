project(vector_heat)

cmake_minimum_required(VERSION 3.1)

set(CMAKE_BUILD_TYPE Release)
set(BASE_CXX_FLAGS "-Wall -Wextra -Werror")

add_subdirectory("/home/yoann/libigl" "/home/yoann/libigl/build")
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
# option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
# option(LIBIGL_USE_STATIC_LIBRARY     "Use libIGL as static librarie" ON)
# find_package(LIBIGL REQUIRED)

set(SRC
	main.cpp
	heat_vector.cpp
)

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME}
	igl::core
	igl::opengl
	igl::opengl_glfw 
)