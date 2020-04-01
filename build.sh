if [ ! -d "libigl" ]; then
	echo "There are no subdirectory 'libigl'."
	echo "LIBIGL is required. If you don't have it already intalled somewhere you need to clone it."
	echo "Do you want to clone the repository `https://github.com/libigl/libigl.git`? (yY/n) "
	read choice
	if [ $choice = "y" ] || [ $choice = "Y" ]; then
		echo "Cloning libigl"
		git clone https://github.com/libigl/libigl.git
	fi
fi

if [ ! -d "build" ]; then
	echo "Creating build directory"
	mkdir build
fi

cd build
cmake .. $@
make -j4