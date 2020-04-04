// Demo of vector plot.
// Compile it with:
//   g++ -o example-vector example-vector.cc -lboost_iostreams -lboost_system -lboost_filesystem

#include <vector>
#include <cmath>
#include <boost/tuple/tuple.hpp>

#include "gnuplot-iostream/gnuplot-iostream.h"
//#include "gnumake.h"
void drawgraph(float *x, float *y, int size)
{
	Gnuplot gp;
	std::vector<std::pair<float, float> > xy_pts_A;
	
	for(int i=0; i< size ; ++i) {
		xy_pts_A.push_back(std::make_pair(x[i], y[i]));
	 	std::cout << "x ["<< i << "]" << x[i] << " y " << y[i] << std::endl;
	}
	//gp << "set xrange [0:20]\nset yrange [0:20]\n";
	gp.clearTmpfiles();
	gp << "set style data linespoints \n";
	gp << "set title 'Model performance'\n set title font ',12' norotate\n"
	   << "set xlabel 'epoch'\n set ylabel 'loss' \n" ;
	
	gp << "plot" << gp.file1d(xy_pts_A) << "  lw 2 lc rgb 'forest-green' with linespoints title 'Error' " << std::endl;

	// Data will be sent via a temporary file.  These are erased when you call
	// gp.clearTmpfiles() or when gp goes out of scope.  If you pass a filename
	// (e.g. "gp.file1d(pts, 'mydata.dat')"), then the named file will be created
	// and won't be deleted (this is useful when creating a script).
}