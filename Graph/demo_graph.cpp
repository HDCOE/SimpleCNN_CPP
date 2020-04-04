// Demo of vector plot.
// Compile it with:
//   g++ -o example-vector example-vector.cc -lboost_iostreams -lboost_system -lboost_filesystem

#include <vector>
#include <cmath>
#include <boost/tuple/tuple.hpp>

#include "draw_graph.h"
//#include "gnumake.h"
int main() {

	float * x = new float[20];
	float * y = new float[20];

	for (int i = 0; i < 20; ++i)
	{
		y[i] = 20 - i;
		x[i] = i;
	}
	drawgraph(x,y,20);
	// Data will be sent via a temporary file.  These are erased when you call
	// gp.clearTmpfiles() or when gp goes out of scope.  If you pass a filename
	// (e.g. "gp.file1d(pts, 'mydata.dat')"), then the named file will be created
	// and won't be deleted (this is useful when creating a script).
}