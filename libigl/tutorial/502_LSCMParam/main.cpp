#include <igl/boundary_loop.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

#include <igl/lscm.h>


Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd V_uv;


int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  // Load a mesh in OFF format
  igl::readOBJ("/home/chern/Projects/generate_caricature/keyu_test/human_47/face_22.obj", V, F);

  // Fix two points on the boundary
  VectorXi bnd,b(2,1);
  igl::boundary_loop(F,bnd);
  b(0) = bnd(0);
  b(1) = bnd(round(bnd.size()/2));
  MatrixXd bc(2,2);
  bc<<0,0,1,0;

  // LSCM parametrization
  igl::lscm(V,F,b,bc,V_uv);

  // Scale the uv
  V_uv *= 5;

  igl::writeOBJ("/home/chern/Projects/generate_caricature/UV.obj",V_uv,F);
}
