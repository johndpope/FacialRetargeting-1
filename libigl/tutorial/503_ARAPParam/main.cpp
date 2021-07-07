#include <igl/arap.h>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd V_uv;
Eigen::MatrixXd initial_guess;

bool show_uv = false;


int main(int argc, char *argv[])
{
  using namespace std;
  // Load a mesh in OFF format
  igl::readOBJ("/home/chern/Projects/generate_caricature/keyu_test/human_47/face_22.obj", V, F);

  // Compute the initial solution for ARAP (harmonic parametrization)
  Eigen::VectorXi bnd;
  igl::boundary_loop(F,bnd);
  Eigen::MatrixXd bnd_uv;
  igl::map_vertices_to_circle(V,bnd,bnd_uv);

  igl::harmonic(V,F,bnd,bnd_uv,1,initial_guess);

  // Add dynamic regularization to avoid to specify boundary conditions
  igl::ARAPData arap_data;
  arap_data.with_dynamics = true;
  Eigen::VectorXi b  = Eigen::VectorXi::Zero(0);
  Eigen::MatrixXd bc = Eigen::MatrixXd::Zero(0,0);

  // Initialize ARAP
  arap_data.max_iter = 100;
  // 2 means that we're going to *solve* in 2d
  arap_precomputation(V,F,2,b,arap_data);


  // Solve arap using the harmonic map as initial guess
  V_uv = initial_guess;

  arap_solve(bc,arap_data,V_uv);


  // Scale UV to make the texture more clear
  V_uv *= 20;

  igl::writeOBJ("/home/chern/Projects/generate_caricature/UV.obj",V_uv,F);
}
