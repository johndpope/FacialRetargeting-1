#include "util_3drotation_log_exp.h"
#include <iostream>
#include "rimd_reconstruction.h"
#include <queue>
#include <fstream>
#include <ostream>
#include <string>

int main(int argc, char *argv[])
{
    RIMD_Reconstruction rimd_r;

    // Example of computing rimd feature for a given model
    rimd_r.read_ref_mesh_from_file("shape_0.obj");
    rimd_r.read_anchor_points_id("one_anchor.txt");
    rimd_r.Preprocess();

    std::string def_name = "shape_29.obj";
    std::cout << def_name << std::endl;
    std::string data_name = "shape_29.dat";
    std::string reconstruct_name = "shape_29_reconstruct.obj";
    rimd_r.read_defor_mesh_from_file(def_name);
    rimd_r.compute_RIMD_of_ref_to_defor();
    rimd_r.InterlateRIMD(1.0, data_name);
    //rimd_r.Reconstruction();
    //TriMesh mesh;
    //rimd_r.GetReconstructionMesh(mesh);
    //OpenMesh::IO::write_mesh(mesh, reconstruct_name);
    //std::cout << reconstruct_name << std::endl;
}
