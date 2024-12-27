/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(rheo/stress,FixRHEOStress);
// clang-format on
#else

#ifndef LMP_FIX_RHEO_STRESS_H
#define LMP_FIX_RHEO_STRESS_H

#include "fix.h"

namespace LAMMPS_NS {

class FixRHEOStress : public Fix {
 public:
  FixRHEOStress(class LAMMPS *, int, char **);
  ~FixRHEOStress() override;
  void post_constructor() override;
  int setmask() override;
  void init() override;
  void pre_force(int) override;
  void end_of_step() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

  void update_one_material_point_stress_elastic(double *stress, const double *velocity_gradient, double density, double dt, int dim);
  void update_one_material_point_stress(double *ptxxdev, double *rho_pressure, double *ptr_t0, double *pnup_tau, double *cauchy_stress, const double *velocity_gradient, double density, double dt, int dim);
  // void update_one_material_point_stress(double *stress, const double *velocity_gradient, double density, double dt, int dim);
  void one_element_test(void);

  class FixRHEO *fix_rheo;
  class FixStoreAtom *store_fix;

 private:
  // double **stress;
  char *id_fix;

  void compute_peratom(void);

  // Set by input file
  double RHO_CRITICAL;
  double E;
  double NU;
  double COHESION;
  double GRAINS_D;
  double GRAINS_RHO;
  double MU_S;
  double MU_2;
  double I_0;

  // Derived elastic parameters
  double G;
  double K;
  double LAMBDA;
};

}    // namespace LAMMPS_NS

#endif
#endif
