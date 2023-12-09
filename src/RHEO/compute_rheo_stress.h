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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(RHEO/STRESS,ComputeRHEOStress)
// clang-format on
#else

#ifndef LMP_COMPUTE_RHEO_STRESS_H
#define LMP_COMPUTE_RHEO_STRESS_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeRHEOStress : public Compute {
 public:
  ComputeRHEOStress(class LAMMPS *, int, char **);
  ~ComputeRHEOStress() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  double memory_usage() override;

  void update_one_material_point_stress_elastic(double *stress, const double *velocity_gradient, double density, double dt, int dim);
  void update_one_material_point_stress(double *ptxxdev, double *rho_pressure, double *ptr_t0, double *pnup_tau, double *cauchy_stress, const double *velocity_gradient, double density, double dt, int dim);
  // void update_one_material_point_stress(double *stress, const double *velocity_gradient, double density, double dt, int dim);
  void one_element_test(void);

  double **stress;
  class FixRHEO *fix_rheo;

 private:
  int nmax_store;

  class NeighList *list;

  void grow_arrays(int);
};

}    // namespace LAMMPS_NS

#endif
#endif
