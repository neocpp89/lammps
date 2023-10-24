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
<<<<<<<< HEAD:src/EXTRA-COMPUTE/compute_rattlers_atom.h
ComputeStyle(rattlers/atom,ComputeRattlersAtom);
// clang-format on
#else

#ifndef LMP_COMPUTE_RATTLERS_ATOM_H
#define LMP_COMPUTE_RATTLERS_ATOM_H
========
ComputeStyle(RHEO/STRESS,ComputeRHEOStress)
// clang-format on
#else

#ifndef LMP_COMPUTE_RHEO_STRESS_H
#define LMP_COMPUTE_RHEO_STRESS_H
>>>>>>>> 155c4c4581 (Granular constitutive update, not hooked up yet):src/RHEO/compute_rheo_stress.h

#include "compute.h"

namespace LAMMPS_NS {

<<<<<<<< HEAD:src/EXTRA-COMPUTE/compute_rattlers_atom.h
class ComputeRattlersAtom : public Compute {
 public:
  ComputeRattlersAtom(class LAMMPS *, int, char **);
  ~ComputeRattlersAtom() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
  double compute_scalar() override;
========
class ComputeRHEOStress : public Compute {
 public:
  ComputeRHEOStress(class LAMMPS *, int, char **);
  ~ComputeRHEOStress() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
>>>>>>>> 155c4c4581 (Granular constitutive update, not hooked up yet):src/RHEO/compute_rheo_stress.h
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  double memory_usage() override;

<<<<<<<< HEAD:src/EXTRA-COMPUTE/compute_rattlers_atom.h
 private:
  int cutstyle, ncontacts_rattler, max_tries, nmax, invoked_peratom;
  int *ncontacts;
  double *rattler;
  class NeighList *list;

========
  void update_one_material_point_stress(double *stress, const double *velocity_gradient, double density);

  double **stress;
  class FixRHEO *fix_rheo;

 private:
  int nmax_store;

  class NeighList *list;

  void grow_arrays(int);
>>>>>>>> 155c4c4581 (Granular constitutive update, not hooked up yet):src/RHEO/compute_rheo_stress.h
};

}    // namespace LAMMPS_NS

#endif
#endif
