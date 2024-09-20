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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(rheo/granular,PairRHEOGranular)
// clang-format on
#else

#ifndef LMP_PAIR_RHEO_GRANULAR_H
#define LMP_PAIR_RHEO_GRANULAR_H

#include "pair.h"

namespace LAMMPS_NS {

class PairRHEOGranular : public Pair {
 public:
  PairRHEOGranular(class LAMMPS *);
  ~PairRHEOGranular() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void setup() override;
  double init_one(int, int) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 protected:
  double h, hsq, rho0, csq, av, rho_damp;        // From fix RHEO
  double **sdiv;
  int nmax_store;
  int interface_flag;

  void allocate();

  class ComputeRHEOKernel *compute_kernel;
  class ComputeRHEOGrad *compute_grad;
  class ComputeRHEOInterface *compute_interface;
  class FixRHEO *fix_rheo;
  class FixRHEOStress *fix_stress;
};

}    // namespace LAMMPS_NS

#endif
#endif
