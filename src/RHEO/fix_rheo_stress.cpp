/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 https://www.lammps.org/, Sandia National Laboratories
 LAMMPS development team: developers@lammps.org

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Joel Clemmer (SNL)
----------------------------------------------------------------------- */

#include "fix_rheo_stress.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "fix_store_atom.h"
#include "group.h"
#include "error.h"
#include "modify.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

#if !defined(DIM)
#define DIM(x) (sizeof(x) / sizeof(x[0]))
#endif

struct material_property {
  bool required;
  const char *name;
  double value;
  bool set_by_input_file;
};

/* ---------------------------------------------------------------------- */

FixRHEOStress::FixRHEOStress(LAMMPS *lmp, int narg, char **arg) :
  id_compute(nullptr), id_fix(nullptr), stress_compute(nullptr), store_fix(nullptr), Fix(lmp, narg, arg)
{
  struct material_property properties[] = {
    { .required = true, .name = "rho_critical", .value = 1.0, .set_by_input_file = false, },
    { .required = true, .name = "E", .value = 1e4, .set_by_input_file = false, },
    { .required = true, .name = "nu", .value = 0.0, .set_by_input_file = false, },
    { .required = true, .name = "cohesion", .value = 0.0, .set_by_input_file = false, },
    { .required = true, .name = "grains_d", .value = 0.005, .set_by_input_file = false, },
    { .required = true, .name = "grains_rho", .value = 2450.0, .set_by_input_file = false, },
    { .required = true, .name = "mu_s", .value = 0.3819, .set_by_input_file = false, },
    { .required = true, .name = "mu_2", .value = 0.6435, .set_by_input_file = false, },
    { .required = true, .name = "I_0", .value = 0.278, .set_by_input_file = false, },
  };

  comm_forward = 6;

  int iarg = 3;
  while (iarg < narg) {
    bool found_property_name = false;
    const int extra_arg = 1;
    if((iarg + extra_arg) >= narg) {
      error->all(FLERR, "Not enough remaining arguments in fix/rheo/stress -- are all property names paired with values?");
    }
    for (size_t i = 0; i < DIM(properties); ++i) {
      struct material_property * const entry = &properties[i];
      if (strcmp(arg[iarg], entry->name) == 0) {
        entry->value = utils::numeric(FLERR, arg[iarg + 1], /*do_abort=*/true, lmp);
        entry->set_by_input_file = true;
        found_property_name = true;
        iarg += extra_arg;
        break;
      }
    }
    if (!found_property_name) {
      error->all(FLERR, "fix/rheo/stress does not know property named: {}", arg[iarg]);
    }
    ++iarg;
  }

  // Will include a trailing space, hopefully ok.
  property_list_for_compute = "";
  for (size_t i = 0; i < DIM(properties); ++i) {
    struct material_property * const entry = &properties[i];
    property_list_for_compute += fmt::format("{} {:.17g} ", entry->name, entry->value);
  }
}

/* ---------------------------------------------------------------------- */

FixRHEOStress::~FixRHEOStress()
{
  modify->delete_compute(id_compute);
  modify->delete_fix(id_fix);
}

/* ---------------------------------------------------------------------- */

void FixRHEOStress::post_constructor()
{
  id_fix = utils::strdup(std::string(id) + "_store");
  // store_fix = dynamic_cast<FixStoreAtom *>(modify->add_fix(fmt::format("{} {} STORE/ATOM d_pxx d_pyy d_pzz d_pxy d_pxz d_pyz", id_fix, group->names[igroup])));
  store_fix = dynamic_cast<FixStoreAtom *>(modify->add_fix(fmt::format("{} {} STORE/ATOM 6 0 1 1", id_fix, group->names[igroup])));
  array_atom = store_fix->astore;

  id_compute = utils::strdup(std::string(id) + "_compute");
  // stress_compute = modify->add_compute(fmt::format("{} {} stress/atom NULL ke pair bond", id_compute, group->names[igroup]));
  stress_compute = modify->add_compute(fmt::format("{} {} RHEO/STRESS {}", id_compute, group->names[igroup], property_list_for_compute));
}

/* ---------------------------------------------------------------------- */

int FixRHEOStress::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRHEOStress::init()
{
  stress_compute->addstep(update->ntimestep+1);
}

/* ---------------------------------------------------------------------- */

void FixRHEOStress::pre_force(int vflag)
{
  stress_compute->compute_peratom();

  // copy compute to fix property atom
  double **saved_stress = store_fix->astore;
  double **stress = stress_compute->array_atom;

  int ntotal = atom->nlocal+atom->nghost;
  for (int i = 0; i < ntotal; i++)
    for (int a = 0; a < 6; a++)
      saved_stress[i][a] = stress[i][a];

  stress_compute->addstep(update->ntimestep + 1);

  // add pre-force and forward to ghosts (not done in store/atom)
  comm->forward_comm(this);
}

/* ---------------------------------------------------------------------- */

void FixRHEOStress::end_of_step()
{
}

/* ---------------------------------------------------------------------- */

int FixRHEOStress::pack_forward_comm(int n, int *list, double *buf,
                                        int /*pbc_flag*/, int * /*pbc*/)
{
  int i, j, a, m;
  double **saved_stress = store_fix->astore;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    for (a = 0; a < 6; a++)
      buf[m++] = saved_stress[j][a];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixRHEOStress::unpack_forward_comm(int n, int first, double *buf)
{
  int i, a, m, last;
  double **saved_stress = store_fix->astore;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++)
    for (a = 0; a < 6; a++)
      saved_stress[i][a] = buf[m++];
}
