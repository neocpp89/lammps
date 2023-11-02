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
   Contributing authors:
   Joel Clemmer (SNL)
----------------------------------------------------------------------- */

#include "pair_rheo_granular.h"

#include "atom.h"
#include "comm.h"
#include "compute_rheo_kernel.h"
#include "compute_rheo_grad.h"
#include "domain.h"
#include "error.h"
#include "fix_rheo.h"
#include "fix_rheo_stress.h"
#include "force.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "update.h"
#include "utils.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace RHEO_NS;
using namespace MathExtra;

static constexpr double EPSILON = 1e-2;

/* ---------------------------------------------------------------------- */

PairRHEOGranular::PairRHEOGranular(LAMMPS *lmp) :
  Pair(lmp), compute_kernel(nullptr), compute_grad(nullptr),
  fix_rheo(nullptr), fix_stress(nullptr)
{
  restartinfo = 0;
  single_enable = 0;
  comm_reverse = 3;
  nmax_store = 0;
}

/* ---------------------------------------------------------------------- */

PairRHEOGranular::~PairRHEOGranular()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

void PairRHEOGranular::compute(int eflag, int vflag)
{
  int i, j, a, b, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, w, wp;
  double rhoi, rhoj, Voli, Volj;
  double *dWij, *dWji;
  double dx[3], sdotdw[3];

  int *ilist, *jlist, *numneigh, **firstneigh;
  double imass, jmass, rsq, r, rinv;

  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  int dim = domain->dimension;

  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  double *rho = atom->rho;
  double *mass = atom->mass;
  double **stress = fix_stress->array_atom;
  double *special_lj = force->special_lj;
  int *type = atom->type;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // allocate space and zero counter

  if (atom->nmax > nmax_store) {
    nmax_store = atom->nmax;
    memory->grow(sdiv, nmax_store, 3, "rheo/granular:sdiv");
  }
  size_t nbytes = 3 * (nmax_store) * sizeof(double);
  memset(&sdiv, 0, nbytes);

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    imass = mass[itype];
    rhoi = rho[i];
    Voli = imass / rho[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      dx[0] = xtmp - x[j][0];
      dx[1] = ytmp - x[j][1];
      dx[2] = ztmp - x[j][2];
      rsq = lensq3(dx);

      if (rsq < hsq) {
        r = sqrt(rsq);
        rinv = 1 / r;

        jmass = mass[jtype];
        rhoj = rho[j];
        Volj = jmass / rho[j];
        jtype = type[j];

        wp = compute_kernel->calc_dw(i, j, dx[0], dx[1], dx[2], r);
        dWij = compute_kernel->dWij;
        dWji = compute_kernel->dWji;

        // Add contributions to stress divergence

        sdotdw[0] = (stress[i][0] - stress[j][0]) * dWij[0];
        sdotdw[0] += (stress[i][3] - stress[j][3]) * dWij[1];
        sdotdw[0] += (stress[i][4] - stress[j][4]) * dWij[2];

        sdotdw[1] = (stress[i][3] - stress[j][3]) * dWij[0];
        sdotdw[1] += (stress[i][1] - stress[j][1]) * dWij[1];
        sdotdw[1] += (stress[i][5] - stress[j][5]) * dWij[2];

        sdotdw[2] = (stress[i][4] - stress[j][4]) * dWij[0];
        sdotdw[2] += (stress[i][5] - stress[j][5]) * dWij[1];
        sdotdw[2] += (stress[i][2] - stress[j][2]) * dWij[2];

        sdiv[i][0] += Volj * sdotdw[0];
        sdiv[i][1] += Volj * sdotdw[1];
        sdiv[i][2] += Volj * sdotdw[2];

        if (newton_pair || j < nlocal) {
          sdiv[j][0] -= Voli * sdotdw[0];
          sdiv[j][1] -= Voli * sdotdw[1];
          sdiv[j][2] -= Voli * sdotdw[2];
        }
      }
    }
  }

  if (newton_pair) comm->reverse_comm(this);

  // Add forces
  for (i = 0; i < atom->nlocal; i++) {
    f[i][0] += 0;
    f[i][1] += 0;
    f[i][2] += 0;
  }
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairRHEOGranular::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairRHEOGranular::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  h = utils::numeric(FLERR,arg[0],false,lmp);
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairRHEOGranular::coeff(int narg, char **arg)
{
  if (narg != 2)
    error->all(FLERR,"Incorrect number of args for pair_style rheo coefficients");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi,error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi,error);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = 0; j <= atom->ntypes; j++) {
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0)
    error->all(FLERR,"Incorrect args for pair rheo coefficients");
}

/* ----------------------------------------------------------------------
 setup specific to this pair style
 ------------------------------------------------------------------------- */

void PairRHEOGranular::setup()
{
  auto fixes = modify->get_fix_by_style("rheo");
  if (fixes.size() == 0) error->all(FLERR, "Need to define fix rheo to use pair rheo");
  fix_rheo = dynamic_cast<FixRHEO *>(fixes[0]);

  // Currently only allow one instance of fix rheo/pressure
  fixes = modify->get_fix_by_style("rheo/stress");
  if (fixes.size() == 0) error->all(FLERR, "Need to define fix rheo/stress to use pair rheo");
  fix_stress = dynamic_cast<FixRHEOStress *>(fixes[0]);

  compute_kernel = fix_rheo->compute_kernel;
  compute_grad = fix_rheo->compute_grad;

  if (h != fix_rheo->h)
    error->all(FLERR, "Pair rheo cutoff {} does not agree with fix rheo cutoff {}", h, fix_rheo->h);

  hsq = h * h;
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairRHEOGranular::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
      error->all(FLERR,"All pair rheo coeffs are not set");
  }

  return h;
}

/* ---------------------------------------------------------------------- */

int PairRHEOGranular::pack_reverse_comm(int n, int first, double *buf)
{
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = sdiv[i][0];
    buf[m++] = sdiv[i][1];
    buf[m++] = sdiv[i][2];
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void PairRHEOGranular::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    sdiv[j][0] += buf[m++];
    sdiv[j][1] += buf[m++];
    sdiv[j][2] += buf[m++];
  }
}
