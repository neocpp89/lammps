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
   Joel Clemmer (SNL), Thomas O'Connor (CMU), Eric Palermo (CMU)
----------------------------------------------------------------------- */

#include "compute_rheo_stress.h"

#include "atom.h"
#include "comm.h"
#include "compute_rheo_grad.h"
#include "domain.h"
#include "error.h"
#include "fix_rheo.h"
#include "force.h"
#include "memory.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace RHEO_NS;

/* ---------------------------------------------------------------------- */

ComputeRHEOStress::ComputeRHEOStress(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), list(nullptr), stress(nullptr), fix_rheo(nullptr)
{
  // TODO: args for material params
  // if (narg < 4) error->all(FLERR,"Illegal compute rheo/grad command");

  // for (int iarg = 3; iarg < narg; iarg++) {
  //   if (strcmp(arg[iarg],"velocity") == 0) velocity_flag = 1;
  //   else if (strcmp(arg[iarg],"rho") == 0) rho_flag = 1;
  //   else if (strcmp(arg[iarg],"temperature") == 0) temperature_flag = 1;
  //   else if (strcmp(arg[iarg],"viscosity") == 0) eta_flag = 1;
  //   else error->all(FLERR, "Illegal compute rheo/grad command, {}", arg[iarg]);
  // }

  size_peratom_cols = 6;
  peratom_flag = 1;
  comm_forward = 6;
  comm_reverse = 6;

  nmax_store = 0;
  grow_arrays(atom->nmax);

}

/* ---------------------------------------------------------------------- */

ComputeRHEOStress::~ComputeRHEOStress()
{
  memory->destroy(stress);
}

/* ---------------------------------------------------------------------- */

static void set_material_params(void);
void ComputeRHEOStress::init()
{
  set_material_params();
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOStress::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

static double negative_root(double a, double b, double c)
{
    double x;
    if (b > 0) {
        x = (-b - sqrt(b*b - 4*a*c)) / (2*a);
    } else {
        x = (2*c) / (-b + sqrt(b*b - 4*a*c));
    }
    return x;
}

enum {
  VoigtXX = 0,
  VoigtYY = 1,
  VoigtZZ = 2,
  VoigtXY = 3,
  VoigtXZ = 4,
  VoigtYZ = 5,
};

enum {
  Full3XX = 0,
  Full3XY = 1,
  Full3XZ = 2,
  Full3YX = 3,
  Full3YY = 4,
  Full3YZ = 5,
  Full3ZX = 6,
  Full3ZY = 7,
  Full3ZZ = 8,
};

enum {
  Full2XX = 0,
  Full2XY = 1,
  Full2YX = 2,
  Full2YY = 3,
};

static void full_from_voigt(double *full, const double *voigt)
{
  full[Full3XX] = voigt[VoigtXX];
  full[Full3XY] = voigt[VoigtXY];
  full[Full3XZ] = voigt[VoigtXZ];
  full[Full3YX] = voigt[VoigtXY];
  full[Full3YY] = voigt[VoigtYY];
  full[Full3YZ] = voigt[VoigtYZ];
  full[Full3ZX] = voigt[VoigtXZ];
  full[Full3ZY] = voigt[VoigtYZ];
  full[Full3ZZ] = voigt[VoigtZZ];
}

static void voigt_from_sym_full(double *voigt, const double *full)
{
  voigt[VoigtXX] = full[Full3XX];
  voigt[VoigtYY] = full[Full3YY];
  voigt[VoigtZZ] = full[Full3ZZ];
  voigt[VoigtXY] = full[Full3XY];
  voigt[VoigtXZ] = full[Full3XZ];
  voigt[VoigtYZ] = full[Full3YZ];
}

static void skw_part(double *skw_A, const double *A, int dim)
{
  if (dim == 3) {
    skw_A[Full3XX] = 0.0;
    skw_A[Full3YY] = 0.0;
    skw_A[Full3ZZ] = 0.0;

    skw_A[Full3XY] = 0.5 * (A[Full3XY] - A[Full3YX]);
    skw_A[Full3YX] = -skw_A[Full3XY];
    skw_A[Full3XZ] = 0.5 * (A[Full3XZ] - A[Full3ZX]);
    skw_A[Full3ZX] = -skw_A[Full3XZ];
    skw_A[Full3YZ] = 0.5 * (A[Full3YZ] - A[Full3ZY]);
    skw_A[Full3ZY] = -skw_A[Full3YZ];
  } else if (dim == 2) {
    // Always expands into a full 3x3 tensor, not a typo.
    skw_A[Full3XX] = 0.0;
    skw_A[Full3YY] = 0.0;

    skw_A[Full3XY] = 0.5 * (A[Full2XY] - A[Full2YX]);
    skw_A[Full3YX] = -skw_A[Full3XY];
  }
}

static void sym_part(double *sym_A, const double *A, int dim)
{
  if (dim == 3) {
    sym_A[Full3XX] = A[Full3XX];
    sym_A[Full3YY] = A[Full3YY];
    sym_A[Full3ZZ] = A[Full3ZZ];

    sym_A[Full3XY] = 0.5 * (A[Full3XY] + A[Full3YX]);
    sym_A[Full3YX] = sym_A[Full3XY];
    sym_A[Full3XZ] = 0.5 * (A[Full3XZ] + A[Full3ZX]);
    sym_A[Full3ZX] = sym_A[Full3XZ];
    sym_A[Full3YZ] = 0.5 * (A[Full3YZ] + A[Full3ZY]);
    sym_A[Full3ZY] = sym_A[Full3YZ];
  } else if (dim == 2) {
    // Always expands into a full 3x3 tensor, not a typo.
    sym_A[Full3XX] = A[Full2XX];
    sym_A[Full3YY] = A[Full2YY];

    sym_A[Full3XY] = 0.5 * (A[Full2XY] + A[Full2YX]);
    sym_A[Full3YX] = sym_A[Full3XY];
  }
}

static void scale(double *A, double s)
{
  for (size_t i = 0; i < 9; ++i) {
    A[i] *= s;
  }
}

static void accumulate(double *A, const double *B)
{
  for (size_t i = 0; i < 9; ++i) {
    A[i] += B[i];
  }
}

static double trace(const double *A)
{
  return A[Full3XX] + A[Full3YY] + A[Full3ZZ];
}

static void copy(double *A, const double *B)
{
  for (size_t i = 0; i < 9; ++i) {
    A[i] = B[i];
  }
}

static void deviator(double *A)
{
  const double one_third_trA = trace(A) / 3.0;
  A[Full3XX] -= one_third_trA;
  A[Full3YY] -= one_third_trA;
  A[Full3ZZ] -= one_third_trA;
}

static double frobenius_norm(double *A)
{
  double s = 0;
  for (size_t i = 0; i < 9; ++i) {
    s += (A[i] * A[i]);
  }
  return sqrt(s);
}

static void identity(double *A)
{
  for (size_t i = 0; i < 9; ++i) {
    A[i] = 0;
  }
  A[Full3XX] = 1.0;
  A[Full3YY] = 1.0;
  A[Full3ZZ] = 1.0;
}

static void multiply(double *C, const double *A, const double *B)
{
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      C[(3 * i) + j] = 0;
      for (size_t k = 0; k < 3; ++k) {
        C[(3 * i) + j] += A[(3 * i) + k] * B[(3 * k) + j];
      }
    }
  }
}

// Material parameters (to be set by fix arguments..).
const static double RHO_CRITICAL = 1500.0;
const static double E = 1e3;
const static double NU = 0.3;
const static double COHESION = 0.0;
const static double GRAINS_D = 0.003;
const static double GRAINS_RHO = 2450.0;
const static double MU_S = 0.3819;
const static double MU_2 = 0.6435;
const static double I_0 = 0.278;

// Derived elastic parameters
// static double G = 0;
// static double K = 0;
// static double LAMBDA = 0;
static const double  G = E / (2.0 * (1.0 + NU));
static const double  K = E / (3.0 * (1.0 - 2*NU));
static const double  LAMBDA = K - 2.0 * G / 3.0;

static void set_material_params(void)
{
  // G = E / (2.0 * (1.0 + NU));
  // K = E / (3.0 * (1.0 - 2*NU));
  // LAMBDA = K - 2.0 * G / 3.0;
}

void ComputeRHEOStress::update_one_material_point_stress_elastic(double *cauchy_stress,
    const double *velocity_gradient, double density)
{
    const int dim = domain->dimension;
    const double dt = update->dt;

    // Assume velocity gradient is laid out like
    //   Lxx, Lxy, Lxz,  Lyx, Lyy, Lyz,  Lzx, Lzy, Lzz
    const double *L = velocity_gradient;


    // Assume stress is laid out in voigt form like
    //   Txx, Tyy, Tzz, Txy, Txz, Tyz
    // then expand.
    double T[9] = {0};
    full_from_voigt(T, cauchy_stress);

    double D[9] = {0};
    double W[9] = {0};

    skw_part(W, L, dim);
    sym_part(D, L, dim);

    /* trial elastic increment using jaumann rate */
    double tmp[9] = {0};
    identity(tmp);
    scale(tmp, LAMBDA * trace(D));

    double jaumann_stress_increment[9] = {0};
    copy(jaumann_stress_increment, D);
    scale(jaumann_stress_increment, 2.0 * G);
    accumulate(jaumann_stress_increment, tmp);

    multiply(tmp, W, T);
    accumulate(jaumann_stress_increment, tmp);

    multiply(tmp, T, W);
    scale(tmp, -1.0);
    accumulate(jaumann_stress_increment, tmp);

    /* trial stress tensor */
    double T_tr[9] = {0};
    copy(T_tr, jaumann_stress_increment);
    scale(T_tr, dt);
    accumulate(T_tr, T);

    voigt_from_sym_full(cauchy_stress, T_tr);
}

void ComputeRHEOStress::update_one_material_point_stress(double *cauchy_stress,
    const double *velocity_gradient, double density)
{
    const int dim = domain->dimension;
    const double dt = update->dt;

    // Assume velocity gradient is laid out like
    //   Lxx, Lxy, Lxz,  Lyx, Lyy, Lyz,  Lzx, Lzy, Lzz
    const double *L = velocity_gradient;


    // Assume stress is laid out in voigt form like
    //   Txx, Tyy, Tzz, Txy, Txz, Tyz
    // then expand.
    double T[9] = {0};
    full_from_voigt(T, cauchy_stress);

    double D[9] = {0};
    double W[9] = {0};

    skw_part(W, L, dim);
    sym_part(D, L, dim);

    /* trial elastic increment using jaumann rate */
    double tmp[9] = {0};
    identity(tmp);
    scale(tmp, LAMBDA * trace(D));

    double jaumann_stress_increment[9] = {0};
    copy(jaumann_stress_increment, D);
    scale(jaumann_stress_increment, 2.0 * G);
    accumulate(jaumann_stress_increment, tmp);

    multiply(tmp, W, T);
    accumulate(jaumann_stress_increment, tmp);

    multiply(tmp, T, W);
    scale(tmp, -1.0);
    accumulate(jaumann_stress_increment, tmp);

    /* trial stress tensor */
    double T_tr[9] = {0};
    copy(T_tr, jaumann_stress_increment);
    scale(T_tr, dt);
    accumulate(T_tr, T);

    /* trial deviator values */
    double T0_tr[9] = {0};
    copy(T0_tr, T_tr);
    deviator(T0_tr);

    const double p_tr = -trace(T_tr) / 3.0;
    const double tau_tr = frobenius_norm(T0_tr) / sqrt(2.0);

    const bool density_flag = (density <= RHO_CRITICAL);

    double nup_tau = 0;
    if (density_flag || p_tr <= COHESION) {
        nup_tau = (tau_tr) / (G * dt);

        // mark stress-free
        for (size_t i = 0; i < 9; ++i) {
          T[i] = 0.0;
        }
    } else if (p_tr > COHESION) {
        const double mu_scaling = 1.0;
        const double S0 = mu_scaling * MU_S * p_tr;
        double tau_tau;
        double scale_factor;
        if (tau_tr <= S0) {
            tau_tau = tau_tr;
            scale_factor = 1.0;
        } else {
            const double S2 = mu_scaling * MU_2 * p_tr;
            const double alpha = G * I_0 * dt * sqrt(p_tr / GRAINS_RHO) / GRAINS_D;
            const double B = -(S2 + tau_tr + alpha);
            const double H = S2 * tau_tr + S0 * alpha;
            tau_tau = negative_root(1.0, B, H);
            scale_factor = (tau_tau / tau_tr);
        }

        nup_tau = ((tau_tr - tau_tau) / G) / dt;

        // Set stress according to T = (tau_tau / tau_tr) * T0_tr - pI.
        identity(T);
        scale(T, -p_tr);
        scale(T0_tr, scale_factor);
        accumulate(T, T0_tr);
    } else {
        error->all(FLERR,"Unhandled stress state detected.");
        nup_tau = 0;
    }

    voigt_from_sym_full(cauchy_stress, T);
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOStress::compute_peratom()
{
  int i;
  const int nlocal = atom->nlocal;

  // grow local stress array if necessary
  // needs to be atom->nmax in length

  if (atom->nmax > nmax_store) {
    memory->destroy(stress);
    grow_arrays(atom->nmax);
  }

  if (fix_rheo == nullptr) {
    error->all(FLERR, "fix rheo not set");
  }

  if (fix_rheo->compute_grad == nullptr) {
    error->all(FLERR, "fix rheo gradient computation not set");
  }

  // TODO: fix, violates Law of Demeter
  double **velocity_gradient = fix_rheo->compute_grad->gradv;
  const double *rho = atom->rho;

  // initialize arrays
  if (atom->nmax > nmax_store) grow_arrays(atom->nmax);

  for (i = 0; i < nlocal; ++i) {
    double *T = stress[i];
    const double *L = velocity_gradient[i];
    const double density = rho[i];
    // update_one_material_point_stress(T, L, density);
    update_one_material_point_stress_elastic(T, L, density);

    // if (i == 501) {
    //     printf("Txx, Tyy, Tzz, Txy, Txz, Tyz = %17.17g %17.17g %17.17g %17.17g %17.17g %17.17g\n",
    //         T[0],
    //         T[1],
    //         T[2],
    //         T[3],
    //         T[4],
    //         T[5]
    //     );
    // }
  }

}

/* ---------------------------------------------------------------------- */

int ComputeRHEOStress::pack_forward_comm(int n, int *list, double *buf,
                                        int pbc_flag, int *pbc)
{
  int i,j,k,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    for (k = 0; k < 6; k++)
      buf[m++] = stress[j][k];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOStress::unpack_forward_comm(int n, int first, double *buf)
{
  int i, k, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    for (k = 0; k < 6; k++)
      stress[i][k] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int ComputeRHEOStress::pack_reverse_comm(int n, int first, double *buf)
{
  int i,k,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    for (k = 0; k < 6; k++)
      buf[m++] = stress[i][k];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOStress::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,k,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    for (k = 0; k < 6; k++)
      stress[j][k] += buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOStress::grow_arrays(int nmax)
{
  memory->grow(stress, nmax, 6, "rheo:stress");
  array_atom = stress;
  nmax_store = nmax;
}

/* ---------------------------------------------------------------------- */

double ComputeRHEOStress::memory_usage()
{
  return (size_t) nmax_store * 6 * sizeof(double);
}
