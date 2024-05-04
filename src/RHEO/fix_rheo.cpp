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

#include "fix_rheo.h"

#include "atom.h"
#include "compute_rheo_grad.h"
#include "compute_rheo_interface.h"
#include "compute_rheo_surface.h"
#include "compute_rheo_kernel.h"
#include "compute_rheo_rho_sum.h"
#include "fix_rheo_stress.h"
#include "compute_rheo_stress.h"
#include "compute_rheo_vshift.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "update.h"
#include "utils.h"

#include <cfloat>
#include <cassert>

using namespace LAMMPS_NS;
using namespace RHEO_NS;
using namespace FixConst;

// #define SD_PRINTF(args...) printf(args);
#define SD_PRINTF(args...)

#define DIM(x) (sizeof(x) / sizeof(x[0]))

/* ---------------------------------------------------------------------- */

FixRHEO::FixRHEO(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), compute_grad(nullptr), compute_kernel(nullptr), compute_surface(nullptr),
  compute_interface(nullptr), compute_rhosum(nullptr), compute_vshift(nullptr), rho0(nullptr), csq(nullptr)
{
  time_integrate = 1;

  viscosity_fix_defined = 0;
  pressure_fix_defined = 0;
  thermal_fix_defined = 0;

  thermal_flag = 0;
  rhosum_flag = 0;
  shift_flag = 0;
  interface_flag = 0;
  surface_flag = 0;

  int i;
  int n = atom->ntypes;
  memory->create(rho0, n + 1, "rheo:rho0");
  memory->create(csq, n + 1, "rheo:csq");
  for (i = 1; i <= n; i++) {
    rho0[i] = 1.0;
    csq[i] = 1.0;
  }

  if (igroup != 0)
    error->all(FLERR, "fix rheo command requires group all");

  if (atom->pressure_flag != 1)
    error->all(FLERR, "fix rheo command requires atom_style with pressure");
  if (atom->rho_flag != 1)
    error->all(FLERR, "fix rheo command requires atom_style with density");
  if (atom->viscosity_flag != 1)
    error->all(FLERR, "fix rheo command requires atom_style with viscosity");
  if (atom->status_flag != 1)
    error->all(FLERR, "fix rheo command requires atom_style with status");

  if (narg < 5)
    error->all(FLERR, "Insufficient arguments for fix rheo command");

  h = utils::numeric(FLERR, arg[3], false, lmp);
  cut = h;
  if (strcmp(arg[4], "quintic") == 0) {
      kernel_style = QUINTIC;
  } else if (strcmp(arg[4],"cubic") == 0) {
      kernel_style = CUBIC;
  } else if (strcmp(arg[4], "RK0") == 0) {
      kernel_style = RK0;
  } else if (strcmp(arg[4], "RK1") == 0) {
      kernel_style = RK1;
  } else if (strcmp(arg[4], "RK2") == 0) {
      kernel_style = RK2;
  } else error->all(FLERR, "Unknown kernel style {} in fix rheo", arg[4]);
  zmin_kernel = utils::numeric(FLERR, arg[5], false, lmp);

  int iarg = 6;
  while (iarg < narg){
    if (strcmp(arg[iarg], "shift") == 0) {
      shift_flag = 1;
    } else if (strcmp(arg[iarg], "thermal") == 0) {
      thermal_flag = 1;
    } else if (strcmp(arg[iarg], "surface/detection") == 0) {
      surface_flag = 1;
      if(iarg + 3 >= narg) error->all(FLERR, "Illegal surface/detection option in fix rheo");
      if (strcmp(arg[iarg + 1], "coordination") == 0) {
        surface_style = COORDINATION;
        zmin_surface = utils::inumeric(FLERR, arg[iarg + 2], false, lmp);
        zmin_splash = utils::inumeric(FLERR, arg[iarg + 3], false, lmp);
      } else if (strcmp(arg[iarg + 1], "divergence") == 0) {
        surface_style = DIVR;
        divr_surface = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
        zmin_splash = utils::inumeric(FLERR, arg[iarg + 3], false, lmp);
      } else {
        error->all(FLERR, "Illegal surface/detection option in fix rheo, {}", arg[iarg + 1]);
      }

      iarg += 3;
    } else if (strcmp(arg[iarg], "interface/reconstruct") == 0) {
      interface_flag = 1;
    } else if (strcmp(arg[iarg], "rho/sum") == 0) {
      rhosum_flag = 1;
    } else if (strcmp(arg[iarg], "density") == 0) {
      if (iarg + n >= narg) error->all(FLERR, "Illegal rho0 option in fix rheo");
      for (i = 1; i <= n; i++)
        rho0[i] = utils::numeric(FLERR, arg[iarg + i], false, lmp);
      iarg += n;
    } else if (strcmp(arg[iarg], "speed/sound") == 0) {
      if (iarg + n >= narg) error->all(FLERR, "Illegal csq option in fix rheo");
      for (i = 1; i <= n; i++) {
        csq[i] = utils::numeric(FLERR, arg[iarg + i], false, lmp);
        csq[i] *= csq[i];
      }
      iarg += n;
    } else {
      error->all(FLERR, "Illegal fix rheo command: {}", arg[iarg]);
    }
    iarg += 1;
  }
}

/* ---------------------------------------------------------------------- */

FixRHEO::~FixRHEO()
{
  if (compute_kernel) modify->delete_compute("rheo_kernel");
  if (compute_grad) modify->delete_compute("rheo_grad");
  if (compute_interface) modify->delete_compute("rheo_interface");
  if (compute_surface) modify->delete_compute("rheo_surface");
  if (compute_rhosum) modify->delete_compute("rheo_rhosum");
  if (compute_vshift) modify->delete_compute("rheo_vshift");

  memory->destroy(csq);
  memory->destroy(rho0);
}


/* ----------------------------------------------------------------------
  Create necessary internal computes
------------------------------------------------------------------------- */

void FixRHEO::post_constructor()
{
  compute_kernel = dynamic_cast<ComputeRHEOKernel *>(modify->add_compute(
    fmt::format("rheo_kernel all RHEO/KERNEL {}", kernel_style)));
  compute_kernel->fix_rheo = this;

  std::string cmd = "rheo_grad all RHEO/GRAD velocity rho viscosity";
  if (thermal_flag) cmd += " energy";
  compute_grad = dynamic_cast<ComputeRHEOGrad *>(modify->add_compute(cmd));
  compute_grad->fix_rheo = this;

  if (rhosum_flag) {
    compute_rhosum = dynamic_cast<ComputeRHEORhoSum *>(modify->add_compute(
      "rheo_rhosum all RHEO/RHO/SUM"));
    compute_rhosum->fix_rheo = this;
  }

  if (shift_flag) {
    compute_vshift = dynamic_cast<ComputeRHEOVShift *>(modify->add_compute(
      "rheo_vshift all RHEO/VSHIFT"));
    compute_vshift->fix_rheo = this;
  }

  if (interface_flag) {
    compute_interface = dynamic_cast<ComputeRHEOInterface *>(modify->add_compute(
      "rheo_interface all RHEO/INTERFACE"));
    compute_interface->fix_rheo = this;
  }

  if (surface_flag) {
    compute_surface = dynamic_cast<ComputeRHEOSurface *>(modify->add_compute(
      "rheo_surface all RHEO/SURFACE"));
    compute_surface->fix_rheo = this;
  }
}

/* ---------------------------------------------------------------------- */

int FixRHEO::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= PRE_FORCE;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRHEO::init()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;

  if (modify->get_fix_by_style("^rheo$").size() > 1)
    error->all(FLERR, "Can only specify one instance of fix rheo");
}

/* ---------------------------------------------------------------------- */

void FixRHEO::setup_pre_force(int /*vflag*/)
{
  // Check to confirm accessory fixes do not preceed FixRHEO
  // Note: fixes set this flag in setup_pre_force()
  if (viscosity_fix_defined || pressure_fix_defined || thermal_fix_defined)
    error->all(FLERR, "Fix RHEO must be defined before all other RHEO fixes");

  // Calculate surfaces
  if (surface_flag) {
    compute_kernel->compute_coordination();
    compute_surface->compute_peratom();
  }

  pre_force(0);
}

/* ---------------------------------------------------------------------- */

static void bc_setup(void);
static double clamp_unity(double v);

void FixRHEO::setup(int /*vflag*/)
{
  // Confirm all accessory fixes are defined
  // Note: fixes set this flag in setup_pre_force()
  if (!viscosity_fix_defined)
    error->all(FLERR, "Missing fix rheo/viscosity");

  if (!pressure_fix_defined)
    error->all(FLERR, "Missing fix rheo/pressure");

  if((!thermal_fix_defined) && thermal_flag)
    error->all(FLERR, "Missing fix rheo/thermal");

  // Reset to zero for future runs
  thermal_fix_defined = 0;
  viscosity_fix_defined = 0;
  pressure_fix_defined = 0;

  // Check fixes cover all atoms (may still fail if atoms are created)
  // FixRHEOPressure currently requires group all
  auto visc_fixes = modify->get_fix_by_style("rheo/viscosity");
  auto therm_fixes = modify->get_fix_by_style("rheo/thermal");

  int *mask = atom->mask;
  int v_coverage_flag = 1;
  int t_coverage_flag = 1;
  int covered;
  for (int i = 0; i < atom->nlocal; i++) {
    covered = 0;
    for (auto fix : visc_fixes)
      if (mask[i] & fix->groupbit) covered = 1;
    if (!covered) v_coverage_flag = 0;
    if (thermal_flag) {
      covered = 0;
      for (auto fix : therm_fixes)
        if (mask[i] & fix->groupbit) covered = 1;
      if (!covered) t_coverage_flag = 0;
    }
  }

  if (!v_coverage_flag)
    error->one(FLERR, "Fix rheo/viscosity does not fully cover all atoms");
  if (!t_coverage_flag)
    error->one(FLERR, "Fix rheo/thermal does not fully cover all atoms");

  if (rhosum_flag)
    compute_rhosum->compute_peratom();

  bc_setup();
}

/* ---------------------------------------------------------------------- */


// Normal points leftward along path (xl, yl) to (xr, yr).
typedef struct {
    double xl;
    double yl;
    double xr;
    double yr;
    double ramp_thickness;
    double dead_thickness;
    double mu;
} sd_boundary_t;

typedef struct {
    double x;
    double y;
    double z;
} vector_3d_t;

// Normal formed from r1 x r2. ramp_thickness is along positive normal and
// dead_thickness is along negative normal. r1 and r2 are HALF of the distance
// in each dimension of the bounding rectangular prism.
typedef struct {
    vector_3d_t origin;
    vector_3d_t r1;
    vector_3d_t r2;

    double ramp_thickness;
    double dead_thickness;
    double mu;

    // computed
    double r1_mag;
    double r2_mag;
    vector_3d_t n1;
    vector_3d_t n2;
    vector_3d_t n3;
} sd_boundary_3d_t;

static double dot(const vector_3d_t * const a, const vector_3d_t * const b)
{
    return (a->x * b->x) + (a->y * b->y) + (a->z * b->z);
}

static double magnitude_squared(const vector_3d_t * const v)
{
    return dot(v, v);
}

static void normalize(vector_3d_t *v)
{
    const double d = sqrt(magnitude_squared(v));
    v->x /= d;
    v->y /= d;
    v->z /= d;
}

static vector_3d_t compute_normal(const vector_3d_t * const r1, const vector_3d_t * const r2)
{
    vector_3d_t n = {
        .x =  ((r1->y * r2->z) - (r1->z * r2->y)),
        .y = -((r1->x * r2->z) - (r1->z * r2->x)),
        .z =  ((r1->x * r2->y) - (r1->y * r2->x)),
    };

    // Actually normalize the normal too...
    normalize(&n);

    return n;
}

// static const double boundary_thickness = 0.1;
static const double boundary_thickness = 0.01;
static const double dead_thickness = 0.1;

static const sd_boundary_t boundaries[] = {
    // bottom wall
    // {-3.0, -10.0, 3.0, -10.0, boundary_thickness, dead_thickness, 1.0},
    {-20.0, -6.0, 20.0, -6.0, boundary_thickness, dead_thickness, 1.0},

    // silo orifice walls
    // {-3.0, 0.0, -1.0, 0.0, boundary_thickness, dead_thickness, 1.0},
    // {1.0, 0.0, 3.0, 0.0, boundary_thickness, dead_thickness, 1.0},
    {-3.0, 0.0, -1.0, -1.0, boundary_thickness, dead_thickness, 1.0},
    {1.0, -1.0, 3.0, 0.0, boundary_thickness, dead_thickness, 1.0},

    // outer walls
    // {-3.0, 3.0, -3.0, -10.0, boundary_thickness, dead_thickness, 0.0},
    // {3.0, -10.0, 3.0, 3.0, boundary_thickness, dead_thickness, 0.0},
    {-3.0, 3.0, -3.0, 0.0, boundary_thickness, dead_thickness, 0.0},
    {3.0, 0.0, 3.0, 3.0, boundary_thickness, dead_thickness, 0.0},
};

static const double scale = 1.001;

static sd_boundary_3d_t b3[] = {
    // 4 angled hopper plates
    {
        .origin = {0.0, -1.0, 2.0},
        .r1 = {-3.0, 0.0, 0.0},
        .r2 = {0.0, 1.0, 1.0},
        .ramp_thickness = boundary_thickness,
        .dead_thickness = dead_thickness,
        .mu = 0.0,
    },
    {
        .origin = {2.0, -1.0, 0.0},
        .r1 = {0.0, 0.0, 3.0},
        .r2 = {1.0, 1.0, 0.0},
        .ramp_thickness = boundary_thickness,
        .dead_thickness = dead_thickness,
        .mu = 0.0,
    },
    {
        .origin = {0.0, -1.0, -2.0},
        .r1 = {3.0, 0.0, 0.0},
        .r2 = {0.0, 1.0, -1.0},
        .ramp_thickness = boundary_thickness,
        .dead_thickness = dead_thickness,
        .mu = 0.0,
    },
    {
        .origin = {-2.0, -1.0, 0.0},
        .r1 = {0.0, 0.0, -3.0},
        .r2 = {-1.0, 1.0, 0.0},
        .ramp_thickness = boundary_thickness,
        .dead_thickness = dead_thickness,
        .mu = 0.0,
    },

    // 4 sidewalls
    {
        .origin = {0.0, 1.5, 3.0},
        .r1 = {-3.0, 0.0, 0.0},
        .r2 = {0.0, 1.5, 0.0},
        .ramp_thickness = boundary_thickness,
        .dead_thickness = dead_thickness,
        .mu = 0.0,
    },
    {
        .origin = {3.0, 1.5, 0.0},
        .r1 = {0.0, 0.0, 3.0},
        .r2 = {0.0, 1.5, 0.0},
        .ramp_thickness = boundary_thickness,
        .dead_thickness = dead_thickness,
        .mu = 0.0,
    },
    {
        .origin = {0.0, 1.5, -3.0},
        .r1 = {3.0, 0.0, 0.0},
        .r2 = {0.0, 1.5, 0.0},
        .ramp_thickness = boundary_thickness,
        .dead_thickness = dead_thickness,
        .mu = 0.0,
    },
    {
        .origin = {-3.0, 1.5, 0.0},
        .r1 = {0.0, 0.0, -3.0},
        .r2 = {0.0, 1.5, 0.0},
        .ramp_thickness = boundary_thickness,
        .dead_thickness = dead_thickness,
        .mu = 0.0,
    },
    // bottom collector
    {
        .origin = {0.0, -6.0, 0.0},
        // .origin = {0.0, -0.0, 0.0},
        .r1 = {40.0, 0.0, 0.0},
        .r2 = {0.0, 0.0, -40.0},
        .ramp_thickness = boundary_thickness,
        .dead_thickness = dead_thickness,
        .mu = 1.0,
    },
};

static void bc_setup(void)
{
    for (size_t bi = 0; bi < sizeof(b3)/sizeof(b3[0]); ++bi) {
        sd_boundary_3d_t * const entry = &b3[bi];
        entry->origin.x *= scale;
        entry->origin.y *= scale;
        entry->origin.z *= scale;
        entry->r1.x *= scale;
        entry->r1.y *= scale;
        entry->r1.z *= scale;
        entry->r2.x *= scale;
        entry->r2.y *= scale;
        entry->r2.z *= scale;
    }

    for (size_t bi = 0; bi < sizeof(b3)/sizeof(b3[0]); ++bi) {
        sd_boundary_3d_t * const entry = &b3[bi];
        entry->n1 = entry->r1;
        normalize(&entry->n1);
        entry->n2 = entry->r2;
        normalize(&entry->n2);
        entry->n3 = compute_normal(&entry->r1, &entry->r2);

        entry->r1_mag = sqrt(magnitude_squared(&entry->r1));
        entry->r2_mag = sqrt(magnitude_squared(&entry->r2));
    }

    for (size_t bi = 0; bi < sizeof(b3)/sizeof(b3[0]); ++bi) {
        const sd_boundary_3d_t * const entry = &b3[bi];
        printf("boundary[%zu]: origin = {%.17g, %.17g, %.17g}\n", bi, entry->origin.x, entry->origin.y, entry->origin.z);
        printf("boundary[%zu]: r1 = {%.17g, %.17g, %.17g}\n", bi, entry->r1.x, entry->r1.y, entry->r1.z);
        printf("boundary[%zu]: r2 = {%.17g, %.17g, %.17g}\n", bi, entry->r2.x, entry->r2.y, entry->r2.z);
        printf("boundary[%zu]: ramp_thickness = %.17g\n", bi, entry->ramp_thickness);
        printf("boundary[%zu]: dead_thickness = %.17g\n", bi, entry->dead_thickness);
        printf("boundary[%zu]: mu = %.17g\n", bi, entry->mu);
        printf("boundary[%zu]: n1 = {%.17g, %.17g, %.17g}\n", bi, entry->n1.x, entry->n1.y, entry->n1.z);
        printf("boundary[%zu]: n2 = {%.17g, %.17g, %.17g}\n", bi, entry->n2.x, entry->n2.y, entry->n2.z);
        printf("boundary[%zu]: n3 = {%.17g, %.17g, %.17g}\n", bi, entry->n3.x, entry->n3.y, entry->n3.z);
        printf("boundary[%zu]: r1_mag = %.17g\n", bi, entry->r1_mag);
        printf("boundary[%zu]: r2_mag = %.17g\n", bi, entry->r2_mag);
        printf("---\n");
    }
}

static void b3_strength(double *strength, bool *in_dead_zone, const vector_3d_t * const xp, const sd_boundary_3d_t * const boundary)
{
    // set outputs
    *strength = 0.0;
    *in_dead_zone = false;

    const vector_3d_t v = {
        .x = xp->x - boundary->origin.x,
        .y = xp->y - boundary->origin.y,
        .z = xp->z - boundary->origin.z,
    };

    const double x1 = dot(&v, &boundary->n1);
    if (-boundary->r1_mag <= x1 && x1 <= boundary->r1_mag) {
        const double x2 = dot(&v, &boundary->n2);
        if (-boundary->r2_mag <= x2 && x2 <= boundary->r2_mag) {
            const double x3 = dot(&v, &boundary->n3);
            if (-boundary->dead_thickness <= x3 && x3 <= 0) {
                *strength = 1.0;
                *in_dead_zone = true;
            } else if (0 < x3 && x3 <= boundary->ramp_thickness) {
                *strength = clamp_unity(1.0 - (x3 / boundary->ramp_thickness));
                *in_dead_zone = false;
            }
        }
    }
}

static void b3_distance(double *distance, const vector_3d_t * const xp, const sd_boundary_3d_t * const boundary)
{
    // set outputs
    *distance = DBL_MAX;

    const vector_3d_t v = {
        .x = xp->x - boundary->origin.x,
        .y = xp->y - boundary->origin.y,
        .z = xp->z - boundary->origin.z,
    };

    const double x1 = dot(&v, &boundary->n1);
    if (-boundary->r1_mag <= x1 && x1 <= boundary->r1_mag) {
        const double x2 = dot(&v, &boundary->n2);
        if (-boundary->r2_mag <= x2 && x2 <= boundary->r2_mag) {
            const double x3 = dot(&v, &boundary->n3);
            *distance = x3;
        }
    }
}

static bool is_wall_close(uint32_t wall_bitmask, size_t wall_index)
{
    return ((wall_bitmask & (UINT32_C(1) << wall_index)) != 0);
}

static void boundary_force_direction_from_levelset(double *strength,
                                                   vector_3d_t *direction,
                                                   uint32_t *walls_bitset,
                                                   const vector_3d_t * const xp)
                                                   // ,
                                                   // const sd_boundary_3d_t * const boundaries,
                                                   // size_t num_boundaries)
{
    // Take the function f = prod(d_1, d_2, ...) (d_i distance from boundary i).
    // The gradient gives us a nice direction, which can be written as
    // sum_over_i(n_i * prod_for_j_not_equal_i(d_j)) where n_i is the normal to
    // the boundary segment.
    static double distances[DIM(b3)] = {0};
    // already part of the boundary!
    // static vector_3d_t normals[DIM(b3)] = {0};

    uint32_t wb = 0;
    for (size_t bi = 0; bi < DIM(b3); ++bi) {
        const sd_boundary_3d_t * const entry = &b3[bi];
        // double s = 0.0;
        // bool in_dead_zone = false;
        // b3_strength(&s, &in_dead_zone, xp, entry);
        // (void)in_dead_zone;
        // distances[bi] = 1.0 - s;
        b3_distance(&distances[bi], xp, entry);

        // Wrong side of the BC, set back to a far away value.
        if (distances[bi] < 0.0) {
            distances[bi] = DBL_MAX;
        }

        if (distances[bi] < entry->ramp_thickness) {
            wb |= (UINT32_C(1) << bi);
        }
    }

    *walls_bitset = wb;

    // At least one wall detected.
    if (wb != 0) {
        double min_d = distances[0];
        size_t min_wall_index = 0;
        for (size_t i = 0; i < DIM(distances); ++i) {
            if (is_wall_close(wb, i) && (distances[i] < min_d)) {
                min_d = distances[i];
                min_wall_index = i;
            }
        }
        // Do we want min distance, or max strength BC?
        *strength = clamp_unity(1.0 - (min_d / b3[min_wall_index].ramp_thickness));
    } else {
        *strength = 0.0;
    }

    if (*strength != 0.0) {
        *direction = (vector_3d_t) {
            0.0,
            0.0,
            0.0
        };
        for (size_t i = 0; i < DIM(distances); ++i) {
            double pi_d = 1.0;
            for (size_t j = 0; j < DIM(distances); ++j) {
                // Not the current wall we're considering (chain rule), and
                // product of all other walls that are within range.
                if ((j != i) && is_wall_close(wb, j)) {
                    // distances should all be positive by this point, so fabs is
                    // unnecessary...
                    // pi_d *= fabs(distances[j]);
                    pi_d *= distances[j];
                }
            }

            direction->x += pi_d * b3[i].n3.x;
            direction->y += pi_d * b3[i].n3.y;
            direction->z += pi_d * b3[i].n3.z;
        }

        normalize(direction);
    }
}

static void boundary_normal(double *xn, double *yn, const sd_boundary_t * const boundary)
{
    const double dx = boundary->xr - boundary->xl;
    const double dy = boundary->yr - boundary->yl;
    const double r = hypot(dx, dy);
    *xn = -dy / r;
    *yn = dx / r;
}

static double clamp_unity(double v)
{
    if (v < 0.0) {
        return 0.0;
    } else if (v > 1.0) {
        return 1.0;
    } else {
        return v;
    }
}

static void boundary_strength(double *strength, bool *in_dead_zone, double x, double y, const sd_boundary_t * const boundary)
{
    const double dtx = x - boundary->xl;
    const double dty = y - boundary->yl;

    const double dx = boundary->xr - boundary->xl;
    const double dy = boundary->yr - boundary->yl;

    const double s = (dx * dtx + dy * dty) / (dx * dx + dy * dy);

    // set outputs
    *strength = 0.0;
    *in_dead_zone = false;

    // Nominally within line segment region.
    if (0.0 <= s && s <= 1.0) {
        const double r = hypot(dx, dy);
        double d = 0.0;
        if (dx != 0.0) {
            d = r * ((dty - dy * s) / dx);
        } else if (dy != 0.0) {
            d = r * (-(dtx - dx * s) / dy);
        }

        if (0.0 <= d && d <= boundary->ramp_thickness) {
            *strength = clamp_unity(1.0 - (d / boundary->ramp_thickness));
            *in_dead_zone = false;
        }

        if (-boundary->dead_thickness <= d && d < 0.0) {
            *strength = 1.0;
            *in_dead_zone = true;
        }
    }
}

void FixRHEO::post_force(int /*vflag*/)
{
  // update v, x and rho of atoms in group
  int i, a, b;
  double dtfm, divu;

  int *type = atom->type;
  int *mask = atom->mask;
  int *status = atom->status;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rho = atom->rho;
  double *drho = atom->drho;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  double **gradr = compute_grad->gradr;
  double **gradv = compute_grad->gradv;
  double **vshift;
  if (shift_flag)
    vshift = compute_vshift->vshift;

  int nlocal = atom->nlocal;
  int rmass_flag = atom->rmass_flag;
  int dim = domain->dimension;

  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;

  auto fixes = modify->get_fix_by_style("rheo/stress");
  if (fixes.size() == 0) error->all(FLERR, "Need to define fix rheo/stress to use pair rheo");
  auto *fix_stress = dynamic_cast<FixRHEOStress *>(fixes[0]);
  auto *stress_compute = dynamic_cast<ComputeRHEOStress *>(fix_stress->stress_compute);
  double **stress = stress_compute->array_atom;

  // hack for BCS
  // [sdunatunga] Tue 13 Feb 2024 07:53:19 AM PST
  for (i = 0; i < nlocal; i++) {
    if (status[i] & STATUS_NO_INTEGRATION) continue;

    if (mask[i] & groupbit) {
      if (rmass_flag) {
        dtfm = dtf / rmass[i];
      } else {
        dtfm = dtf / mass[type[i]];
      }

        const double ftest[] = {
            -v[i][0] / dtfm,
            -v[i][1] / dtfm,
            -v[i][2] / dtfm,
        };

        const vector_3d_t xp = {
            .x = x[i][0],
            .y = x[i][1],
            .z = x[i][2],
        };

        vector_3d_t fdir = {
            0.0,
            0.0,
            0.0,
        };

        double s = 0.0;
        uint32_t walls_bitset = 0;
        boundary_force_direction_from_levelset(&s, &fdir, &walls_bitset, &xp);

        bool is_any_wall_sticky = false;
        for (size_t i = 0; i < DIM(b3); ++i) {
            const sd_boundary_3d_t * const entry = &b3[i];
            // Check if we are in contact with a sticky wall.
            if (is_wall_close(walls_bitset, i)) {
                is_any_wall_sticky |= (entry->mu != 0.0);
            }
        }

        stress[i][12] = s;
        stress[i][13] = xp.x;
        stress[i][14] = xp.y;
        stress[i][15] = xp.z;
        // stress[i][16] = in_dead_zone;
        stress[i][17] = walls_bitset;
        stress[i][18] = ftest[0];
        stress[i][19] = ftest[1];
        stress[i][20] = ftest[2];
        stress[i][21] = f[i][0];
        stress[i][22] = f[i][1];
        stress[i][23] = f[i][2];
        stress[i][24] = fdir.x;
        stress[i][25] = fdir.y;
        stress[i][26] = fdir.z;
        if (s != 0.0) {
            // We can slide along all walls in this contact, so we need a
            // direction.
            if (!is_any_wall_sticky) {
                const vector_3d_t normal = fdir;

                const vector_3d_t vf = {
                    .x = f[i][0],
                    .y = f[i][1],
                    .z = f[i][2],
                };

                const vector_3d_t vft = {
                    .x = ftest[0],
                    .y = ftest[1],
                    .z = ftest[2],
                };

                const double fn_mag = dot(&vf, &normal);

                const double fn[] = {
                    normal.x * fn_mag,
                    normal.y * fn_mag,
                    normal.z * fn_mag,
                };

                const double fw_mag = dot(&vft, &normal);

                const double fw[] = {
                    normal.x * fw_mag,
                    normal.y * fw_mag,
                    normal.z * fw_mag,
                };

                const double deltaf[] = {
                    s * (fw[0] - fn[0]),
                    s * (fw[1] - fn[1]),
                    s * (fw[2] - fn[2]),
                };

                f[i][0] = deltaf[0] + f[i][0];
                f[i][1] = deltaf[1] + f[i][1];
                f[i][2] = deltaf[2] + f[i][2];

                stress[i][6] = deltaf[0];
                stress[i][7] = deltaf[1];
                stress[i][8] = deltaf[2];
            } else {
                const double deltaf[] = {
                    s * (ftest[0] - f[i][0]),
                    s * (ftest[1] - f[i][1]),
                    s * (ftest[2] - f[i][2]),
                };

                f[i][0] = deltaf[0] + f[i][0];
                f[i][1] = deltaf[1] + f[i][1];
                f[i][2] = deltaf[2] + f[i][2];

                stress[i][9] = deltaf[0];
                stress[i][10] = deltaf[1];
                stress[i][11] = deltaf[2];
            }
        }

// 2D BC only
#if 0
        for (size_t bi = 0; bi < sizeof(boundaries)/sizeof(boundaries[0]); ++bi) {
            double s = 0.0;
            bool in_dead_zone = false;
            boundary_strength(&s, &in_dead_zone, x[i][0], x[i][1], &boundaries[bi]);

            if (s != 0.0) {
                double n[3] = {
                    0.0,
                    0.0,
                    0.0,
                };

                // Flip normal if in the dead zone to get the right force
                // direction.
                if (boundaries[bi].mu == 0.0) {
                    boundary_normal(&n[0], &n[1], &boundaries[bi]);
                    if (in_dead_zone) {
                        n[0] = -n[0];
                        n[1] = -n[1];
                        n[2] = -n[2];
                    }
                    f[i][0] = n[0] * s * ftest[0] + (1.0 - s) * f[i][0];
                    f[i][1] = n[1] * s * ftest[1] + (1.0 - s) * f[i][1];
                    f[i][2] = n[2] * s * ftest[2] + (1.0 - s) * f[i][2];
                } else {
                    f[i][0] = s * ftest[0] + (1.0 - s) * f[i][0];
                    f[i][1] = s * ftest[1] + (1.0 - s) * f[i][1];
                    f[i][2] = s * ftest[2] + (1.0 - s) * f[i][2];
                }
                // FIXME : only applies the first wall this particle checks against.
                break;
            }
        }
#endif

#if 0
        // silo type BCS
        // bottom wall
        {
            const double width = 0.2;
            const double y_c = -10.0;
            const double y = x[i][1];
            double s = ((y - y_c) / width);
            if (s > 1) {
                s = 1;
            } else if (s < 0) {
                s = 0;
            }
            f[i][0] = (1.0 - s) * ftest[0] + s * f[i][0];
            f[i][1] = (1.0 - s) * ftest[1] + s * f[i][1];
            f[i][2] = (1.0 - s) * ftest[2] + s * f[i][2];

            // if (y < y_c) {
            //     v[i][0] *= s;
            //     v[i][1] *= s;
            //     v[i][2] *= s;
            // }
            // if (y < (y_c - width)) {
            //     v[i][0] = 0;
            //     v[i][1] = 0;
            //     v[i][2] = 0;
            // }
        }
        // left side
        {
            const double width = 0.1;
            const double y_c = 0;
            const double _x = x[i][0];
            const double y = x[i][1];
            double s = ((y - y_c) / width);
            // double s = ((y_c - y) / width);
            // if (((_x > 1) || (_x < -1)) && (y < y_c)) {
            //     v[i][0] *= s;
            //     v[i][1] *= s;
            //     v[i][2] *= s;
            // }
            // if (((_x > 1) || (_x < -1)) && ((y < (y_c - width)) && (y > -1))) {
            //     v[i][0] = 0;
            //     v[i][1] = 0;
            //     v[i][2] = 0;
            // }
            if (s > 1) {
                s = 1;
            } else if (s < 0) {
                s = 0;
            }

            if ((-1 <= _x && _x <= 1) || y < (y_c - width)) {
                // Free particle below the ledge
                s = 1;
            }
/*
            if (((_x > 1) || (_x < -1)) && (y < y_c)) {
            } else {
                s = 1.0;
            }
            if (((_x > 1) || (_x < -1)) && ((y < (y_c - width)) && (y > -1))) {
                s = 0;
            }
*/
            f[i][0] = (1.0 - s) * ftest[0] + s * f[i][0];
            f[i][1] = (1.0 - s) * ftest[1] + s * f[i][1];
            f[i][2] = (1.0 - s) * ftest[2] + s * f[i][2];
        }
#endif
    }
  }
}

void FixRHEO::initial_integrate(int /*vflag*/)
{
  // update v, x and rho of atoms in group
  int i, a, b;
  double dtfm, divu;

  int *type = atom->type;
  int *mask = atom->mask;
  int *status = atom->status;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rho = atom->rho;
  double *drho = atom->drho;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  double **gradr = compute_grad->gradr;
  double **gradv = compute_grad->gradv;
  double **vshift;
  if (shift_flag)
    vshift = compute_vshift->vshift;

  int nlocal = atom->nlocal;
  int rmass_flag = atom->rmass_flag;
  int dim = domain->dimension;

  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;

  auto fixes = modify->get_fix_by_style("rheo/stress");
  if (fixes.size() == 0) error->all(FLERR, "Need to define fix rheo/stress to use pair rheo");
  auto *fix_stress = dynamic_cast<FixRHEOStress *>(fixes[0]);
  auto *stress_compute = dynamic_cast<ComputeRHEOStress *>(fix_stress->stress_compute);
  double **stress = stress_compute->array_atom;

  // [sdunatunga] Tue 23 Apr 2024 12:10:17 AM PDT
  // post_force contents used to be here

  //Density Half-step
  for (i = 0; i < nlocal; i++) {
    if (status[i] & STATUS_NO_INTEGRATION) continue;

    if (mask[i] & groupbit) {
      if (rmass_flag) {
        dtfm = dtf / rmass[i];
      } else {
        dtfm = dtf / mass[type[i]];
      }

      v[i][0] += dtfm * f[i][0];
      v[i][1] += dtfm * f[i][1];
      v[i][2] += dtfm * f[i][2];

      if (atom->tag[i] == 1) {
          SD_PRINTF("initial_integrate v = [%17.9g %17.9g %17.9g]\n", v[i][0], v[i][1], v[i][2]);
      }
    }
  }

  // Update gradients and interpolate solid properties
  compute_grad->forward_fields(); // also forwards v and rho for chi
  if (interface_flag) {
      // Need to save, wiped in exchange
      compute_interface->store_forces();
      compute_interface->compute_peratom();
  }
  compute_grad->compute_peratom();

  // Position half-step
  for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
          for (a = 0; a < dim; a++) {
              x[i][a] += dtv * v[i][a];
          }
      }
  }

  // Update density using div(u)
  if (!rhosum_flag) {
      for (i = 0; i < nlocal; i++) {
          if (mask[i] & groupbit) {
              if (status[i] & STATUS_NO_INTEGRATION) continue;
              if (status[i] & PHASECHECK) continue;

              divu = 0;
              for (a = 0; a < dim; a++) {
                  divu += gradv[i][a * (1 + dim)];
              }
              rho[i] += dtf * (drho[i] - rho[i] * divu);
          }
      }
  }

  // Shifting atoms
  if (shift_flag) {
      for (i = 0; i < nlocal; i++) {

          if (status[i] & STATUS_NO_SHIFT) continue;

          if (mask[i] & groupbit) {
              for (a = 0; a < dim; a++) {
                  x[i][a] += dtv * vshift[i][a];
                  for (b = 0; b < dim; b++) {
                      v[i][a] += dtv * vshift[i][b] * gradv[i][a * dim + b];
                  }
              }

              if (!rhosum_flag) {
                  if (status[i] & PHASECHECK) continue;
                  for (a = 0; a < dim; a++) {
                      rho[i] += dtv * vshift[i][a] * gradr[i][a];
                  }
              }
          }
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixRHEO::pre_force(int /*vflag*/)
{
    compute_kernel->compute_coordination(); // Needed for rho sum

    if (rhosum_flag)
        compute_rhosum->compute_peratom();

    compute_kernel->compute_peratom();

    if (interface_flag) {
        // Note on first setup, have no forces for pressure to reference
        compute_interface->compute_peratom();
    }

    // No need to forward v, rho, or T for compute_grad since already done
    compute_grad->compute_peratom();
    compute_grad->forward_gradients();

    if (shift_flag)
        compute_vshift->compute_peratom();

    // Remove temporary options
    int *mask = atom->mask;
    int *status = atom->status;
    int nall = atom->nlocal + atom->nghost;
    for (int i = 0; i < nall; i++)
        if (mask[i] & groupbit)
            status[i] &= OPTIONSMASK;

    // Calculate surfaces, update status
    if (surface_flag) {
        compute_surface->compute_peratom();
        if (shift_flag)
            compute_vshift->correct_surfaces();
    }
}

/* ---------------------------------------------------------------------- */

void FixRHEO::final_integrate()
{
    int nlocal = atom->nlocal;
    if (igroup == atom->firstgroup)
        nlocal = atom->nfirst;

    double dtfm, divu;
    int i, a;

    double **x = atom->x;
    double **v = atom->v;
    double **f = atom->f;
    double **gradv = compute_grad->gradv;
    double *rho = atom->rho;
    double *drho = atom->drho;
    double *mass = atom->mass;
    double *rmass = atom->rmass;
    int *type = atom->type;
    int *mask = atom->mask;
    int *status = atom->status;

    int rmass_flag = atom->rmass_flag;
    int dim = domain->dimension;

    // Update velocity
    for (i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            if (status[i] & STATUS_NO_INTEGRATION) continue;

            if (rmass_flag) {
                dtfm = dtf / rmass[i];
            } else {
                dtfm = dtf / mass[type[i]];
            }

            for (a = 0; a < dim; a++) {
                v[i][a] += dtfm * f[i][a];
                if ((atom->tag[i] == 1) && (a == 0)) {
                    SD_PRINTF("final_integrate v = [%17.9g %17.9g %17.9g]\n", v[i][0], v[i][1], v[i][2]);
        }
      }
    }
  }

  // Update density using divu
  if (!rhosum_flag) {
    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (status[i] & STATUS_NO_INTEGRATION) continue;
        if (status[i] & PHASECHECK) continue;

        divu = 0;
        for (a = 0; a < dim; a++) {
          divu += gradv[i][a * (1 + dim)];
        }
        rho[i] += dtf * (drho[i] - rho[i] * divu);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixRHEO::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}
