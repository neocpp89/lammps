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
#include <cstdlib>

using namespace LAMMPS_NS;
using namespace RHEO_NS;
using namespace FixConst;

// #define SD_PRINTF(args...) printf(args);
#define SD_PRINTF(args...)

#define DIM(x) (sizeof(x) / sizeof(x[0]))

typedef struct {
    double x;
    double y;
    double z;
} vector_3d_t;

typedef struct {
    vector_3d_t normal;
    vector_3d_t a;
    vector_3d_t b;
    vector_3d_t c;
    bool sticky;
    double thickness;
} stl_facet_t;

typedef struct {
    double *sdf_values;
    bool *sticky_bits;

    size_t ispan;
    size_t jspan;
    size_t kspan;
    double dx;
    double dy;
    double dz;

    // Not sure if extreme is really needed but could avoid a few divides at
    // the cost of loading 3 doubles.
    vector_3d_t origin;
    vector_3d_t extreme;

    // This is the triangle list at nodes (0, dx, 2*dx, ...)
    uint64_t **triangle_lists;

    // This is the triangle list across cells, spanning (0, dx), then (dx,
    // 2*dx) etc. Note that there are only ispan-1, jspan-1, and kspan-1
    // entries in each direction, so the indexing changes.
    // TODO: Need to make sure that we compute this correctly when we optimize
    // later, e.g. we don't miss if a sliver of a triangle is in a cell.
    uint64_t **cell_centered_triangle_lists;
} stl_voxel_grid_t;

static stl_facet_t *loaded_facets = NULL;
static size_t num_loaded_facets = 0;
static double boundary_thickness = 0.01;
static uint64_t sticky_bitmask = 0;
static stl_voxel_grid_t voxel_grid = {0};

static void cross(vector_3d_t * const result,
                  const vector_3d_t * const a,
                  const vector_3d_t * const b);
static void vsub(vector_3d_t * const result,
                 const vector_3d_t * const a,
                 const vector_3d_t * const b);
static void normalize(vector_3d_t *v);
static void print_facet(const stl_facet_t * const facet);
static bool parse_stl_file(stl_facet_t *facets, size_t *num_facets, FILE *fp);

static void stl_facet_distance(double *distance, const vector_3d_t * const xp, const stl_facet_t * const facet);
static void make_stl_voxel_grid(stl_voxel_grid_t *vgrid, const stl_facet_t * const facets, size_t num_facets, double spacing);

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
  size_t max_stl_facets = 4096;
  FILE *boundary_fp = NULL;
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
    } else if (strcmp(arg[iarg], "boundary/stlfile") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal stlfile option in fix rheo");
      boundary_fp = fopen(arg[iarg + 1], "r");
      iarg += 1;
    } else if (strcmp(arg[iarg], "boundary/rampthickness") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal ramp thickness option in fix rheo");
      boundary_thickness  = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 1;
    } else if (strcmp(arg[iarg], "boundary/stlmaxfacets") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal number of max stl facets.");
      // sticky_bitmask = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      max_stl_facets = strtoull(arg[iarg + 1], NULL, 0);
      iarg += 1;
    } else if (strcmp(arg[iarg], "boundary/stickybitmask") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal sticky bc option in fix rheo");
      // sticky_bitmask = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      sticky_bitmask = strtoull(arg[iarg + 1], NULL, 0);
      iarg += 1;
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

  if (boundary_fp != NULL) {
    // num_loaded_facets = DIM(loaded_facets);
    num_loaded_facets = max_stl_facets;
    loaded_facets = static_cast<stl_facet_t *>(calloc(max_stl_facets, sizeof(loaded_facets[0])));
    if (loaded_facets == NULL) {
        error->all(FLERR, "Could not allocate space for {} facets.", max_stl_facets);
    }
    parse_stl_file(loaded_facets, &num_loaded_facets, boundary_fp);
    for (size_t i = 0; i < num_loaded_facets; ++i) {
        stl_facet_t * const entry = &loaded_facets[i];
        if (i < 64) {
            entry->sticky = ((sticky_bitmask & (UINT64_C(1) << i)) != 0);
            entry->thickness = boundary_thickness;

            // If all three normal components are 0, assume that we're supposed
            // to take the three vertices in increasing angle (counter
            // clockwise) and compute the normal from those.
            if (entry->normal.x == 0.0 &&
                entry->normal.y == 0.0 &&
                entry->normal.z == 0.0) {
                // vectors from P to x.
                vector_3d_t v_ab = {0};
                vector_3d_t v_ac = {0};
                vsub(&v_ab, &entry->b, &entry->a);
                vsub(&v_ac, &entry->c, &entry->a);

                // will actual normalize in next step.
                vector_3d_t s = {0};
                cross(&s, &v_ab, &v_ac);
                entry->normal = s;
            }
            normalize(&entry->normal);
        }
        printf("facet %zu:\n", i);
        print_facet(entry);
    }
    fclose(boundary_fp);

    make_stl_voxel_grid(&voxel_grid, loaded_facets, num_loaded_facets, 3.0 * h);
    // make_stl_voxel_grid(&voxel_grid, loaded_facets, num_loaded_facets, 5.0 * boundary_thickness);
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

static void cross(vector_3d_t * const result,
                  const vector_3d_t * const a,
                  const vector_3d_t * const b)
{
    result->x =  ((a->y * b->z) - (a->z * b->y));
    result->y = -((a->x * b->z) - (a->z * b->x));
    result->z =  ((a->x * b->y) - (a->y * b->x));
}

static void vsub(vector_3d_t * const result,
                 const vector_3d_t * const a,
                 const vector_3d_t * const b)
{
    result->x = a->x - b->x;
    result->y = a->y - b->y;
    result->z = a->z - b->z;
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

// This function will only work if p is in the plane given by a, b, c. This
// should be okay since we'll use the boundary layer check first to get the
// distance, which can use to project down into this plane outside this
// function.
static bool is_projected_point_in_triangle_3d(const vector_3d_t * const p,
                                              const vector_3d_t * const a,
                                              const vector_3d_t * const b,
                                              const vector_3d_t * const c)
{
    // vectors from P to x.
    vector_3d_t v_pa = {0};
    vector_3d_t v_pb = {0};
    vector_3d_t v_pc = {0};
    vsub(&v_pa, a, p);
    vsub(&v_pb, b, p);
    vsub(&v_pc, c, p);

    // cross products for various sub triangles to get normals
    vector_3d_t s_ab = {0};
    vector_3d_t s_bc = {0};
    vector_3d_t s_ca = {0};
    cross(&s_ab, &v_pa, &v_pb);
    cross(&s_bc, &v_pb, &v_pc);
    cross(&s_ca, &v_pc, &v_pa);

    const bool s1 = dot(&s_ab, &s_bc) > 0;
    const bool s2 = dot(&s_ca, &s_bc) > 0;
    return (s1 && s2);
}

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

enum stl_parser_expect_states {
    EXPECT_HEADER,
    EXPECT_FACET,
    EXPECT_LOOP,
    EXPECT_VERTEX,
    EXPECT_ENDLOOP,
    EXPECT_ENDFACET,
    AT_CAPACITY,
};

static void parse_vector_3d(vector_3d_t *v, const char *s)
{
    assert(v != NULL);
    assert(s != NULL);

    const char *delim = " \t";
    size_t component = 0;
    char s_copy[1024] = {0};
    strncpy(s_copy, s, sizeof(s_copy));
    char *tok = strtok(s_copy, delim);
    while (tok != NULL) {
        if (component < 3) {
            double value = 0;
            int result = sscanf(tok, "%lg", &value);
            assert(result == 1);
            switch(component) {
                case 0: v->x = value; break;
                case 1: v->y = value; break;
                case 2: v->z = value; break;
            }
        }
        component++;
        tok = strtok(NULL, delim);
    }
}

static bool parse_stl_file(stl_facet_t *facets, size_t *num_facets, FILE *fp)
{
    assert(facets != NULL);
    assert(num_facets != NULL);
    assert(fp != NULL);

    char line[256] = {0};
    size_t vertex_number = 0;

    const size_t capacity = *num_facets;
    *num_facets = 0;

    enum stl_parser_expect_states state = EXPECT_HEADER;
    while (fgets(line, sizeof(line), fp) != NULL) {
        printf("LINE: %s", line);
        printf("STATE: %d\n", state);
        stl_facet_t * const entry = &facets[*num_facets];
        switch (state) {
            case EXPECT_HEADER: {
                if (strncmp(line, "solid", 5) == 0) {
                    state = EXPECT_FACET;
                }
            } break;
            case EXPECT_FACET: {
                if (strncmp(line, "facet normal ", 13) == 0) {
                    char *nx_start = &line[13];
                    parse_vector_3d(&entry->normal, nx_start);
                    state = EXPECT_LOOP;
                }
            } break;
            case EXPECT_LOOP: {
                size_t skip = 0;
                while ((line[skip] != 0) && isspace(line[skip])) {
                    skip++;
                }
                if (strncmp(&line[skip], "outer loop", 10) == 0) {
                    vertex_number = 0;
                    state = EXPECT_VERTEX;
                }
            } break;
            case EXPECT_VERTEX: {
                size_t skip = 0;
                while ((line[skip] != 0) && isspace(line[skip])) {
                    skip++;
                }
                if (strncmp(&line[skip], "vertex ", 7) == 0) {
                    vector_3d_t v = {0};
                    parse_vector_3d(&v, &line[skip+7]);
                    switch (vertex_number) {
                        case 0: entry->a = v; break;
                        case 1: entry->b = v; break;
                        case 2: entry->c = v; break;
                    };
                    vertex_number++;

                    if (vertex_number >= 3) {
                        state = EXPECT_ENDLOOP;
                    }
                }
            } break;
            case EXPECT_ENDLOOP: {
                size_t skip = 0;
                while ((line[skip] != 0) && isspace(line[skip])) {
                    skip++;
                }
                if (strncmp(&line[skip], "endloop", 7) == 0) {
                    state = EXPECT_ENDFACET;
                }
            } break;
            case EXPECT_ENDFACET: {
                if (strncmp(line, "endfacet", 8) == 0) {
                    (*num_facets)++;
                    if (*num_facets >= capacity) {
                        state = AT_CAPACITY;
                    } else {
                        state = EXPECT_FACET;
                    }
                }
            } break;
            case AT_CAPACITY: break;
        }
        // TODO: Should check for an "endsolid" line...
    }

    return true;
}

static void print_vector(const vector_3d_t * const v)
{
    printf("{%.17g, %.17g, %.17g}\n", v->x, v->y, v->z);
}

static void print_facet(const stl_facet_t * const facet)
{
    printf("  normal = ");
    print_vector(&facet->normal);
    printf("  a = ");
    print_vector(&facet->a);
    printf("  b = ");
    print_vector(&facet->b);
    printf("  c = ");
    print_vector(&facet->c);
    printf("  sticky = %s\n", facet->sticky ? "yes" : "no");
    printf("  thickness = %g\n", facet->thickness);
}

static void update_triangle_list(uint64_t **list, uint64_t index_to_add)
{
    if (*list == NULL) {
        *list = static_cast<uint64_t *>(calloc(2, sizeof(*list[0])));
        assert(*list != NULL);
        // first element is number of entries that follow.
        (*list)[0] = 1;
        (*list)[1] = index_to_add;
    } else {
        assert(*list != NULL);
        const size_t capacity = (*list)[0];

        uint64_t *new_list = static_cast<uint64_t *>(calloc(capacity + 2, sizeof(*list[0])));
        assert(new_list != NULL);

        new_list[0] = capacity + 1;
        for (size_t i = 0; i < capacity; ++i) {
            new_list[i + 1] = (*list)[i + 1];
        }
        new_list[capacity + 1] = index_to_add;
        free(*list);
        *list = new_list;
    }
}

static size_t sdf_index_from_spans(const stl_voxel_grid_t *vgrid, size_t i, size_t j, size_t k)
{
    return (i * vgrid->jspan * vgrid->kspan) + (j * vgrid->kspan) + k;
}

static size_t sdf_cell_index_from_spans(const stl_voxel_grid_t *vgrid, size_t i, size_t j, size_t k)
{
    return (i * (vgrid->jspan - 1) * (vgrid->kspan - 1)) + (j * (vgrid->kspan - 1)) + k;
}

static void get_stl_facet_bounds(vector_3d_t *vmin, vector_3d_t *vmax, const stl_facet_t * const facet)
{
    *vmin = (vector_3d_t){0};
    *vmax = (vector_3d_t){0};

    vmax->x = std::max(std::max(std::max(facet->a.x, facet->b.x), facet->c.x), vmax->x);
    vmax->y = std::max(std::max(std::max(facet->a.y, facet->b.y), facet->c.y), vmax->y);
    vmax->z = std::max(std::max(std::max(facet->a.z, facet->b.z), facet->c.z), vmax->z);
    vmin->x = std::min(std::min(std::min(facet->a.x, facet->b.x), facet->c.x), vmin->x);
    vmin->y = std::min(std::min(std::min(facet->a.y, facet->b.y), facet->c.y), vmin->y);
    vmin->z = std::min(std::min(std::min(facet->a.z, facet->b.z), facet->c.z), vmin->z);
}

static void make_stl_voxel_grid(stl_voxel_grid_t *vgrid, const stl_facet_t * const facets, size_t num_facets, double spacing)
{
    vector_3d_t vmax = {0};
    vector_3d_t vmin = {0};
    // TODO: Fix slack for facet thickness
    const double border = 6.0 * spacing;
    for (size_t i = 0; i < num_facets; ++i) {
        const stl_facet_t * const entry = &facets[i];
        vmax.x = std::max(std::max(std::max(entry->a.x, entry->b.x), entry->c.x), vmax.x);
        vmax.y = std::max(std::max(std::max(entry->a.y, entry->b.y), entry->c.y), vmax.y);
        vmax.z = std::max(std::max(std::max(entry->a.z, entry->b.z), entry->c.z), vmax.z);
        vmin.x = std::min(std::min(std::min(entry->a.x, entry->b.x), entry->c.x), vmin.x);
        vmin.y = std::min(std::min(std::min(entry->a.y, entry->b.y), entry->c.y), vmin.y);
        vmin.z = std::min(std::min(std::min(entry->a.z, entry->b.z), entry->c.z), vmin.z);
    }

    print_vector(&vmin);
    print_vector(&vmax);

    vgrid->dx = spacing;
    vgrid->dy = spacing;
    vgrid->dz = spacing;

    vgrid->origin = (vector_3d_t) {
        vmin.x - border,
        vmin.y - border,
        vmin.z - border,
    };

    // May be a bit wasteful, can use fmod if we care enough.
    vgrid->ispan = 1 + ((vmax.x - vmin.x + 2 * border) / vgrid->dx);
    vgrid->jspan = 1 + ((vmax.y - vmin.y + 2 * border) / vgrid->dy);
    vgrid->kspan = 1 + ((vmax.z - vmin.z + 2 * border) / vgrid->dz);

    vgrid->extreme = (vector_3d_t) {
        vgrid->origin.x + (vgrid->dx * vgrid->ispan),
        vgrid->origin.y + (vgrid->dy * vgrid->jspan),
        vgrid->origin.z + (vgrid->dz * vgrid->kspan),
    };

    const size_t num_sdf_values = vgrid->ispan * vgrid->jspan * vgrid->kspan;
    const size_t num_bytes = num_sdf_values * sizeof(vgrid->sdf_values[0]);

    printf("Allocating %zu bytes for STL voxel grid (%zu grid points).\n", num_bytes, num_sdf_values);
    vgrid->sdf_values = static_cast<double *>(calloc(num_sdf_values, sizeof(vgrid->sdf_values[0])));
    vgrid->sticky_bits = static_cast<bool *>(calloc(num_sdf_values, sizeof(vgrid->sticky_bits[0])));
    vgrid->triangle_lists = static_cast<uint64_t **>(calloc(num_sdf_values, sizeof(vgrid->triangle_lists[0])));
    vgrid->cell_centered_triangle_lists = static_cast<uint64_t **>(calloc(num_sdf_values, sizeof(vgrid->cell_centered_triangle_lists[0])));

    if (vgrid->sdf_values == NULL) {
        fprintf(stderr, "Unable to allocate memory for STL voxel grid, wanted %zu bytes.\n", num_bytes);
    }

    if (vgrid->sticky_bits == NULL) {
        fprintf(stderr, "Unable to allocate memory for STL voxel grid sticky bits, wanted %zu bytes.\n", num_sdf_values * sizeof(vgrid->sticky_bits[0]));
    }

    for (size_t i = 0; i < num_sdf_values; ++i) {
        vgrid->sdf_values[i] = DBL_MAX;
    }

    for (size_t i = 0; i < vgrid->ispan; ++i) {
        const double xpx = vgrid->origin.x + (i * vgrid->dx);
        for (size_t j = 0; j < vgrid->jspan; ++j) {
            const double xpy = vgrid->origin.y + (j * vgrid->dy);
            for (size_t k = 0; k < vgrid->kspan; ++k) {
                const double xpz = vgrid->origin.z + (k * vgrid->dz);
                const vector_3d_t xp = {
                    .x = xpx,
                    .y = xpy,
                    .z = xpz,
                };
                const size_t sdf_index = sdf_index_from_spans(vgrid, i, j, k);

                // TODO: Can make this more efficient by swapping iteration
                // order with the grid, letting each triangle only apply to the
                // few points around it.
                for (size_t m = 0; m < num_facets; ++m) {
                    const stl_facet_t * const entry = &facets[m];
                    vector_3d_t vfmin = {0};
                    vector_3d_t vfmax = {0};
                    get_stl_facet_bounds(&vfmin, &vfmax, entry);
                    // TODO: A more expensive check here will reduce the amount of work later.
                    if (((vfmin.x - vgrid->dx) < xpx) && (xpx <= (vfmax.x + vgrid->dx)) &&
                        ((vfmin.y - vgrid->dy) < xpy) && (xpy <= (vfmax.y + vgrid->dy)) &&
                        ((vfmin.z - vgrid->dz) < xpz) && (xpz <= (vfmax.z + vgrid->dz))) {
                        update_triangle_list(&vgrid->triangle_lists[sdf_index], m);
                    }
                }

/*
                for (size_t m = 0; m < num_facets; ++m) {
                    const stl_facet_t * const entry = &facets[m];
                    const double facet_border = 5.0 * entry->thickness;
                    double s = 0;
                    stl_facet_distance(&s, &xp, entry);
                    if (s <= 0.0 && s <= facet_border) {
                        // *strength = clamp_unity((1.0 - (min_d / loaded_facets[min_wall_index].thickness)) / 0.9);
                        vgrid->sdf_values[sdf_index] = std::min(vgrid->sdf_values[sdf_index], s / entry->thickness);
                        if (s <= 1.0) {
                            vgrid->sticky_bits[sdf_index] |= entry->sticky;
                        }
                    }
                }
*/
            }
        }
        printf("%06zu / %06zu planes computed.\n", i, vgrid->ispan);
    }

    for (size_t i = 0; i < (vgrid->ispan - 1); ++i) {
        for (size_t j = 0; j < (vgrid->jspan - 1); ++j) {
            for (size_t k = 0; k < (vgrid->kspan - 1); ++k) {
                const uint64_t * const triangle_lists[] = {
                    vgrid->triangle_lists[sdf_index_from_spans(vgrid, i, j, k)],
                    vgrid->triangle_lists[sdf_index_from_spans(vgrid, i, j, k+1)],
                    vgrid->triangle_lists[sdf_index_from_spans(vgrid, i, j+1, k)],
                    vgrid->triangle_lists[sdf_index_from_spans(vgrid, i+1, j, k)],
                    vgrid->triangle_lists[sdf_index_from_spans(vgrid, i, j+1, k+1)],
                    vgrid->triangle_lists[sdf_index_from_spans(vgrid, i+1, j+1, k)],
                    vgrid->triangle_lists[sdf_index_from_spans(vgrid, i+1, j, k+1)],
                    vgrid->triangle_lists[sdf_index_from_spans(vgrid, i+1, j+1, k+1)],
                };
                uint64_t tl_index[DIM(triangle_lists)] = {0};
                const uint64_t tl_capacity[] = {
                    (triangle_lists[0] == NULL) ? 0 : triangle_lists[0][0],
                    (triangle_lists[1] == NULL) ? 0 : triangle_lists[1][0],
                    (triangle_lists[2] == NULL) ? 0 : triangle_lists[2][0],
                    (triangle_lists[3] == NULL) ? 0 : triangle_lists[3][0],
                    (triangle_lists[4] == NULL) ? 0 : triangle_lists[4][0],
                    (triangle_lists[5] == NULL) ? 0 : triangle_lists[5][0],
                    (triangle_lists[6] == NULL) ? 0 : triangle_lists[6][0],
                    (triangle_lists[7] == NULL) ? 0 : triangle_lists[7][0],
                };

                const size_t sdf_cell_index = sdf_cell_index_from_spans(vgrid, i, j, k);

                while (
                    (tl_index[0] < tl_capacity[0]) ||
                    (tl_index[1] < tl_capacity[1]) ||
                    (tl_index[2] < tl_capacity[2]) ||
                    (tl_index[3] < tl_capacity[3]) ||
                    (tl_index[4] < tl_capacity[4]) ||
                    (tl_index[5] < tl_capacity[5]) ||
                    (tl_index[6] < tl_capacity[6]) ||
                    (tl_index[7] < tl_capacity[7])
                ) {

                    uint64_t minidx = UINT64_MAX;
                    for (size_t m = 0; m < DIM(triangle_lists); ++m) {
                        if (tl_index[m] < tl_capacity[m]) {
                            const uint64_t candidate = triangle_lists[m][tl_index[m] + 1];
                            if (candidate < minidx) {
                                minidx = candidate;
                            }
                        }
                    }

                    assert(minidx != UINT64_MAX);
                    update_triangle_list(&vgrid->cell_centered_triangle_lists[sdf_cell_index], minidx);

                    for (size_t m = 0; m < DIM(triangle_lists); ++m) {
                        if (tl_index[m] < tl_capacity[m]) {
                            const uint64_t candidate = triangle_lists[m][tl_index[m] + 1];
                            if (candidate == minidx) {
                                tl_index[m]++;
                            }
                        }
                    }
                }
            }
        }
        printf("%06zu / %06zu cell planes computed.\n", i, vgrid->ispan - 1);
    }

    {
        char filename[256] = {0};
        const char dir[] = "/Crucial2TB/sdunatunga";
        snprintf(filename, sizeof(filename), "%s/lmp_%p.csv", dir, vgrid->sdf_values);
        FILE *g = fopen(filename, "w");
        fprintf(g, "x,y,z,s,n\n");
        for (size_t i = 0; i < vgrid->ispan; ++i) {
            const double xpx = vgrid->origin.x + (i * vgrid->dx);
            for (size_t j = 0; j < vgrid->jspan; ++j) {
                const double xpy = vgrid->origin.y + (j * vgrid->dy);
                for (size_t k = 0; k < vgrid->kspan; ++k) {
                    const double xpz = vgrid->origin.z + (k * vgrid->dz);
                    const vector_3d_t xp = {
                        .x = xpx,
                        .y = xpy,
                        .z = xpz,
                    };
                    const size_t sdf_index = sdf_index_from_spans(vgrid, i, j, k);
                    const size_t n = (vgrid->triangle_lists[sdf_index] == NULL) ? 0 : vgrid->triangle_lists[sdf_index][0];
                    fprintf(g, "%.17g,%.17g,%.17g,%.17g,%zu\n", xpx, xpy, xpz, vgrid->sdf_values[sdf_index], n);
                }
            }
            printf("%06zu / %06zu planes written to disk at %s.\n", i, vgrid->ispan, filename);
        }
        fclose(g);
    }

    {
        char filename[256] = {0};
        const char dir[] = "/Crucial2TB/sdunatunga";
        snprintf(filename, sizeof(filename), "%s/lmpcell_%p.csv", dir, vgrid->cell_centered_triangle_lists);
        FILE *g = fopen(filename, "w");
        fprintf(g, "x,y,z,s,n\n");
        for (size_t i = 0; i < vgrid->ispan - 1; ++i) {
            const double xpx = vgrid->origin.x + (i * vgrid->dx);
            for (size_t j = 0; j < vgrid->jspan - 1; ++j) {
                const double xpy = vgrid->origin.y + (j * vgrid->dy);
                for (size_t k = 0; k < vgrid->kspan - 1; ++k) {
                    const double xpz = vgrid->origin.z + (k * vgrid->dz);
                    const vector_3d_t xp = {
                        .x = xpx + vgrid->dx / 2.0,
                        .y = xpy + vgrid->dy / 2.0,
                        .z = xpz + vgrid->dz / 2.0,
                    };
                    const size_t sdf_cell_index = sdf_cell_index_from_spans(vgrid, i, j, k);
                    const size_t n = (vgrid->cell_centered_triangle_lists[sdf_cell_index] == NULL) ? 0 : vgrid->cell_centered_triangle_lists[sdf_cell_index][0];
                    if (n > 0) {
                        fprintf(g, "%.17g,%.17g,%.17g,%zu\n", xp.x, xp.y, xp.z, n);
                        fprintf(g, "#");
                        for (size_t m = 0; m < n; ++m) {
                            fprintf(g, " %zu", vgrid->cell_centered_triangle_lists[sdf_cell_index][m + 1]);
                        }
                        fprintf(g, "\n");
                    }
                }
            }
            printf("%06zu / %06zu cell planes written to disk at %s.\n", i, vgrid->ispan - 1, filename);
        }
        fclose(g);
    }

    print_vector(&(vgrid->origin));
    print_vector(&(vgrid->extreme));

    // vgrid->extreme = (vector_3d_t) {
    //     vmax.x + border,
    //     vmax.y + border,
    //     vmax.z + border,
    // };
}

static void sdf_and_normal_from_vgrid_old(double *sdf, vector_3d_t *normal, bool *sticky, const stl_voxel_grid_t * const vgrid, const vector_3d_t * const xp)
{
    if (
        (xp->x < vgrid->origin.x) || (xp->x > vgrid->extreme.x) ||
        (xp->y < vgrid->origin.y) || (xp->y > vgrid->extreme.y) ||
        (xp->z < vgrid->origin.z) || (xp->z > vgrid->extreme.z)
    ) {
        return;
    }

    const size_t i = (xp->x - vgrid->origin.x) / vgrid->dx;
    const size_t j = (xp->y - vgrid->origin.y) / vgrid->dy;
    const size_t k = (xp->z - vgrid->origin.z) / vgrid->dz;
    const double wip = (((xp->x - vgrid->origin.x) / vgrid->dx) - i);
    const double wjp = (((xp->y - vgrid->origin.y) / vgrid->dy) - j);
    const double wkp = (((xp->z - vgrid->origin.z) / vgrid->dz) - k);
    const double wi = 1.0 - wip;
    const double wj = 1.0 - wjp;
    const double wk = 1.0 - wkp;

    double s[] = {
        vgrid->sdf_values[sdf_index_from_spans(vgrid, i, j, k)],
        vgrid->sdf_values[sdf_index_from_spans(vgrid, i, j, k+1)],
        vgrid->sdf_values[sdf_index_from_spans(vgrid, i, j+1, k)],
        vgrid->sdf_values[sdf_index_from_spans(vgrid, i+1, j, k)],
        vgrid->sdf_values[sdf_index_from_spans(vgrid, i, j+1, k+1)],
        vgrid->sdf_values[sdf_index_from_spans(vgrid, i+1, j+1, k)],
        vgrid->sdf_values[sdf_index_from_spans(vgrid, i+1, j, k+1)],
        vgrid->sdf_values[sdf_index_from_spans(vgrid, i+1, j+1, k+1)],
    };

    // Make s = 1 the boundary.
    bool allzero = true;
    for (size_t i = 0; i < DIM(s); ++i) {
        s[i] = clamp_unity(1.0 - s[i]);
        if (s[i] != 0.0) {
            allzero = false;
        }
    }

    if (allzero) {
        *sdf = 0.0;
        return;
    }

    const bool b[] = {
        vgrid->sticky_bits[sdf_index_from_spans(vgrid, i, j, k)],
        vgrid->sticky_bits[sdf_index_from_spans(vgrid, i, j, k+1)],
        vgrid->sticky_bits[sdf_index_from_spans(vgrid, i, j+1, k)],
        vgrid->sticky_bits[sdf_index_from_spans(vgrid, i+1, j, k)],
        vgrid->sticky_bits[sdf_index_from_spans(vgrid, i, j+1, k+1)],
        vgrid->sticky_bits[sdf_index_from_spans(vgrid, i+1, j+1, k)],
        vgrid->sticky_bits[sdf_index_from_spans(vgrid, i+1, j, k+1)],
        vgrid->sticky_bits[sdf_index_from_spans(vgrid, i+1, j+1, k+1)],
    };

    *sticky = false;
    for (size_t i = 0; i < DIM(b); ++i) {
        *sticky |= b[i];
    }

    const double sp = (
        wi * wj * wk * s[0] +
        wi * wj * wkp * s[1] +
        wi * wjp * wk * s[2] +
        wip * wj * wk * s[3] +
        wi * wjp * wkp * s[4] +
        wip * wjp * wk * s[5] +
        wip * wj * wkp * s[6] +
        wip * wjp * wkp * s[7]
    );

    *sdf = sp;

    const double dspdx = (
        -wj * wk * s[0] +
        -wj * wkp * s[1] +
        -wjp * wk * s[2] +
        wj * wk * s[3] +
        -wjp * wkp * s[4] +
        wjp * wk * s[5] +
        wj * wkp * s[6] +
        wjp * wkp * s[7]
    );

    const double dspdy = (
        -wi * wk * s[0] +
        -wi * wkp * s[1] +
        wi * wk * s[2] +
        -wip * wk * s[3] +
        wi * wkp * s[4] +
        wip * wk * s[5] +
        -wip * wkp * s[6] +
        wip * wkp * s[7]
    );

    const double dspdz = (
        -wi * wj * s[0] +
        wi * wj * s[1] +
        -wi * wjp * s[2] +
        -wip * wj * s[3] +
        wi * wjp * s[4] +
        -wip * wjp * s[5] +
        wip * wj * s[6] +
        wip * wjp * s[7]
    );

    // we flipped the sign of s, so we need to flip this as well to get the
    // right normal.
    normal->x = -dspdx;
    normal->y = -dspdy;
    normal->z = -dspdz;

    normalize(normal);
}

static void sdf_and_normal_from_vgrid(double *sdf, vector_3d_t *normal, bool *sticky, const stl_voxel_grid_t * const vgrid, const vector_3d_t * const xp)
{
    if (
        (xp->x < vgrid->origin.x) || (xp->x > vgrid->extreme.x) ||
        (xp->y < vgrid->origin.y) || (xp->y > vgrid->extreme.y) ||
        (xp->z < vgrid->origin.z) || (xp->z > vgrid->extreme.z)
    ) {
        return;
    }

    const size_t i = (xp->x - vgrid->origin.x) / vgrid->dx;
    const size_t j = (xp->y - vgrid->origin.y) / vgrid->dy;
    const size_t k = (xp->z - vgrid->origin.z) / vgrid->dz;
    const double wip = (((xp->x - vgrid->origin.x) / vgrid->dx) - i);
    const double wjp = (((xp->y - vgrid->origin.y) / vgrid->dy) - j);
    const double wkp = (((xp->z - vgrid->origin.z) / vgrid->dz) - k);
    const double wi = 1.0 - wip;
    const double wj = 1.0 - wjp;
    const double wk = 1.0 - wkp;

    const uint64_t * const triangle_list =
        vgrid->cell_centered_triangle_lists[sdf_cell_index_from_spans(vgrid, i, j, k)];

    if (triangle_list == NULL) {
        *sdf = 0.0;
        return;
    }

    const size_t num_triangles_in_cell_list = static_cast<size_t>(triangle_list[0]);

    static double *distances = NULL;
    static size_t capacity_distances = 0;
    if (distances == NULL) {
        capacity_distances = num_triangles_in_cell_list;
        distances = static_cast<double *>(calloc(capacity_distances, sizeof(distances[0])));
    }

    if (capacity_distances < num_triangles_in_cell_list) {
        free(distances);
        capacity_distances = num_triangles_in_cell_list;
        distances = static_cast<double *>(calloc(capacity_distances, sizeof(distances[0])));
    }

    // Now this just looks like the old style of problem but checking against a
    // smaller number of triangles.
    {
        // Take the function f = prod(d_1, d_2, ...) (d_i distance from boundary i).
        // The gradient gives us a nice direction, which can be written as
        // sum_over_i(n_i * prod_for_j_not_equal_i(d_j)) where n_i is the normal to
        // the boundary segment.

        bool any_wall_detected = false;
        for (size_t i = 0; i < num_triangles_in_cell_list; ++i) {
            const size_t bi = triangle_list[i + 1];
            const stl_facet_t * const entry = &loaded_facets[bi];
            stl_facet_distance(&distances[i], xp, entry);

            // Wrong side of the BC, set back to a far away value.
            if (distances[i] < 0.0) {
                distances[i] = DBL_MAX;
            }

            if (distances[i] < entry->thickness) {
                any_wall_detected = true;
            }
        }

        // At least one wall detected.
        if (any_wall_detected) {
            double min_d = distances[0];
            size_t min_wall_index = 0;
            for (size_t i = 0; i < num_triangles_in_cell_list; ++i) {
                const size_t bi = triangle_list[i + 1];
                const stl_facet_t * const entry = &loaded_facets[bi];
                if ((distances[i] < entry->thickness) && (distances[i] < min_d)) {
                    min_d = distances[i];
                    min_wall_index = bi;
                }
            }
            // Do we want min distance, or max strength BC?
            // *strength = clamp_unity(1.0 - (min_d / loaded_facets[min_wall_index].thickness));
            // Clip the last closest tenth so there's a bit of dead zone where the
            // BC is at full strength.
            *sdf = clamp_unity((1.0 - (min_d / loaded_facets[min_wall_index].thickness)) / 0.9);
        } else {
            *sdf = 0.0;
        }

        if (*sdf != 0.0) {
            *normal = (vector_3d_t) {
                0.0,
                0.0,
                0.0
            };
            for (size_t i = 0; i < num_triangles_in_cell_list; ++i) {
                double pi_d = 1.0;
                const size_t bi = triangle_list[i + 1];
                for (size_t j = 0; j < num_triangles_in_cell_list; ++j) {
                    const size_t bj = triangle_list[j + 1];
                    const stl_facet_t * const entry = &loaded_facets[bj];
                    // Not the current wall we're considering (chain rule), and
                    // product of all other walls that are within range.
                    if ((j != i) && (distances[j] < entry->thickness)) {
                        // distances should all be positive by this point, so fabs is
                        // unnecessary...
                        // pi_d *= fabs(distances[j]);
                        pi_d *= distances[j];
                    }
                }

                normal->x += pi_d * loaded_facets[bi].normal.x;
                normal->y += pi_d * loaded_facets[bi].normal.y;
                normal->z += pi_d * loaded_facets[bi].normal.z;
            }

            normalize(normal);


            bool is_any_wall_sticky = false;
            for (size_t i = 0; i < num_triangles_in_cell_list; ++i) {
                const size_t bi = triangle_list[i + 1];
                const stl_facet_t * const entry = &loaded_facets[bi];
                // Check if we are in contact with a sticky wall.
                if (distances[i] < entry->thickness) {
                    is_any_wall_sticky |= entry->sticky;
                }
            }
            *sticky = is_any_wall_sticky;
        }
    }
}

static void bc_setup(void)
{
/*
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
*/
}

static void stl_facet_distance(double *distance, const vector_3d_t * const xp, const stl_facet_t * const facet)
{
    // set outputs
    *distance = DBL_MAX;

    const vector_3d_t v = {
        .x = xp->x - facet->a.x,
        .y = xp->y - facet->a.y,
        .z = xp->z - facet->a.z,
    };

    const double x3 = dot(&v, &facet->normal);
    if (0.0 <= x3 && x3 <= 5.0 * facet->thickness) {
        const vector_3d_t p = {
            .x = xp->x - (x3 * facet->normal.x),
            .y = xp->y - (x3 * facet->normal.y),
            .z = xp->z - (x3 * facet->normal.z),
        };
        if (is_projected_point_in_triangle_3d(&p, &facet->a, &facet->b, &facet->c)) {
            // const double x3 = dot(&v, &facet->normal);
            *distance = x3;
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

static void clear_wall_bitset(uint64_t *wall_bitset)
{
    *wall_bitset = 0;
}

static bool is_wall_close(uint64_t wall_bitset, size_t wall_index)
{
    return ((wall_bitset & (UINT64_C(1) << wall_index)) != 0);
}

static void set_wall_bitset(uint64_t *wall_bitset, size_t wall_index)
{
    *wall_bitset |= (UINT64_C(1) << wall_index);
}

#if 0
static void boundary_force_direction_from_levelset(double *strength,
                                                   vector_3d_t *direction,
                                                   uint64_t *walls_bitset,
                                                   const vector_3d_t * const xp)
                                                   // ,
                                                   // const sd_boundary_3d_t * const boundaries,
                                                   // size_t num_boundaries)
{
    // Take the function f = prod(d_1, d_2, ...) (d_i distance from boundary i).
    // The gradient gives us a nice direction, which can be written as
    // sum_over_i(n_i * prod_for_j_not_equal_i(d_j)) where n_i is the normal to
    // the boundary segment.
    static double distances[DIM(loaded_facets)] = {0};
    // already part of the boundary!
    // static vector_3d_t normals[DIM(b3)] = {0};

    uint64_t wb = 0;
    for (size_t bi = 0; bi < num_loaded_facets; ++bi) {
        const stl_facet_t * const entry = &loaded_facets[bi];
        stl_facet_distance(&distances[bi], xp, entry);

        // Wrong side of the BC, set back to a far away value.
        if (distances[bi] < 0.0) {
            distances[bi] = DBL_MAX;
        }

        if (distances[bi] < entry->thickness) {
            set_wall_bitset(&wb, bi);
        }
    }

    *walls_bitset = wb;

    // At least one wall detected.
    if (wb != 0) {
        double min_d = distances[0];
        size_t min_wall_index = 0;
        for (size_t i = 0; i < num_loaded_facets; ++i) {
            if (is_wall_close(wb, i) && (distances[i] < min_d)) {
                min_d = distances[i];
                min_wall_index = i;
            }
        }
        // Do we want min distance, or max strength BC?
        // *strength = clamp_unity(1.0 - (min_d / loaded_facets[min_wall_index].thickness));
        // Clip the last closest tenth so there's a bit of dead zone where the
        // BC is at full strength.
        *strength = clamp_unity((1.0 - (min_d / loaded_facets[min_wall_index].thickness)) / 0.9);
    } else {
        *strength = 0.0;
    }

    if (*strength != 0.0) {
        *direction = (vector_3d_t) {
            0.0,
            0.0,
            0.0
        };
        for (size_t i = 0; i < num_loaded_facets; ++i) {
            double pi_d = 1.0;
            for (size_t j = 0; j < num_loaded_facets; ++j) {
                // Not the current wall we're considering (chain rule), and
                // product of all other walls that are within range.
                if ((j != i) && is_wall_close(wb, j)) {
                    // distances should all be positive by this point, so fabs is
                    // unnecessary...
                    // pi_d *= fabs(distances[j]);
                    pi_d *= distances[j];
                }
            }

            direction->x += pi_d * loaded_facets[i].normal.x;
            direction->y += pi_d * loaded_facets[i].normal.y;
            direction->z += pi_d * loaded_facets[i].normal.z;
        }

        normalize(direction);
    }
}
#endif

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
        bool is_any_wall_sticky = false;
        uint64_t walls_bitset = 0;
        sdf_and_normal_from_vgrid(&s, &fdir, &is_any_wall_sticky, &voxel_grid, &xp);
        // uint64_t walls_bitset = 0;
        // boundary_force_direction_from_levelset(&s, &fdir, &walls_bitset, &xp);
        // bool is_any_wall_sticky = false;
        // for (size_t i = 0; i < num_loaded_facets; ++i) {
        //     const stl_facet_t * const entry = &loaded_facets[i];
        //     // Check if we are in contact with a sticky wall.
        //     if (is_wall_close(walls_bitset, i)) {
        //         is_any_wall_sticky |= entry->sticky;
        //     }
        // }

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
