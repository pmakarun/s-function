#ifndef SFUN_HEADER_H
# define SFUN_HEADER_H


# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus


/*******************
* OSQP Versioning *
*******************/
# define OSQP_VERSION ("0.6.0") /* string literals automatically null-terminated
                                   */

/******************
* Solver Status  *
******************/
# define OSQP_DUAL_INFEASIBLE_INACCURATE (4)
# define OSQP_PRIMAL_INFEASIBLE_INACCURATE (3)
# define OSQP_SOLVED_INACCURATE (2)
# define OSQP_SOLVED (1)
# define OSQP_MAX_ITER_REACHED (-2)
# define OSQP_PRIMAL_INFEASIBLE (-3)    /* primal infeasible  */
# define OSQP_DUAL_INFEASIBLE (-4)      /* dual infeasible */
# define OSQP_SIGINT (-5)               /* interrupted by user */
# ifdef PROFILING
#  define OSQP_TIME_LIMIT_REACHED (-6)
# endif // ifdef PROFILING
# define OSQP_NON_CVX (-7)              /* problem non convex */
# define OSQP_UNSOLVED (-10)            /* Unsolved. Only setup function has been called */


/*************************
* Linear System Solvers *
*************************/
enum linsys_solver_type { QDLDL_SOLVER, MKL_PARDISO_SOLVER };
extern const char * LINSYS_SOLVER_NAME[];


/******************
* Solver Errors  *
******************/
enum osqp_error_type {
    OSQP_DATA_VALIDATION_ERROR = 1,  /* Start errors from 1 */
    OSQP_SETTINGS_VALIDATION_ERROR,
    OSQP_LINSYS_SOLVER_LOAD_ERROR,
    OSQP_LINSYS_SOLVER_INIT_ERROR,
    OSQP_NONCVX_ERROR,
    OSQP_MEM_ALLOC_ERROR,
    OSQP_WORKSPACE_NOT_INIT_ERROR,
};
extern const char * OSQP_ERROR_MESSAGE[];


/**********************************
* Solver Parameters and Settings *
**********************************/

# define RHO (0.1)
# define SIGMA (1E-06)
# define MAX_ITER (4000)
# define EPS_ABS (1E-3)
# define EPS_REL (1E-3)
# define EPS_PRIM_INF (1E-4)
# define EPS_DUAL_INF (1E-4)
# define ALPHA (1.6)
# define LINSYS_SOLVER (QDLDL_SOLVER)

# define RHO_MIN (1e-06)
# define RHO_MAX (1e06)
# define RHO_EQ_OVER_RHO_INEQ (1e03)
# define RHO_TOL (1e-04) ///< tolerance for detecting if an inequality is set to equality


# ifndef EMBEDDED
#  define DELTA (1E-6)
#  define POLISH (0)
#  define POLISH_REFINE_ITER (3)
#  define VERBOSE (1)
# endif // ifndef EMBEDDED

# define SCALED_TERMINATION (0)
# define CHECK_TERMINATION (25)
# define WARM_START (1)
# define SCALING (10)

# define MIN_SCALING (1e-04) ///< minimum scaling value
# define MAX_SCALING (1e+04) ///< maximum scaling value


# ifndef OSQP_NULL
#  define OSQP_NULL 0
# endif /* ifndef OSQP_NULL */

# ifndef OSQP_NAN
#  define OSQP_NAN ((c_float)0x7fc00000UL)  // not a number
# endif /* ifndef OSQP_NAN */

# ifndef OSQP_INFTY
#  define OSQP_INFTY ((c_float)1e30)        // infinity
# endif /* ifndef OSQP_INFTY */


# if EMBEDDED != 1
#  define ADAPTIVE_RHO (1)
#  define ADAPTIVE_RHO_INTERVAL (0)
#  define ADAPTIVE_RHO_FRACTION (0.4)           ///< fraction of setup time after which we update rho
#  define ADAPTIVE_RHO_MULTIPLE_TERMINATION (4) ///< multiple of check_termination after which we update rho (if PROFILING disabled)
#  define ADAPTIVE_RHO_FIXED (100)              ///< number of iterations after which we update rho if termination_check  and PROFILING are disabled
#  define ADAPTIVE_RHO_TOLERANCE (5)            ///< tolerance for adopting new rho; minimum ratio between new rho and the current one
# endif // if EMBEDDED != 1

# ifdef PROFILING
#  define TIME_LIMIT (0)                        ///< Disable time limit as default
# endif // ifdef PROFILING

/* Printing */
# define PRINT_INTERVAL 200


/* DEBUG */
/* #undef DEBUG */

/* Operating system */
/* #undef IS_LINUX */
/* #undef IS_MAC */
#define IS_WINDOWS

/* EMBEDDED */
#define EMBEDDED (2)

/* PRINTING */
/* #undef PRINTING */

/* PROFILING */
/* #undef PROFILING */

/* CTRLC */
/* #undef CTRLC */

/* DFLOAT */
/* #undef DFLOAT */

/* DLONG */
#define DLONG

/* ENABLE_MKL_PARDISO */
/* #undef ENABLE_MKL_PARDISO */

/* MEMORY MANAGEMENT */
/* #undef OSQP_CUSTOM_MEMORY */
#ifdef OSQP_CUSTOM_MEMORY
#include ""
#endif


/*
   Define OSQP compiler flags
 */

// cmake generated compiler flags

/* DATA CUSTOMIZATIONS (depending on memory manager)-----------------------   */

// We do not need memory allocation functions if EMBEDDED is enabled
# ifndef EMBEDDED

/* define custom printfs and memory allocation (e.g. matlab/python) */
#  ifdef MATLAB
    #   include "mex.h"
static void* c_calloc(size_t num, size_t size) {
  void *m = mxCalloc(num, size);
  mexMakeMemoryPersistent(m);
  return m;
}

static void* c_malloc(size_t size) {
  void *m = mxMalloc(size);
  mexMakeMemoryPersistent(m);
  return m;
}

static void* c_realloc(void *ptr, size_t size) {
  void *m = mxRealloc(ptr, size);
  mexMakeMemoryPersistent(m);
  return m;
}

    #   define c_free mxFree
#  elif defined PYTHON
// Define memory allocation for python. Note that in Python 2 memory manager
// Calloc is not implemented
    #   include <Python.h>
    #   define c_malloc PyMem_Malloc
    #   if PY_MAJOR_VERSION >= 3
    #    define c_calloc PyMem_Calloc
    #   else  /* if PY_MAJOR_VERSION >= 3 */
static void* c_calloc(size_t num, size_t size) {
  void *m = PyMem_Malloc(num * size);
  memset(m, 0, num * size);
  return m;
}
    #   endif /* if PY_MAJOR_VERSION >= 3 */
    #   define c_free PyMem_Free
    #   define c_realloc PyMem_Realloc

# elif !defined OSQP_CUSTOM_MEMORY
/* If no custom memory allocator defined, use
 * standard linux functions. Custom memory allocator definitions
 * appear in the osqp_configure.h generated file. */
    #  include <stdlib.h>
    #  define c_malloc  malloc
    #  define c_calloc  calloc
    #  define c_free    free
    #  define c_realloc realloc
#  endif /* ifdef MATLAB */

# endif // end ifndef EMBEDDED


/* Use customized number representation -----------------------------------   */
# ifdef DLONG            // long integers
typedef long long c_int; /* for indices */
# else // standard integers
typedef int c_int;       /* for indices */
# endif /* ifdef DLONG */


# ifndef DFLOAT         // Doubles
typedef double c_float; /* for numerical values  */
# else                  // Floats
typedef float c_float;  /* for numerical values  */
# endif /* ifndef DFLOAT */


/* Use customized operations */

# ifndef c_absval
#  define c_absval(x) (((x) < 0) ? -(x) : (x))
# endif /* ifndef c_absval */

# ifndef c_max
#  define c_max(a, b) (((a) > (b)) ? (a) : (b))
# endif /* ifndef c_max */

# ifndef c_min
#  define c_min(a, b) (((a) < (b)) ? (a) : (b))
# endif /* ifndef c_min */

// Round x to the nearest multiple of N
# ifndef c_roundmultiple
#  define c_roundmultiple(x, N) ((x) + .5 * (N)-c_fmod((x) + .5 * (N), (N)))
# endif /* ifndef c_roundmultiple */


/* Use customized functions -----------------------------------------------   */

# if EMBEDDED != 1

#  include <math.h>
#  ifndef DFLOAT // Doubles
#   define c_sqrt sqrt
#   define c_fmod fmod
#  else          // Floats
#   define c_sqrt sqrtf
#   define c_fmod fmodf
#  endif /* ifndef DFLOAT */

# endif // end EMBEDDED


# ifdef PRINTING
#  include <stdio.h>
#  include <string.h>

#  ifdef MATLAB
#   define c_print mexPrintf

// The following trick slows down the performance a lot. Since many solvers
// actually
// call mexPrintf and immediately force print buffer flush
// otherwise messages don't appear until solver termination
// ugly because matlab does not provide a vprintf mex interface
// #include <stdarg.h>
// static int c_print(char *msg, ...)
// {
//   va_list argList;
//   va_start(argList, msg);
//   //message buffer
//   int bufferSize = 256;
//   char buffer[bufferSize];
//   vsnprintf(buffer,bufferSize-1, msg, argList);
//   va_end(argList);
//   int out = mexPrintf(buffer); //print to matlab display
//   mexEvalString("drawnow;");   // flush matlab print buffer
//   return out;
// }
#  elif defined PYTHON
#   include <Python.h>
#   define c_print PySys_WriteStdout
#  elif defined R_LANG
#   include <R_ext/Print.h>
#   define c_print Rprintf
#  else  /* ifdef MATLAB */
#   define c_print printf
#  endif /* ifdef MATLAB */

/* Print error macro */
#  define c_eprint(...) c_print("ERROR in %s: ", __FUNCTION__); c_print(\
    __VA_ARGS__); c_print("\n");

# endif  /* PRINTING */


/******************
* Internal types *
******************/

/**
 *  Matrix in compressed-column or triplet form
 */
typedef struct {
  c_int    nzmax; ///< maximum number of entries
  c_int    m;     ///< number of rows
  c_int    n;     ///< number of columns
  c_int   *p;     ///< column pointers (size n+1); col indices (size nzmax) start from 0 when using triplet format (direct KKT matrix formation)
  c_int   *i;     ///< row indices, size nzmax starting from 0
  c_float *x;     ///< numerical values, size nzmax
  c_int    nz;    ///< number of entries in triplet matrix, -1 for csc
} csc;

/**
 * Linear system solver structure (sublevel objects initialize it differently)
 */

typedef struct linsys_solver LinSysSolver;

/**
 * OSQP Timer for statistics
 */
typedef struct OSQP_TIMER OSQPTimer;

/**
 * Problem scaling matrices stored as vectors
 */
typedef struct {
  c_float  c;    ///< cost function scaling
  c_float *D;    ///< primal variable scaling
  c_float *E;    ///< dual variable scaling
  c_float  cinv; ///< cost function rescaling
  c_float *Dinv; ///< primal variable rescaling
  c_float *Einv; ///< dual variable rescaling
} OSQPScaling;

/**
 * Solution structure
 */
typedef struct {
  c_float *x; ///< primal solution
  c_float *y; ///< Lagrange multiplier associated to \f$l <= Ax <= u\f$
} OSQPSolution;


/**
 * Solver return information
 */
typedef struct {
  c_int iter;          ///< number of iterations taken
  char  status[32];    ///< status string, e.g. 'solved'
  c_int status_val;    ///< status as c_int, defined in constants.h

# ifndef EMBEDDED
  c_int status_polish; ///< polish status: successful (1), unperformed (0), (-1) unsuccessful
# endif // ifndef EMBEDDED

  c_float obj_val;     ///< primal objective
  c_float pri_res;     ///< norm of primal residual
  c_float dua_res;     ///< norm of dual residual

# ifdef PROFILING
  c_float setup_time;  ///< time taken for setup phase (seconds)
  c_float solve_time;  ///< time taken for solve phase (seconds)
  c_float update_time; ///< time taken for update phase (seconds)
  c_float polish_time; ///< time taken for polish phase (seconds)
  c_float run_time;    ///< total time  (seconds)
# endif // ifdef PROFILING

# if EMBEDDED != 1
  c_int   rho_updates;  ///< number of rho updates
  c_float rho_estimate; ///< best rho estimate so far from residuals
# endif // if EMBEDDED != 1
} OSQPInfo;


# ifndef EMBEDDED

/**
 * Polish structure
 */
typedef struct {
  csc *Ared;          ///< active rows of A
  ///<    Ared = vstack[Alow, Aupp]
  c_int    n_low;     ///< number of lower-active rows
  c_int    n_upp;     ///< number of upper-active rows
  c_int   *A_to_Alow; ///< Maps indices in A to indices in Alow
  c_int   *A_to_Aupp; ///< Maps indices in A to indices in Aupp
  c_int   *Alow_to_A; ///< Maps indices in Alow to indices in A
  c_int   *Aupp_to_A; ///< Maps indices in Aupp to indices in A
  c_float *x;         ///< optimal x-solution obtained by polish
  c_float *z;         ///< optimal z-solution obtained by polish
  c_float *y;         ///< optimal y-solution obtained by polish
  c_float  obj_val;   ///< objective value at polished solution
  c_float  pri_res;   ///< primal residual at polished solution
  c_float  dua_res;   ///< dual residual at polished solution
} OSQPPolish;
# endif // ifndef EMBEDDED


/**********************************
* Main structures and Data Types *
**********************************/

/**
 * Data structure
 */
typedef struct {
  c_int    n; ///< number of variables n
  c_int    m; ///< number of constraints m
  csc     *P; ///< the upper triangular part of the quadratic cost matrix P in csc format (size n x n).
  csc     *A; ///< linear constraints matrix A in csc format (size m x n)
  c_float *q; ///< dense array for linear part of cost function (size n)
  c_float *l; ///< dense array for lower bound (size m)
  c_float *u; ///< dense array for upper bound (size m)
} OSQPData;


/**
 * Settings struct
 */
typedef struct {
  c_float rho;                    ///< ADMM step rho
  c_float sigma;                  ///< ADMM step sigma
  c_int   scaling;                ///< heuristic data scaling iterations; if 0, then disabled.

# if EMBEDDED != 1
  c_int   adaptive_rho;           ///< boolean, is rho step size adaptive?
  c_int   adaptive_rho_interval;  ///< number of iterations between rho adaptations; if 0, then it is automatic
  c_float adaptive_rho_tolerance; ///< tolerance X for adapting rho. The new rho has to be X times larger or 1/X times smaller than the current one to trigger a new factorization.
#  ifdef PROFILING
  c_float adaptive_rho_fraction;  ///< interval for adapting rho (fraction of the setup time)
#  endif // Profiling
# endif // EMBEDDED != 1

  c_int                   max_iter;      ///< maximum number of iterations
  c_float                 eps_abs;       ///< absolute convergence tolerance
  c_float                 eps_rel;       ///< relative convergence tolerance
  c_float                 eps_prim_inf;  ///< primal infeasibility tolerance
  c_float                 eps_dual_inf;  ///< dual infeasibility tolerance
  c_float                 alpha;         ///< relaxation parameter
  enum linsys_solver_type linsys_solver; ///< linear system solver to use

# ifndef EMBEDDED
  c_float delta;                         ///< regularization parameter for polishing
  c_int   polish;                        ///< boolean, polish ADMM solution
  c_int   polish_refine_iter;            ///< number of iterative refinement steps in polishing

  c_int verbose;                         ///< boolean, write out progress
# endif // ifndef EMBEDDED

  c_int scaled_termination;              ///< boolean, use scaled termination criteria
  c_int check_termination;               ///< integer, check termination interval; if 0, then termination checking is disabled
  c_int warm_start;                      ///< boolean, warm start

# ifdef PROFILING
  c_float time_limit;                    ///< maximum number of seconds allowed to solve the problem; if 0, then disabled
# endif // ifdef PROFILING
} OSQPSettings;


/**
 * OSQP Workspace
 */
typedef struct {
  /// Problem data to work on (possibly scaled)
  OSQPData *data;

  /// Linear System solver structure
  LinSysSolver *linsys_solver;

# ifndef EMBEDDED
  /// Polish structure
  OSQPPolish *pol;
# endif // ifndef EMBEDDED

  /**
   * @name Vector used to store a vectorized rho parameter
   * @{
   */
  c_float *rho_vec;     ///< vector of rho values
  c_float *rho_inv_vec; ///< vector of inv rho values

  /** @} */

# if EMBEDDED != 1
  c_int *constr_type; ///< Type of constraints: loose (-1), equality (1), inequality (0)
# endif // if EMBEDDED != 1

  /**
   * @name Iterates
   * @{
   */
  c_float *x;        ///< Iterate x
  c_float *y;        ///< Iterate y
  c_float *z;        ///< Iterate z
  c_float *xz_tilde; ///< Iterate xz_tilde

  c_float *x_prev;   ///< Previous x

  /**< NB: Used also as workspace vector for dual residual */
  c_float *z_prev;   ///< Previous z

  /**< NB: Used also as workspace vector for primal residual */

  /**
   * @name Primal and dual residuals workspace variables
   *
   * Needed for residuals computation, tolerances computation,
   * approximate tolerances computation and adapting rho
   * @{
   */
  c_float *Ax;  ///< scaled A * x
  c_float *Px;  ///< scaled P * x
  c_float *Aty; ///< scaled A * x

  /** @} */

  /**
   * @name Primal infeasibility variables
   * @{
   */
  c_float *delta_y;   ///< difference between consecutive dual iterates
  c_float *Atdelta_y; ///< A' * delta_y

  /** @} */

  /**
   * @name Dual infeasibility variables
   * @{
   */
  c_float *delta_x;  ///< difference between consecutive primal iterates
  c_float *Pdelta_x; ///< P * delta_x
  c_float *Adelta_x; ///< A * delta_x

  /** @} */

  /**
   * @name Temporary vectors used in scaling
   * @{
   */

  c_float *D_temp;   ///< temporary primal variable scaling vectors
  c_float *D_temp_A; ///< temporary primal variable scaling vectors storing norms of A columns
  c_float *E_temp;   ///< temporary constraints scaling vectors storing norms of A' columns


  /** @} */

  OSQPSettings *settings; ///< problem settings
  OSQPScaling  *scaling;  ///< scaling vectors
  OSQPSolution *solution; ///< problem solution
  OSQPInfo     *info;     ///< solver information

# ifdef PROFILING
  OSQPTimer *timer;       ///< timer object

  /// flag indicating whether the solve function has been run before
  c_int first_run;

  /// flag indicating whether the update_time should be cleared
  c_int clear_update_time;

  /// flag indicating that osqp_update_rho is called from osqp_solve function
  c_int rho_update_from_solve;
# endif // ifdef PROFILING

# ifdef PRINTING
  c_int summary_printed; ///< Has last summary been printed? (true/false)
# endif // ifdef PRINTING

} OSQPWorkspace;


/**
 * Define linsys_solver prototype structure
 *
 * NB: The details are defined when the linear solver is initialized depending
 *      on the choice
 */
struct linsys_solver {
  enum linsys_solver_type type;                 ///< linear system solver type functions
  c_int (*solve)(LinSysSolver *self,
                 c_float      *b);              ///< solve linear system

# ifndef EMBEDDED
  void (*free)(LinSysSolver *self);             ///< free linear system solver (only in desktop version)
# endif // ifndef EMBEDDED

# if EMBEDDED != 1
  c_int (*update_matrices)(LinSysSolver *s,
                           const csc *P,            ///< update matrices P
                           const csc *A);           //   and A in the solver

  c_int (*update_rho_vec)(LinSysSolver  *s,
                          const c_float *rho_vec);  ///< Update rho_vec
# endif // if EMBEDDED != 1

# ifndef EMBEDDED
  c_int nthreads; ///< number of threads active
# endif // ifndef EMBEDDED
};


/***********************************************************
* Auxiliary functions needed to compute ADMM iterations * *
***********************************************************/
# if EMBEDDED != 1

/**
 * Compute rho estimate from residuals
 * @param work Workspace
 * @return     rho estimate
 */
c_float compute_rho_estimate(OSQPWorkspace *work);

/**
 * Adapt rho value based on current unscaled primal/dual residuals
 * @param work Workspace
 * @return     Exitflag
 */
c_int   adapt_rho(OSQPWorkspace *work);

/**
 * Set values of rho vector based on constraint types
 * @param work Workspace
 */
void    set_rho_vec(OSQPWorkspace *work);

/**
 * Update values of rho vector based on updated constraints.
 * If the constraints change, update the linear systems solver.
 *
 * @param work Workspace
 * @return     Exitflag
 */
c_int   update_rho_vec(OSQPWorkspace *work);

# endif // EMBEDDED

/**
 * Swap c_float vector pointers
 * @param a first vector
 * @param b second vector
 */
void swap_vectors(c_float **a,
                  c_float **b);


/**
 * Cold start workspace variables xz and y
 * @param work Workspace
 */
void cold_start(OSQPWorkspace *work);


/**
 * Update x_tilde and z_tilde variable (first ADMM step)
 * @param work [description]
 */
void update_xz_tilde(OSQPWorkspace *work);


/**
 * Update x (second ADMM step)
 * Update also delta_x (For for dual infeasibility)
 * @param work Workspace
 */
void update_x(OSQPWorkspace *work);


/**
 * Update z (third ADMM step)
 * @param work Workspace
 */
void update_z(OSQPWorkspace *work);


/**
 * Update y variable (fourth ADMM step)
 * Update also delta_y to check for primal infeasibility
 * @param work Workspace
 */
void update_y(OSQPWorkspace *work);


/**
 * Compute objective function from data at value x
 * @param  work OSQPWorkspace structure
 * @param  x    Value x
 * @return      Objective function value
 */
c_float compute_obj_val(OSQPWorkspace *work,
                        c_float       *x);

/**
 * Check whether QP has solution
 * @param info OSQPInfo
 */
c_int has_solution(OSQPInfo *info);

/**
 * Store the QP solution
 * @param work Workspace
 */
void store_solution(OSQPWorkspace *work);


/**
 * Update solver information
 * @param work               Workspace
 * @param iter               Iteration number
 * @param compute_objective  Boolean (if compute the objective or not)
 * @param polish             Boolean (if called from polish)
 */
void update_info(OSQPWorkspace *work,
                 c_int          iter,
                 c_int          compute_objective,
                 c_int          polish);


/**
 * Reset solver information (after problem updates)
 * @param info               Information structure
 */
void reset_info(OSQPInfo *info);


/**
 * Update solver status (value and string)
 * @param info OSQPInfo
 * @param status_val new status value
 */
void update_status(OSQPInfo *info,
                   c_int     status_val);


/**
 * Check if termination conditions are satisfied
 * If the boolean flag is ON, it checks for approximate conditions (10 x larger
 * tolerances than the ones set)
 *
 * @param  work        Workspace
 * @param  approximate Boolean
 * @return      Residuals check
 */
c_int check_termination(OSQPWorkspace *work,
                        c_int          approximate);


# ifndef EMBEDDED

/**
 * Validate problem data
 * @param  data OSQPData to be validated
 * @return      Exitflag to check
 */
c_int validate_data(const OSQPData *data);


/**
 * Validate problem settings
 * @param  settings OSQPSettings to be validated
 * @return          Exitflag to check
 */
c_int validate_settings(const OSQPSettings *settings);


# endif // #ifndef EMBEDDED



/* OSQP error macro */
# if __STDC_VERSION__ >= 199901L
/* The C99 standard gives the __func__ macro, which is preferred over __FUNCTION__ */
#  define osqp_error(error_code) _osqp_error(error_code, __func__);
#else
#  define osqp_error(error_code) _osqp_error(error_code, __FUNCTION__);
#endif



/**
 * Internal function to print error description and return error code.
 * @param  Error code
 * @param  Function name
 * @return Error code
 */
  c_int _osqp_error(enum osqp_error_type error_code,
		    const char * function_name);

# ifndef EMBEDDED

#  include "cs.h"

/**
 * Form square symmetric KKT matrix of the form
 *
 * [P + param1 I,            A';
 *  A             -diag(param2)]
 *
 * NB: Only the upper triangular part is stuffed!
 *
 *
 *  If Pdiag_idx is not OSQP_NULL, it saves the index of the diagonal
 * elements of P there and the number of diagonal elements in Pdiag_n.
 *
 * Similarly, if rhotoKKT is not null,
 * it saves where the values of param2 go in the final KKT matrix
 *
 * NB: Pdiag_idx needs to be freed!
 *
 * @param  P          cost matrix (already just upper triangular part)
 * @param  A          linear constraint matrix
 * @param  format     CSC (0) or CSR (1)
 * @param  param1     regularization parameter
 * @param  param2     regularization parameter (vector)
 * @param  PtoKKT     (modified) index mapping from elements of P to KKT matrix
 * @param  AtoKKT     (modified) index mapping from elements of A to KKT matrix
 * @param  Pdiag_idx  (modified) Address of the index of diagonal elements in P
 * @param  Pdiag_n    (modified) Address to the number of diagonal elements in P
 * @param  param2toKKT    (modified) index mapping from param2 to elements of
 *KKT
 * @return            return status flag
 */
csc* form_KKT(const csc  *P,
              const  csc *A,
              c_int       format,
              c_float     param1,
              c_float    *param2,
              c_int      *PtoKKT,
              c_int      *AtoKKT,
              c_int     **Pdiag_idx,
              c_int      *Pdiag_n,
              c_int      *param2toKKT);
# endif // ifndef EMBEDDED


# if EMBEDDED != 1

/**
 * Update KKT matrix using the elements of P
 *
 * @param KKT       KKT matrix in CSC form (upper-triangular)
 * @param P         P matrix in CSC form (upper-triangular)
 * @param PtoKKT    Vector of pointers from P->x to KKT->x
 * @param param1    Parameter added to the diagonal elements of P
 * @param Pdiag_idx Index of diagonal elements in P->x
 * @param Pdiag_n   Number of diagonal elements of P
 */
void update_KKT_P(csc          *KKT,
                  const csc    *P,
                  const c_int  *PtoKKT,
                  const c_float param1,
                  const c_int  *Pdiag_idx,
                  const c_int   Pdiag_n);


/**
 * Update KKT matrix using the elements of A
 *
 * @param KKT       KKT matrix in CSC form (upper-triangular)
 * @param A         A matrix in CSC form (upper-triangular)
 * @param AtoKKT    Vector of pointers from A->x to KKT->x
 */
void update_KKT_A(csc         *KKT,
                  const csc   *A,
                  const c_int *AtoKKT);


/**
 * Update KKT matrix with new param2
 *
 * @param KKT           KKT matrix
 * @param param2        Parameter of the KKT matrix (vector)
 * @param param2toKKT   index where param2 enters in the KKT matrix
 * @param m             number of constraints
 */
void update_KKT_param2(csc           *KKT,
                       const c_float *param2,
                       const c_int   *param2toKKT,
                       const c_int    m);

# endif // EMBEDDED != 1


/* VECTOR FUNCTIONS ----------------------------------------------------------*/

# ifndef EMBEDDED

/* copy vector a into output (Uses MALLOC)*/
c_float* vec_copy(c_float *a,
                  c_int    n);
# endif // ifndef EMBEDDED

/* copy vector a into preallocated vector b */
void prea_vec_copy(const c_float *a,
                   c_float       *b,
                   c_int          n);

/* copy integer vector a into preallocated vector b */
void prea_int_vec_copy(const c_int *a,
                       c_int       *b,
                       c_int        n);

/* set float vector to scalar */
void vec_set_scalar(c_float *a,
                    c_float  sc,
                    c_int    n);

/* set integer vector to scalar */
void int_vec_set_scalar(c_int *a,
                        c_int  sc,
                        c_int  n);

/* add scalar to vector*/
void vec_add_scalar(c_float *a,
                    c_float  sc,
                    c_int    n);

/* multiply scalar to vector */
void vec_mult_scalar(c_float *a,
                     c_float  sc,
                     c_int    n);

/* c = a + sc*b */
void vec_add_scaled(c_float       *c,
                    const c_float *a,
                    const c_float *b,
                    c_int          n,
                    c_float        sc);

/* ||v||_inf */
c_float vec_norm_inf(const c_float *v,
                     c_int          l);

/* ||Sv||_inf */
c_float vec_scaled_norm_inf(const c_float *S,
                            const c_float *v,
                            c_int          l);

/* ||a - b||_inf */
c_float vec_norm_inf_diff(const c_float *a,
                          const c_float *b,
                          c_int          l);

/* mean of vector elements */
c_float vec_mean(const c_float *a,
                 c_int          n);

# if EMBEDDED != 1

/* Vector elementwise reciprocal b = 1./a (needed for scaling)*/
void vec_ew_recipr(const c_float *a,
                   c_float       *b,
                   c_int          n);
# endif // if EMBEDDED != 1

/* Inner product a'b */
c_float vec_prod(const c_float *a,
                 const c_float *b,
                 c_int          n);

/* Elementwise product a.*b stored in c*/
void vec_ew_prod(const c_float *a,
                 const c_float *b,
                 c_float       *c,
                 c_int          n);

# if EMBEDDED != 1

/* Elementwise sqrt of the vector elements */
void vec_ew_sqrt(c_float *a,
                 c_int    n);

/* Elementwise max between each vector component and max_val */
void vec_ew_max(c_float *a,
                c_int    n,
                c_float  max_val);

/* Elementwise min between each vector component and max_val */
void vec_ew_min(c_float *a,
                c_int    n,
                c_float  min_val);

/* Elementwise maximum between vectors c = max(a, b) */
void vec_ew_max_vec(const c_float *a,
                    const c_float *b,
                    c_float       *c,
                    c_int          n);

/* Elementwise minimum between vectors c = min(a, b) */
void vec_ew_min_vec(const c_float *a,
                    const c_float *b,
                    c_float       *c,
                    c_int          n);

# endif // if EMBEDDED != 1


/* MATRIX FUNCTIONS ----------------------------------------------------------*/

/* multiply scalar to matrix */
void mat_mult_scalar(csc    *A,
                     c_float sc);

/* Premultiply matrix A by diagonal matrix with diagonal d,
   i.e. scale the rows of A by d
 */
void mat_premult_diag(csc           *A,
                      const c_float *d);

/* Premultiply matrix A by diagonal matrix with diagonal d,
   i.e. scale the columns of A by d
 */
void mat_postmult_diag(csc           *A,
                       const c_float *d);


/* Matrix-vector multiplication
 *    y  =  A*x  (if plus_eq == 0)
 *    y +=  A*x  (if plus_eq == 1)
 *    y -=  A*x  (if plus_eq == -1)
 */
void mat_vec(const csc     *A,
             const c_float *x,
             c_float       *y,
             c_int          plus_eq);


/* Matrix-transpose-vector multiplication
 *    y  =  A'*x  (if plus_eq == 0)
 *    y +=  A'*x  (if plus_eq == 1)
 *    y -=  A'*x  (if plus_eq == -1)
 * If skip_diag == 1, then diagonal elements of A are assumed to be zero.
 */
void mat_tpose_vec(const csc     *A,
                   const c_float *x,
                   c_float       *y,
                   c_int          plus_eq,
                   c_int          skip_diag);


# if EMBEDDED != 1

/**
 * Infinity norm of each matrix column
 * @param M	Input matrix
 * @param E     Vector of infinity norms
 *
 */
void mat_inf_norm_cols(const csc *M,
                       c_float   *E);

/**
 * Infinity norm of each matrix row
 * @param M	Input matrix
 * @param E     Vector of infinity norms
 *
 */
void mat_inf_norm_rows(const csc *M,
                       c_float   *E);

/**
 * Infinity norm of each matrix column
 * Matrix M is symmetric upper-triangular
 *
 * @param M	Input matrix (symmetric, upper-triangular)
 * @param E     Vector of infinity norms
 *
 */
void mat_inf_norm_cols_sym_triu(const csc *M,
                                c_float   *E);

# endif // EMBEDDED != 1

/**
 * Compute quadratic form f(x) = 1/2 x' P x
 * @param  P quadratic matrix in CSC form (only upper triangular)
 * @param  x argument float vector
 * @return   quadratic form value
 */
c_float quad_form(const csc     *P,
                  const c_float *x);


/******************
* Versioning     *
******************/

/**
 * Return OSQP version
 * @return  OSQP version
 */
const char* osqp_version(void);


/**********************
* Utility Functions  *
**********************/

# ifndef EMBEDDED

/**
 * Copy settings creating a new settings structure (uses MALLOC)
 * @param  settings Settings to be copied
 * @return          New settings structure
 */
OSQPSettings* copy_settings(const OSQPSettings *settings);

# endif // #ifndef EMBEDDED

/**
 * Custom string copy to avoid string.h library
 * @param dest   destination string
 * @param source source string
 */
void c_strcpy(char       dest[],
              const char source[]);


# ifdef PRINTING

/**
 * Print Header before running the algorithm
 * @param work     osqp workspace
 */
void print_setup_header(const OSQPWorkspace *work);

/**
 * Print header with data to be displayed per iteration
 */
void print_header(void);

/**
 * Print iteration summary
 * @param work current workspace
 */
void print_summary(OSQPWorkspace *work);

/**
 * Print information after polish
 * @param work current workspace
 */
void print_polish(OSQPWorkspace *work);

/**
 * Print footer when algorithm terminates
 * @param info   info structure
 * @param polish is polish enabled?
 */
void print_footer(OSQPInfo *info,
                  c_int     polish);


# endif // ifdef PRINTING


/*********************************
* Timer Structs and Functions * *
*********************************/

/*! \cond PRIVATE */

# ifdef PROFILING

// Windows
#  ifdef IS_WINDOWS

  // Some R packages clash with elements
  // of the windows.h header, so use a
  // slimmer version for conflict avoidance
# ifdef R_LANG
#define NOGDI
# endif

#   include <windows.h>

struct OSQP_TIMER {
  LARGE_INTEGER tic;
  LARGE_INTEGER toc;
  LARGE_INTEGER freq;
};

// Mac
#  elif defined IS_MAC

#   include <mach/mach_time.h>

/* Use MAC OSX  mach_time for timing */
struct OSQP_TIMER {
  uint64_t                  tic;
  uint64_t                  toc;
  mach_timebase_info_data_t tinfo;
};

// Linux
#  else // ifdef IS_WINDOWS

/* Use POSIX clock_gettime() for timing on non-Windows machines */
#   include <time.h>
#   include <sys/time.h>


struct OSQP_TIMER {
  struct timespec tic;
  struct timespec toc;
};

#  endif // ifdef IS_WINDOWS

/*! \endcond */

/**
 * Timer Methods
 */

/**
 * Start timer
 * @param t Timer object
 */
void    osqp_tic(OSQPTimer *t);

/**
 * Report time
 * @param  t Timer object
 * @return   Reported time
 */
c_float osqp_toc(OSQPTimer *t);

# endif /* END #ifdef PROFILING */


/* ================================= DEBUG FUNCTIONS ======================= */

/*! \cond PRIVATE */


# ifndef EMBEDDED

/* Compare CSC matrices */
c_int is_eq_csc(csc    *A,
                csc    *B,
                c_float tol);

/* Convert sparse CSC to dense */
c_float* csc_to_dns(csc *M);

# endif // #ifndef EMBEDDED


# ifdef PRINTING
#  include <stdio.h>


/* Print a csc sparse matrix */
void print_csc_matrix(csc        *M,
                      const char *name);

/* Dump csc sparse matrix to file */
void dump_csc_matrix(csc        *M,
                     const char *file_name);

/* Print a triplet format sparse matrix */
void print_trip_matrix(csc        *M,
                       const char *name);

/* Print a dense matrix */
void print_dns_matrix(c_float    *M,
                      c_int       m,
                      c_int       n,
                      const char *name);

/* Print vector  */
void print_vec(c_float    *v,
               c_int       n,
               const char *name);

/* Dump vector to file */
void dump_vec(c_float    *v,
              c_int       len,
              const char *file_name);

// Print int array
void print_vec_int(c_int      *x,
                   c_int       n,
                   const char *name);

# endif // ifdef PRINTING

/*! \endcond */


// Library to deal with sparse matrices enabled only if embedded not defined
# ifndef EMBEDDED
#  include "cs.h"
# endif // ifndef EMBEDDED

/********************
* Main Solver API  *
********************/

/**
 * @name Main solver API
 * @{
 */

/**
 * Set default settings from constants.h file
 * assumes settings already allocated in memory
 * @param settings settings structure
 */
void osqp_set_default_settings(OSQPSettings *settings);


# ifndef EMBEDDED

/**
 * Initialize OSQP solver allocating memory.
 *
 * All the inputs must be already allocated in memory before calling.
 *
 * It performs:
 * - data and settings validation
 * - problem data scaling
 * - automatic parameters tuning (if enabled)
 * - setup linear system solver:
 *      - direct solver: KKT matrix factorization is performed here
 *      - indirect solver: KKT matrix preconditioning is performed here
 *
 * NB: This is the only function that allocates dynamic memory and is not used
 *during code generation
 *
 * @param  workp        Solver workspace pointer
 * @param  data         Problem data
 * @param  settings     Solver settings
 * @return              Exitflag for errors (0 if no errors)
 */
c_int osqp_setup(OSQPWorkspace** workp, const OSQPData* data, const OSQPSettings* settings);

# endif // #ifndef EMBEDDED

/**
 * Solve quadratic program
 *
 * The final solver information is stored in the \a work->info  structure
 *
 * The solution is stored in the  \a work->solution  structure
 *
 * If the problem is primal infeasible, the certificate is stored
 * in \a work->delta_y
 *
 * If the problem is dual infeasible, the certificate is stored in \a
 * work->delta_x
 *
 * @param  work Workspace allocated
 * @return      Exitflag for errors
 */
c_int osqp_solve(OSQPWorkspace *work);


# ifndef EMBEDDED

/**
 * Cleanup workspace by deallocating memory
 *
 * This function is not used in code generation
 * @param  work Workspace
 * @return      Exitflag for errors
 */
c_int osqp_cleanup(OSQPWorkspace *work);

# endif // ifndef EMBEDDED

/** @} */


/********************************************
* Sublevel API                             *
*                                          *
* Edit data without performing setup again *
********************************************/

/**
 * @name Sublevel API
 * @{
 */

/**
 * Update linear cost in the problem
 * @param  work  Workspace
 * @param  q_new New linear cost
 * @return       Exitflag for errors and warnings
 */
c_int osqp_update_lin_cost(OSQPWorkspace *work,
                           const c_float *q_new);


/**
 * Update lower and upper bounds in the problem constraints
 * @param  work   Workspace
 * @param  l_new New lower bound
 * @param  u_new New upper bound
 * @return        Exitflag: 1 if new lower bound is not <= than new upper bound
 */
c_int osqp_update_bounds(OSQPWorkspace *work,
                         const c_float *l_new,
                         const c_float *u_new);


/**
 * Update lower bound in the problem constraints
 * @param  work   Workspace
 * @param  l_new New lower bound
 * @return        Exitflag: 1 if new lower bound is not <= than upper bound
 */
c_int osqp_update_lower_bound(OSQPWorkspace *work,
                              const c_float *l_new);


/**
 * Update upper bound in the problem constraints
 * @param  work   Workspace
 * @param  u_new New upper bound
 * @return        Exitflag: 1 if new upper bound is not >= than lower bound
 */
c_int osqp_update_upper_bound(OSQPWorkspace *work,
                              const c_float *u_new);


/**
 * Warm start primal and dual variables
 * @param  work Workspace structure
 * @param  x    Primal variable
 * @param  y    Dual variable
 * @return      Exitflag
 */
c_int osqp_warm_start(OSQPWorkspace *work,
                      const c_float *x,
                      const c_float *y);


/**
 * Warm start primal variable
 * @param  work Workspace structure
 * @param  x    Primal variable
 * @return      Exitflag
 */
c_int osqp_warm_start_x(OSQPWorkspace *work,
                        const c_float *x);


/**
 * Warm start dual variable
 * @param  work Workspace structure
 * @param  y    Dual variable
 * @return      Exitflag
 */
c_int osqp_warm_start_y(OSQPWorkspace *work,
                        const c_float *y);


# if EMBEDDED != 1

/**
 * Update elements of matrix P (upper triangular)
 * without changing sparsity structure.
 *
 *
 *  If Px_new_idx is OSQP_NULL, Px_new is assumed to be as long as P->x
 *  and the whole P->x is replaced.
 *
 * @param  work       Workspace structure
 * @param  Px_new     Vector of new elements in P->x (upper triangular)
 * @param  Px_new_idx Index mapping new elements to positions in P->x
 * @param  P_new_n    Number of new elements to be changed
 * @return            output flag:  0: OK
 *                                  1: P_new_n > nnzP
 *                                 <0: error in the update
 */
c_int osqp_update_P(OSQPWorkspace *work,
                    const c_float *Px_new,
                    const c_int   *Px_new_idx,
                    c_int          P_new_n);


/**
 * Update elements of matrix A without changing sparsity structure.
 *
 *
 *  If Ax_new_idx is OSQP_NULL, Ax_new is assumed to be as long as A->x
 *  and the whole A->x is replaced.
 *
 * @param  work       Workspace structure
 * @param  Ax_new     Vector of new elements in A->x
 * @param  Ax_new_idx Index mapping new elements to positions in A->x
 * @param  A_new_n    Number of new elements to be changed
 * @return            output flag:  0: OK
 *                                  1: A_new_n > nnzA
 *                                 <0: error in the update
 */
c_int osqp_update_A(OSQPWorkspace *work,
                    const c_float *Ax_new,
                    const c_int   *Ax_new_idx,
                    c_int          A_new_n);


/**
 * Update elements of matrix P (upper triangular) and elements of matrix A
 * without changing sparsity structure.
 *
 *
 *  If Px_new_idx is OSQP_NULL, Px_new is assumed to be as long as P->x
 *  and the whole P->x is replaced.
 *
 *  If Ax_new_idx is OSQP_NULL, Ax_new is assumed to be as long as A->x
 *  and the whole A->x is replaced.
 *
 * @param  work       Workspace structure
 * @param  Px_new     Vector of new elements in P->x (upper triangular)
 * @param  Px_new_idx Index mapping new elements to positions in P->x
 * @param  P_new_n    Number of new elements to be changed
 * @param  Ax_new     Vector of new elements in A->x
 * @param  Ax_new_idx Index mapping new elements to positions in A->x
 * @param  A_new_n    Number of new elements to be changed
 * @return            output flag:  0: OK
 *                                  1: P_new_n > nnzP
 *                                  2: A_new_n > nnzA
 *                                 <0: error in the update
 */
c_int osqp_update_P_A(OSQPWorkspace *work,
                      const c_float *Px_new,
                      const c_int   *Px_new_idx,
                      c_int          P_new_n,
                      const c_float *Ax_new,
                      const c_int   *Ax_new_idx,
                      c_int          A_new_n);

/**
 * Update rho. Limit it between RHO_MIN and RHO_MAX.
 * @param  work         Workspace
 * @param  rho_new      New rho setting
 * @return              Exitflag
 */
c_int osqp_update_rho(OSQPWorkspace *work,
                      c_float        rho_new);

# endif // if EMBEDDED != 1

/** @} */


/**
 * @name Update settings
 * @{
 */


/**
 * Update max_iter setting
 * @param  work         Workspace
 * @param  max_iter_new New max_iter setting
 * @return              Exitflag
 */
c_int osqp_update_max_iter(OSQPWorkspace *work,
                           c_int          max_iter_new);


/**
 * Update absolute tolernace value
 * @param  work        Workspace
 * @param  eps_abs_new New absolute tolerance value
 * @return             Exitflag
 */
c_int osqp_update_eps_abs(OSQPWorkspace *work,
                          c_float        eps_abs_new);


/**
 * Update relative tolernace value
 * @param  work        Workspace
 * @param  eps_rel_new New relative tolerance value
 * @return             Exitflag
 */
c_int osqp_update_eps_rel(OSQPWorkspace *work,
                          c_float        eps_rel_new);


/**
 * Update primal infeasibility tolerance
 * @param  work          Workspace
 * @param  eps_prim_inf_new  New primal infeasibility tolerance
 * @return               Exitflag
 */
c_int osqp_update_eps_prim_inf(OSQPWorkspace *work,
                               c_float        eps_prim_inf_new);


/**
 * Update dual infeasibility tolerance
 * @param  work          Workspace
 * @param  eps_dual_inf_new  New dual infeasibility tolerance
 * @return               Exitflag
 */
c_int osqp_update_eps_dual_inf(OSQPWorkspace *work,
                               c_float        eps_dual_inf_new);


/**
 * Update relaxation parameter alpha
 * @param  work  Workspace
 * @param  alpha_new New relaxation parameter value
 * @return       Exitflag
 */
c_int osqp_update_alpha(OSQPWorkspace *work,
                        c_float        alpha_new);


/**
 * Update warm_start setting
 * @param  work           Workspace
 * @param  warm_start_new New warm_start setting
 * @return                Exitflag
 */
c_int osqp_update_warm_start(OSQPWorkspace *work,
                             c_int          warm_start_new);


/**
 * Update scaled_termination setting
 * @param  work                 Workspace
 * @param  scaled_termination_new  New scaled_termination setting
 * @return                      Exitflag
 */
c_int osqp_update_scaled_termination(OSQPWorkspace *work,
                                     c_int          scaled_termination_new);

/**
 * Update check_termination setting
 * @param  work                   Workspace
 * @param  check_termination_new  New check_termination setting
 * @return                        Exitflag
 */
c_int osqp_update_check_termination(OSQPWorkspace *work,
                                    c_int          check_termination_new);


# ifndef EMBEDDED

/**
 * Update regularization parameter in polish
 * @param  work      Workspace
 * @param  delta_new New regularization parameter
 * @return           Exitflag
 */
c_int osqp_update_delta(OSQPWorkspace *work,
                        c_float        delta_new);


/**
 * Update polish setting
 * @param  work          Workspace
 * @param  polish_new New polish setting
 * @return               Exitflag
 */
c_int osqp_update_polish(OSQPWorkspace *work,
                         c_int          polish_new);


/**
 * Update number of iterative refinement steps in polish
 * @param  work                Workspace
 * @param  polish_refine_iter_new New iterative reginement steps
 * @return                     Exitflag
 */
c_int osqp_update_polish_refine_iter(OSQPWorkspace *work,
                                     c_int          polish_refine_iter_new);


/**
 * Update verbose setting
 * @param  work        Workspace
 * @param  verbose_new New verbose setting
 * @return             Exitflag
 */
c_int osqp_update_verbose(OSQPWorkspace *work,
                          c_int          verbose_new);


# endif // #ifndef EMBEDDED

# ifdef PROFILING

/**
 * Update time_limit setting
 * @param  work            Workspace
 * @param  time_limit_new  New time_limit setting
 * @return                 Exitflag
 */
c_int osqp_update_time_limit(OSQPWorkspace *work,
                             c_float        time_limit_new);
# endif // ifdef PROFILING

/** @} */


/* Define Projections onto set C involved in the ADMM algorithm */

/**
 * Project z onto \f$C = [l, u]\f$
 * @param z    Vector to project
 * @param work Workspace
 */
void project(OSQPWorkspace *work,
             c_float       *z);


/**
 * Ensure z satisfies box constraints and y is is normal cone of z
 * @param work Workspace
 * @param z    Primal variable z
 * @param y    Dual variable y
 */
void project_normalcone(OSQPWorkspace *work,
                        c_float       *z,
                        c_float       *y);


// QDLDL integer and float types

typedef long long    QDLDL_int;   /* for indices */
typedef double  QDLDL_float; /* for numerical values  */
typedef unsigned char   QDLDL_bool;  /* for boolean values  */


/**
  * Compute the elimination tree for a quasidefinite matrix
  * in compressed sparse column form, where the input matrix is
  * assumed to contain data for the upper triangular part of A only,
  * and there are no duplicate indices.
  *
  * Returns an elimination tree for the factorization A = LDL^T and a
  * count of the nonzeros in each column of L that are strictly below the
  * diagonal.
  *
  * Does not use MALLOC.  It is assumed that the arrays work, Lnz, and
  * etree will be allocated with a number of elements equal to n.
  *
  * The data in (n,Ap,Ai) are from a square matrix A in CSC format, and
  * should include the upper triangular part of A only.
  *
  * This function is only intended for factorisation of QD matrices specified
  * by their upper triangular part.  An error is returned if any column has
  * data below the diagonal or s completely empty.
  *
  * For matrices with a non-empty column but a zero on the corresponding diagonal,
  * this function will *not* return an error, as it may still be possible to factor
  * such a matrix in LDL form.   No promises are made in this case though...
  *
  * @param  n      number of columns in CSC matrix A (assumed square)
  * @param  Ap     column pointers (size n+1) for columns of A
  * @param  Ai     row indices of A.  Has Ap[n] elements
  * @param  work   work vector (size n) (no meaning on return)
  * @param  Lnz    count of nonzeros in each column of L (size n) below diagonal
  * @param  etree  elimination tree (size n)
  * @return total  sum of Lnz (i.e. total nonzeros in L below diagonal). Returns
  *                -1 if the input does not have triu structure or has an empty
  *                column.
  *
*/
 QDLDL_int QDLDL_etree(const QDLDL_int   n,
                       const QDLDL_int* Ap,
                       const QDLDL_int* Ai,
                       QDLDL_int* work,
                       QDLDL_int* Lnz,
                       QDLDL_int* etree);


/**
  * Compute an LDL decomposition for a quasidefinite matrix
  * in compressed sparse column form, where the input matrix is
  * assumed to contain data for the upper triangular part of A only,
  * and there are no duplicate indices.
  *
  * Returns factors L, D and Dinv = 1./D.
  *
  * Does not use MALLOC.  It is assumed that L will be a compressed
  * sparse column matrix with data (n,Lp,Li,Lx)  with sufficient space
  * allocated, with a number of nonzeros equal to the count given
  * as a return value by QDLDL_etree
  *
  * @param  n      number of columns in L and A (both square)
  * @param  Ap     column pointers (size n+1) for columns of A (not modified)
  * @param  Ai     row indices of A.  Has Ap[n] elements (not modified)
  * @param  Ax     data of A.  Has Ap[n] elements (not modified)
  * @param  Lp     column pointers (size n+1) for columns of L
  * @param  Li     row indices of L.  Has Lp[n] elements
  * @param  Lx     data of L.  Has Lp[n] elements
  * @param  D      vectorized factor D.  Length is n
  * @param  Dinv   reciprocal of D.  Length is n
  * @param  Lnz    count of nonzeros in each column of L below diagonal,
  *                as given by QDLDL_etree (not modified)
  * @param  etree  elimination tree as as given by QDLDL_etree (not modified)
  * @param  bwork  working array of bools. Length is n
  * @param  iwork  working array of integers. Length is 3*n
  * @param  fwork  working array of floats. Length is n
  * @return        Returns a count of the number of positive elements
  *                in D.  Returns -1 and exits immediately if any element
  *                of D evaluates exactly to zero (matrix is not quasidefinite
  *                or otherwise LDL factorisable)
  *
*/
QDLDL_int QDLDL_factor(const QDLDL_int    n,
                  const QDLDL_int*   Ap,
                  const QDLDL_int*   Ai,
                  const QDLDL_float* Ax,
                  QDLDL_int*   Lp,
                  QDLDL_int*   Li,
                  QDLDL_float* Lx,
                  QDLDL_float* D,
                  QDLDL_float* Dinv,
                  const QDLDL_int* Lnz,
                  const QDLDL_int* etree,
                  QDLDL_bool* bwork,
                  QDLDL_int* iwork,
                  QDLDL_float* fwork);


/**
  * Solves LDL'x = b
  *
  * It is assumed that L will be a compressed
  * sparse column matrix with data (n,Lp,Li,Lx).
  *
  * @param  n      number of columns in L
  * @param  Lp     column pointers (size n+1) for columns of L
  * @param  Li     row indices of L.  Has Lp[n] elements
  * @param  Lx     data of L.  Has Lp[n] elements
  * @param  Dinv   reciprocal of D.  Length is n
  * @param  x      initialized to b.  Equal to x on return
  *
*/
void QDLDL_solve(const QDLDL_int    n,
                 const QDLDL_int*   Lp,
                 const QDLDL_int*   Li,
                 const QDLDL_float* Lx,
                 const QDLDL_float* Dinv,
                 QDLDL_float* x);


/**
 * Solves (L+I)x = b
 *
 * It is assumed that L will be a compressed
 * sparse column matrix with data (n,Lp,Li,Lx).
 *
 * @param  n      number of columns in L
 * @param  Lp     column pointers (size n+1) for columns of L
 * @param  Li     row indices of L.  Has Lp[n] elements
 * @param  Lx     data of L.  Has Lp[n] elements
 * @param  x      initialized to b.  Equal to x on return
 *
*/
void QDLDL_Lsolve(const QDLDL_int    n,
                  const QDLDL_int*   Lp,
                  const QDLDL_int*   Li,
                  const QDLDL_float* Lx,
                  QDLDL_float* x);


/**
 * Solves (L+I)'x = b
 *
 * It is assumed that L will be a compressed
 * sparse column matrix with data (n,Lp,Li,Lx).
 *
 * @param  n      number of columns in L
 * @param  Lp     column pointers (size n+1) for columns of L
 * @param  Li     row indices of L.  Has Lp[n] elements
 * @param  Lx     data of L.  Has Lp[n] elements
 * @param  x      initialized to b.  Equal to x on return
 *
*/
void QDLDL_Ltsolve(const QDLDL_int    n,
                   const QDLDL_int*   Lp,
                   const QDLDL_int*   Li,
                   const QDLDL_float* Lx,
                   QDLDL_float* x);


/**
 * QDLDL solver structure
 */
typedef struct qdldl qdldl_solver;

struct qdldl {
    enum linsys_solver_type type;

    /**
     * @name Functions
     * @{
     */
    c_int (*solve)(struct qdldl * self, c_float * b);

#ifndef EMBEDDED
    void (*free)(struct qdldl * self); ///< Free workspace (only if desktop)
#endif

    // This used only in non embedded or embedded 2 version
#if EMBEDDED != 1
    c_int (*update_matrices)(struct qdldl * self, const csc *P, const csc *A);  ///< Update solver matrices
    c_int (*update_rho_vec)(struct qdldl * self, const c_float * rho_vec);      ///< Update rho_vec parameter
#endif

#ifndef EMBEDDED
    c_int nthreads;
#endif
    /** @} */

    /**
     * @name Attributes
     * @{
     */
    csc *L;                 ///< lower triangular matrix in LDL factorization
    c_float *Dinv;          ///< inverse of diag matrix in LDL (as a vector)
    c_int   *P;             ///< permutation of KKT matrix for factorization
    c_float *bp;            ///< workspace memory for solves
    c_float *sol;           ///< solution to the KKT system
    c_float *rho_inv_vec;   ///< parameter vector
    c_float sigma;          ///< scalar parameter
#ifndef EMBEDDED
    c_int polish;           ///< polishing flag
#endif
    c_int n;                ///< number of QP variables
    c_int m;                ///< number of QP constraints


#if EMBEDDED != 1
    // These are required for matrix updates
    c_int * Pdiag_idx, Pdiag_n;  ///< index and number of diagonal elements in P
    csc   * KKT;                 ///< Permuted KKT matrix in sparse form (used to update P and A matrices)
    c_int * PtoKKT, * AtoKKT;    ///< Index of elements from P and A to KKT matrix
    c_int * rhotoKKT;            ///< Index of rho places in KKT matrix
    // QDLDL Numeric workspace
    QDLDL_float *D;
    QDLDL_int   *etree;
    QDLDL_int   *Lnz;
    QDLDL_int   *iwork;
    QDLDL_bool  *bwork;
    QDLDL_float *fwork;
#endif

    /** @} */
};



/**
 * Initialize QDLDL Solver
 *
 * @param  s         Pointer to a private structure
 * @param  P         Cost function matrix (upper triangular form)
 * @param  A         Constraints matrix
 * @param  sigma     Algorithm parameter. If polish, then sigma = delta.
 * @param  rho_vec   Algorithm parameter. If polish, then rho_vec = OSQP_NULL.
 * @param  polish    Flag whether we are initializing for polish or not
 * @return           Exitflag for error (0 if no errors)
 */
c_int init_linsys_solver_qdldl(qdldl_solver ** sp, const csc * P, const csc * A, c_float sigma, const c_float * rho_vec, c_int polish);

/**
 * Solve linear system and store result in b
 * @param  s        Linear system solver structure
 * @param  b        Right-hand side
 * @return          Exitflag
 */
c_int solve_linsys_qdldl(qdldl_solver * s, c_float * b);


#if EMBEDDED != 1
/**
 * Update linear system solver matrices
 * @param  s        Linear system solver structure
 * @param  P        Matrix P
 * @param  A        Matrix A
 * @return          Exitflag
 */
c_int update_linsys_solver_matrices_qdldl(qdldl_solver * s, const csc *P, const csc *A);




/**
 * Update rho_vec parameter in linear system solver structure
 * @param  s        Linear system solver structure
 * @param  rho_vec  new rho_vec value
 * @return          exitflag
 */
c_int update_linsys_solver_rho_vec_qdldl(qdldl_solver * s, const c_float * rho_vec);

#endif

#ifndef EMBEDDED
/**
 * Free linear system solver
 * @param s linear system solver object
 */
void free_linsys_solver_qdldl(qdldl_solver * s);
#endif


// Enable data scaling if EMBEDDED is disabled or if EMBEDDED == 2
# if EMBEDDED != 1

/**
 * Scale problem matrices
 * @param  work Workspace
 * @return      exitflag
 */
c_int scale_data(OSQPWorkspace *work);
# endif // if EMBEDDED != 1


/**
 * Unscale problem matrices
 * @param  work Workspace
 * @return      exitflag
 */
c_int unscale_data(OSQPWorkspace *work);


/**
 * Unscale solution
 * @param  work Workspace
 * @return      exitflag
 */
c_int unscale_solution(OSQPWorkspace *work);


// Data structure prototypes
extern csc Pdata;
extern csc Adata;
extern c_float qdata[172];
extern c_float ldata[304];
extern c_float udata[304];
extern OSQPData data;

// Settings structure prototype
extern OSQPSettings settings;

// Scaling structure prototypes
extern c_float Dscaling[172];
extern c_float Dinvscaling[172];
extern c_float Escaling[304];
extern c_float Einvscaling[304];
extern OSQPScaling scaling;

// Prototypes for linsys_solver structure
extern csc linsys_solver_L;
extern c_float linsys_solver_Dinv[476];
extern c_int linsys_solver_P[476];
extern c_float linsys_solver_bp[476];
extern c_float linsys_solver_sol[476];
extern c_float linsys_solver_rho_inv_vec[304];
extern c_int linsys_solver_Pdiag_idx[117];
extern csc linsys_solver_KKT;
extern c_int linsys_solver_PtoKKT[117];
extern c_int linsys_solver_AtoKKT[884];
extern c_int linsys_solver_rhotoKKT[304];
extern QDLDL_float linsys_solver_D[476];
extern QDLDL_int linsys_solver_etree[476];
extern QDLDL_int linsys_solver_Lnz[476];
extern QDLDL_int   linsys_solver_iwork[1428];
extern QDLDL_bool  linsys_solver_bwork[476];
extern QDLDL_float linsys_solver_fwork[476];
extern qdldl_solver linsys_solver;

// Prototypes for solution
extern c_float xsolution[172];
extern c_float ysolution[304];

extern OSQPSolution solution;

// Prototype for info structure
extern OSQPInfo info;

// Prototypes for the workspace
extern c_float work_rho_vec[304];
extern c_float work_rho_inv_vec[304];
extern c_int work_constr_type[304];
extern c_float work_x[172];
extern c_float work_y[304];
extern c_float work_z[304];
extern c_float work_xz_tilde[476];
extern c_float work_x_prev[172];
extern c_float work_z_prev[304];
extern c_float work_Ax[304];
extern c_float work_Px[172];
extern c_float work_Aty[172];
extern c_float work_delta_y[304];
extern c_float work_Atdelta_y[172];
extern c_float work_delta_x[172];
extern c_float work_Pdelta_x[172];
extern c_float work_Adelta_x[304];
extern c_float work_D_temp[172];
extern c_float work_D_temp_A[172];
extern c_float work_E_temp[304];

extern OSQPWorkspace workspace;

// FSBRT
extern c_float l_new[304];
extern c_float u_new[304];


typedef struct {
  double x1;
  double x2;
  double x3;
  double x4;
  double x5;
  double x6;
  double x7;
  double x8;
  double x9;
  double x10;
  double x11;
  double x12;
} StateSpace;

typedef struct {
  double u1;
  double u2;
  double u3;
  double u4;
} ControlInputs;


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef HEADER_H
