#include "stdio.h"
#include <string.h>

#include "sfun_header.h"


/***********************************************************
* Auxiliary functions needed to compute ADMM iterations * *
***********************************************************/
#if EMBEDDED != 1
c_float compute_rho_estimate(OSQPWorkspace *work) {
  c_int   n, m;                       // Dimensions
  c_float pri_res, dua_res;           // Primal and dual residuals
  c_float pri_res_norm, dua_res_norm; // Normalization for the residuals
  c_float temp_res_norm;              // Temporary residual norm
  c_float rho_estimate;               // Rho estimate value

  // Get problem dimensions
  n = work->data->n;
  m = work->data->m;

  // Get primal and dual residuals
  pri_res = vec_norm_inf(work->z_prev, m);
  dua_res = vec_norm_inf(work->x_prev, n);

  // Normalize primal residual
  pri_res_norm  = vec_norm_inf(work->z, m);           // ||z||
  temp_res_norm = vec_norm_inf(work->Ax, m);          // ||Ax||
  pri_res_norm  = c_max(pri_res_norm, temp_res_norm); // max (||z||,||Ax||)
  pri_res      /= (pri_res_norm + 1e-10);             // Normalize primal
                                                      // residual (prevent 0
                                                      // division)

  // Normalize dual residual
  dua_res_norm  = vec_norm_inf(work->data->q, n);     // ||q||
  temp_res_norm = vec_norm_inf(work->Aty, n);         // ||A' y||
  dua_res_norm  = c_max(dua_res_norm, temp_res_norm);
  temp_res_norm = vec_norm_inf(work->Px, n);          //  ||P x||
  dua_res_norm  = c_max(dua_res_norm, temp_res_norm); // max(||q||,||A' y||,||P
                                                      // x||)
  dua_res      /= (dua_res_norm + 1e-10);             // Normalize dual residual
                                                      // (prevent 0 division)


  // Return rho estimate
  rho_estimate = work->settings->rho * c_sqrt(pri_res / (dua_res + 1e-10)); // (prevent
                                                                            // 0
                                                                            // division)
  rho_estimate = c_min(c_max(rho_estimate, RHO_MIN), RHO_MAX);              // Constrain
                                                                            // rho
                                                                            // values
  return rho_estimate;
}

c_int adapt_rho(OSQPWorkspace *work) {
  c_int   exitflag; // Exitflag
  c_float rho_new;  // New rho value

  exitflag = 0;     // Initialize exitflag to 0

  // Compute new rho
  rho_new = compute_rho_estimate(work);

  // Set rho estimate in info
  work->info->rho_estimate = rho_new;

  // Check if the new rho is large or small enough and update it in case
  if ((rho_new > work->settings->rho * work->settings->adaptive_rho_tolerance) ||
      (rho_new < work->settings->rho /  work->settings->adaptive_rho_tolerance)) {
    exitflag                 = osqp_update_rho(work, rho_new);
    work->info->rho_updates += 1;
  }

  return exitflag;
}

void set_rho_vec(OSQPWorkspace *work) {
  c_int i;

  work->settings->rho = c_min(c_max(work->settings->rho, RHO_MIN), RHO_MAX);

  for (i = 0; i < work->data->m; i++) {
    if ((work->data->l[i] < -OSQP_INFTY * MIN_SCALING) &&
        (work->data->u[i] > OSQP_INFTY * MIN_SCALING)) {
      // Loose bounds
      work->constr_type[i] = -1;
      work->rho_vec[i]     = RHO_MIN;
    } else if (work->data->u[i] - work->data->l[i] < RHO_TOL) {
      // Equality constraints
      work->constr_type[i] = 1;
      work->rho_vec[i]     = RHO_EQ_OVER_RHO_INEQ * work->settings->rho;
    } else {
      // Inequality constraints
      work->constr_type[i] = 0;
      work->rho_vec[i]     = work->settings->rho;
    }
    work->rho_inv_vec[i] = 1. / work->rho_vec[i];
  }
}

c_int update_rho_vec(OSQPWorkspace *work) {
  c_int i, exitflag, constr_type_changed;

  exitflag            = 0;
  constr_type_changed = 0;

  for (i = 0; i < work->data->m; i++) {
    if ((work->data->l[i] < -OSQP_INFTY * MIN_SCALING) &&
        (work->data->u[i] > OSQP_INFTY * MIN_SCALING)) {
      // Loose bounds
      if (work->constr_type[i] != -1) {
        work->constr_type[i] = -1;
        work->rho_vec[i]     = RHO_MIN;
        work->rho_inv_vec[i] = 1. / RHO_MIN;
        constr_type_changed  = 1;
      }
    } else if (work->data->u[i] - work->data->l[i] < RHO_TOL) {
      // Equality constraints
      if (work->constr_type[i] != 1) {
        work->constr_type[i] = 1;
        work->rho_vec[i]     = RHO_EQ_OVER_RHO_INEQ * work->settings->rho;
        work->rho_inv_vec[i] = 1. / work->rho_vec[i];
        constr_type_changed  = 1;
      }
    } else {
      // Inequality constraints
      if (work->constr_type[i] != 0) {
        work->constr_type[i] = 0;
        work->rho_vec[i]     = work->settings->rho;
        work->rho_inv_vec[i] = 1. / work->settings->rho;
        constr_type_changed  = 1;
      }
    }
  }

  // Update rho_vec in KKT matrix if constraints type has changed
  if (constr_type_changed == 1) {
    exitflag = work->linsys_solver->update_rho_vec(work->linsys_solver,
                                                   work->rho_vec);
  }

  return exitflag;
}

#endif // EMBEDDED != 1


void swap_vectors(c_float **a, c_float **b) {
  c_float *temp;

  temp = *b;
  *b   = *a;
  *a   = temp;
}

void cold_start(OSQPWorkspace *work) {
  vec_set_scalar(work->x, 0., work->data->n);
  vec_set_scalar(work->z, 0., work->data->m);
  vec_set_scalar(work->y, 0., work->data->m);
}

static void compute_rhs(OSQPWorkspace *work) {
  c_int i; // Index

  for (i = 0; i < work->data->n; i++) {
    // Cycle over part related to x variables
    work->xz_tilde[i] = work->settings->sigma * work->x_prev[i] -
                        work->data->q[i];
  }

  for (i = 0; i < work->data->m; i++) {
    // Cycle over dual variable in the first step (nu)
    work->xz_tilde[i + work->data->n] = work->z_prev[i] - work->rho_inv_vec[i] *
                                        work->y[i];
  }
}

void update_xz_tilde(OSQPWorkspace *work) {
  // Compute right-hand side
  compute_rhs(work);

  // Solve linear system
  work->linsys_solver->solve(work->linsys_solver, work->xz_tilde);
}

void update_x(OSQPWorkspace *work) {
  c_int i;

  // update x
  for (i = 0; i < work->data->n; i++) {
    work->x[i] = work->settings->alpha * work->xz_tilde[i] +
                 ((c_float)1.0 - work->settings->alpha) * work->x_prev[i];
  }

  // update delta_x
  for (i = 0; i < work->data->n; i++) {
    work->delta_x[i] = work->x[i] - work->x_prev[i];
  }
}

void update_z(OSQPWorkspace *work) {
  c_int i;

  // update z
  for (i = 0; i < work->data->m; i++) {
    work->z[i] = work->settings->alpha * work->xz_tilde[i + work->data->n] +
                 ((c_float)1.0 - work->settings->alpha) * work->z_prev[i] +
                 work->rho_inv_vec[i] * work->y[i];
  }

  // project z
  project(work, work->z);
}

void update_y(OSQPWorkspace *work) {
  c_int i; // Index

  for (i = 0; i < work->data->m; i++) {
    work->delta_y[i] = work->rho_vec[i] *
                       (work->settings->alpha *
                        work->xz_tilde[i + work->data->n] +
                        ((c_float)1.0 - work->settings->alpha) * work->z_prev[i] -
                        work->z[i]);
    work->y[i] += work->delta_y[i];
  }
}

c_float compute_obj_val(OSQPWorkspace *work, c_float *x) {
  c_float obj_val;

  obj_val = quad_form(work->data->P, x) +
            vec_prod(work->data->q, x, work->data->n);

  if (work->settings->scaling) {
    obj_val *= work->scaling->cinv;
  }

  return obj_val;
}

c_float compute_pri_res(OSQPWorkspace *work, c_float *x, c_float *z) {
  // NB: Use z_prev as working vector
  // pr = Ax - z

  mat_vec(work->data->A, x, work->Ax, 0); // Ax
  vec_add_scaled(work->z_prev, work->Ax, z, work->data->m, -1);

  // If scaling active -> rescale residual
  if (work->settings->scaling && !work->settings->scaled_termination) {
    return vec_scaled_norm_inf(work->scaling->Einv, work->z_prev, work->data->m);
  }

  // Return norm of the residual
  return vec_norm_inf(work->z_prev, work->data->m);
}

c_float compute_pri_tol(OSQPWorkspace *work, c_float eps_abs, c_float eps_rel) {
  c_float max_rel_eps, temp_rel_eps;

  // max_rel_eps = max(||z||, ||A x||)
  if (work->settings->scaling && !work->settings->scaled_termination) {
    // ||Einv * z||
    max_rel_eps =
      vec_scaled_norm_inf(work->scaling->Einv, work->z, work->data->m);

    // ||Einv * A * x||
    temp_rel_eps = vec_scaled_norm_inf(work->scaling->Einv,
                                       work->Ax,
                                       work->data->m);

    // Choose maximum
    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
  } else { // No unscaling required
    // ||z||
    max_rel_eps = vec_norm_inf(work->z, work->data->m);

    // ||A * x||
    temp_rel_eps = vec_norm_inf(work->Ax, work->data->m);

    // Choose maximum
    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
  }

  // eps_prim
  return eps_abs + eps_rel * max_rel_eps;
}

c_float compute_dua_res(OSQPWorkspace *work, c_float *x, c_float *y) {
  // NB: Use x_prev as temporary vector
  // NB: Only upper triangular part of P is stored.
  // dr = q + A'*y + P*x

  // dr = q
  prea_vec_copy(work->data->q, work->x_prev, work->data->n);

  // P * x (upper triangular part)
  mat_vec(work->data->P, x, work->Px, 0);

  // P' * x (lower triangular part with no diagonal)
  mat_tpose_vec(work->data->P, x, work->Px, 1, 1);

  // dr += P * x (full P matrix)
  vec_add_scaled(work->x_prev, work->x_prev, work->Px, work->data->n, 1);

  // dr += A' * y
  if (work->data->m > 0) {
    mat_tpose_vec(work->data->A, y, work->Aty, 0, 0);
    vec_add_scaled(work->x_prev, work->x_prev, work->Aty, work->data->n, 1);
  }

  // If scaling active -> rescale residual
  if (work->settings->scaling && !work->settings->scaled_termination) {
    return work->scaling->cinv * vec_scaled_norm_inf(work->scaling->Dinv,
                                                     work->x_prev,
                                                     work->data->n);
  }

  return vec_norm_inf(work->x_prev, work->data->n);
}

c_float compute_dua_tol(OSQPWorkspace *work, c_float eps_abs, c_float eps_rel) {
  c_float max_rel_eps, temp_rel_eps;

  // max_rel_eps = max(||q||, ||A' y|, ||P x||)
  if (work->settings->scaling && !work->settings->scaled_termination) {
    // || Dinv q||
    max_rel_eps = vec_scaled_norm_inf(work->scaling->Dinv,
                                      work->data->q,
                                      work->data->n);

    // || Dinv A' y ||
    temp_rel_eps = vec_scaled_norm_inf(work->scaling->Dinv,
                                       work->Aty,
                                       work->data->n);
    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);

    // || Dinv P x||
    temp_rel_eps = vec_scaled_norm_inf(work->scaling->Dinv,
                                       work->Px,
                                       work->data->n);
    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);

    // Multiply by cinv
    max_rel_eps *= work->scaling->cinv;
  } else { // No scaling required
    // ||q||
    max_rel_eps = vec_norm_inf(work->data->q, work->data->n);

    // ||A'*y||
    temp_rel_eps = vec_norm_inf(work->Aty, work->data->n);
    max_rel_eps  = c_max(max_rel_eps, temp_rel_eps);

    // ||P*x||
    temp_rel_eps = vec_norm_inf(work->Px, work->data->n);
    max_rel_eps  = c_max(max_rel_eps, temp_rel_eps);
  }

  // eps_dual
  return eps_abs + eps_rel * max_rel_eps;
}

c_int is_primal_infeasible(OSQPWorkspace *work, c_float eps_prim_inf) {
  // This function checks for the primal infeasibility termination criteria.
  //
  // 1) A' * delta_y < eps * ||delta_y||
  //
  // 2) u'*max(delta_y, 0) + l'*min(delta_y, 0) < -eps * ||delta_y||
  //

  c_int i; // Index for loops
  c_float norm_delta_y;
  c_float ineq_lhs = 0.0;

  // Project delta_y onto the polar of the recession cone of [l,u]
  for (i = 0; i < work->data->m; i++) {
    if (work->data->u[i] > OSQP_INFTY * MIN_SCALING) {          // Infinite upper bound
      if (work->data->l[i] < -OSQP_INFTY * MIN_SCALING) {       // Infinite lower bound
        // Both bounds infinite
        work->delta_y[i] = 0.0;
      } else {
        // Only upper bound infinite
        work->delta_y[i] = c_min(work->delta_y[i], 0.0);
      }
    } else if (work->data->l[i] < -OSQP_INFTY * MIN_SCALING) {  // Infinite lower bound
      // Only lower bound infinite
      work->delta_y[i] = c_max(work->delta_y[i], 0.0);
    }
  }

  // Compute infinity norm of delta_y (unscale if necessary)
  if (work->settings->scaling && !work->settings->scaled_termination) {
    // Use work->Adelta_x as temporary vector
    vec_ew_prod(work->scaling->E, work->delta_y, work->Adelta_x, work->data->m);
    norm_delta_y = vec_norm_inf(work->Adelta_x, work->data->m);
  } else {
    norm_delta_y = vec_norm_inf(work->delta_y, work->data->m);
  }

  if (norm_delta_y > eps_prim_inf) { // ||delta_y|| > 0

    for (i = 0; i < work->data->m; i++) {
      ineq_lhs += work->data->u[i] * c_max(work->delta_y[i], 0) + \
                  work->data->l[i] * c_min(work->delta_y[i], 0);
    }

    // Check if the condition is satisfied: ineq_lhs < -eps
    if (ineq_lhs < -eps_prim_inf * norm_delta_y) {
      // Compute and return ||A'delta_y|| < eps_prim_inf
      mat_tpose_vec(work->data->A, work->delta_y, work->Atdelta_y, 0, 0);

      // Unscale if necessary
      if (work->settings->scaling && !work->settings->scaled_termination) {
        vec_ew_prod(work->scaling->Dinv,
                    work->Atdelta_y,
                    work->Atdelta_y,
                    work->data->n);
      }

      return vec_norm_inf(work->Atdelta_y, work->data->n) < eps_prim_inf * norm_delta_y;
    }
  }

  // Conditions not satisfied -> not primal infeasible
  return 0;
}

c_int is_dual_infeasible(OSQPWorkspace *work, c_float eps_dual_inf) {
  // This function checks for the scaled dual infeasibility termination
  // criteria.
  //
  // 1) q * delta_x < - eps * || delta_x ||
  //
  // 2) ||P * delta_x || < eps * || delta_x ||
  //
  // 3) -> (A * delta_x)_i > -eps * || delta_x ||,    l_i != -inf
  //    -> (A * delta_x)_i <  eps * || delta_x ||,    u_i != inf
  //


  c_int   i; // Index for loops
  c_float norm_delta_x;
  c_float cost_scaling;

  // Compute norm of delta_x
  if (work->settings->scaling && !work->settings->scaled_termination) { // Unscale
                                                                        // if
                                                                        // necessary
    norm_delta_x = vec_scaled_norm_inf(work->scaling->D,
                                       work->delta_x,
                                       work->data->n);
    cost_scaling = work->scaling->c;
  } else {
    norm_delta_x = vec_norm_inf(work->delta_x, work->data->n);
    cost_scaling = 1.0;
  }

  // Prevent 0 division || delta_x || > 0
  if (norm_delta_x > eps_dual_inf) {
    // Normalize delta_x by its norm

    /* vec_mult_scalar(work->delta_x, 1./norm_delta_x, work->data->n); */

    // Check first if q'*delta_x < 0
    if (vec_prod(work->data->q, work->delta_x, work->data->n) <
        -cost_scaling * eps_dual_inf * norm_delta_x) {
      // Compute product P * delta_x (NB: P is store in upper triangular form)
      mat_vec(work->data->P, work->delta_x, work->Pdelta_x, 0);
      mat_tpose_vec(work->data->P, work->delta_x, work->Pdelta_x, 1, 1);

      // Scale if necessary
      if (work->settings->scaling && !work->settings->scaled_termination) {
        vec_ew_prod(work->scaling->Dinv,
                    work->Pdelta_x,
                    work->Pdelta_x,
                    work->data->n);
      }

      // Check if || P * delta_x || = 0
      if (vec_norm_inf(work->Pdelta_x, work->data->n) <
          cost_scaling * eps_dual_inf * norm_delta_x) {
        // Compute A * delta_x
        mat_vec(work->data->A, work->delta_x, work->Adelta_x, 0);

        // Scale if necessary
        if (work->settings->scaling && !work->settings->scaled_termination) {
          vec_ew_prod(work->scaling->Einv,
                      work->Adelta_x,
                      work->Adelta_x,
                      work->data->m);
        }

        // De Morgan Law Applied to dual infeasibility conditions for A * x
        // NB: Note that MIN_SCALING is used to adjust the infinity value
        //     in case the problem is scaled.
        for (i = 0; i < work->data->m; i++) {
          if (((work->data->u[i] < OSQP_INFTY * MIN_SCALING) &&
               (work->Adelta_x[i] >  eps_dual_inf * norm_delta_x)) ||
              ((work->data->l[i] > -OSQP_INFTY * MIN_SCALING) &&
               (work->Adelta_x[i] < -eps_dual_inf * norm_delta_x))) {
            // At least one condition not satisfied -> not dual infeasible
            return 0;
          }
        }

        // All conditions passed -> dual infeasible
        return 1;
      }
    }
  }

  // Conditions not satisfied -> not dual infeasible
  return 0;
}

c_int has_solution(OSQPInfo * info){

  return ((info->status_val != OSQP_PRIMAL_INFEASIBLE) &&
      (info->status_val != OSQP_PRIMAL_INFEASIBLE_INACCURATE) &&
      (info->status_val != OSQP_DUAL_INFEASIBLE) &&
      (info->status_val != OSQP_DUAL_INFEASIBLE_INACCURATE) &&
      (info->status_val != OSQP_NON_CVX));

}

void store_solution(OSQPWorkspace *work) {
#ifndef EMBEDDED
  c_float norm_vec;
#endif /* ifndef EMBEDDED */

  if (has_solution(work->info)) {
    prea_vec_copy(work->x, work->solution->x, work->data->n); // primal
    prea_vec_copy(work->y, work->solution->y, work->data->m); // dual

    // Unscale solution if scaling has been performed
    if (work->settings->scaling)
      unscale_solution(work);
  } else {
    // No solution present. Solution is NaN
    vec_set_scalar(work->solution->x, OSQP_NAN, work->data->n);
    vec_set_scalar(work->solution->y, OSQP_NAN, work->data->m);

#ifndef EMBEDDED

    // Normalize infeasibility certificates if embedded is off
    // NB: It requires a division
    if ((work->info->status_val == OSQP_PRIMAL_INFEASIBLE) ||
        ((work->info->status_val == OSQP_PRIMAL_INFEASIBLE_INACCURATE))) {
      norm_vec = vec_norm_inf(work->delta_y, work->data->m);
      vec_mult_scalar(work->delta_y, 1. / norm_vec, work->data->m);
    }

    if ((work->info->status_val == OSQP_DUAL_INFEASIBLE) ||
        ((work->info->status_val == OSQP_DUAL_INFEASIBLE_INACCURATE))) {
      norm_vec = vec_norm_inf(work->delta_x, work->data->n);
      vec_mult_scalar(work->delta_x, 1. / norm_vec, work->data->n);
    }

#endif /* ifndef EMBEDDED */

    // Cold start iterates to 0 for next runs (they cannot start from NaN)
    cold_start(work);
  }
}

void update_info(OSQPWorkspace *work,
                 c_int          iter,
                 c_int          compute_objective,
                 c_int          polish) {
  c_float *x, *z, *y;                   // Allocate pointers to variables
  c_float *obj_val, *pri_res, *dua_res; // objective value, residuals

#ifdef PROFILING
  c_float *run_time;                    // Execution time
#endif /* ifdef PROFILING */

#ifndef EMBEDDED

  if (polish) {
    x       = work->pol->x;
    y       = work->pol->y;
    z       = work->pol->z;
    obj_val = &work->pol->obj_val;
    pri_res = &work->pol->pri_res;
    dua_res = &work->pol->dua_res;
# ifdef PROFILING
    run_time = &work->info->polish_time;
# endif /* ifdef PROFILING */
  } else {
#endif // EMBEDDED
  x                = work->x;
  y                = work->y;
  z                = work->z;
  obj_val          = &work->info->obj_val;
  pri_res          = &work->info->pri_res;
  dua_res          = &work->info->dua_res;
  work->info->iter = iter; // Update iteration number
#ifdef PROFILING
  run_time = &work->info->solve_time;
#endif /* ifdef PROFILING */
#ifndef EMBEDDED
}

#endif /* ifndef EMBEDDED */


  // Compute the objective if needed
  if (compute_objective) {
    *obj_val = compute_obj_val(work, x);
  }

  // Compute primal residual
  if (work->data->m == 0) {
    // No constraints -> Always primal feasible
    *pri_res = 0.;
  } else {
    *pri_res = compute_pri_res(work, x, z);
  }

  // Compute dual residual
  *dua_res = compute_dua_res(work, x, y);

  // Update timing
#ifdef PROFILING
  *run_time = osqp_toc(work->timer);
#endif /* ifdef PROFILING */

#ifdef PRINTING
  work->summary_printed = 0; // The just updated info have not been printed
#endif /* ifdef PRINTING */
}


void reset_info(OSQPInfo *info) {
#ifdef PROFILING

  // Initialize info values.
  info->solve_time = 0.0;  // Solve time to zero
# ifndef EMBEDDED
  info->polish_time = 0.0; // Polish time to zero
# endif /* ifndef EMBEDDED */

  // NB: We do not reset the setup_time because it is performed only once
#endif /* ifdef PROFILING */

  update_status(info, OSQP_UNSOLVED); // Problem is unsolved

#if EMBEDDED != 1
  info->rho_updates = 0;              // Rho updates are now 0
#endif /* if EMBEDDED != 1 */
}

void update_status(OSQPInfo *info, c_int status_val) {
  // Update status value
  info->status_val = status_val;

  // Update status string depending on status val
  if (status_val == OSQP_SOLVED) c_strcpy(info->status, "solved");

  if (status_val == OSQP_SOLVED_INACCURATE) c_strcpy(info->status,
                                                     "solved inaccurate");
  else if (status_val == OSQP_PRIMAL_INFEASIBLE) c_strcpy(info->status,
                                                          "primal infeasible");
  else if (status_val == OSQP_PRIMAL_INFEASIBLE_INACCURATE) c_strcpy(info->status,
                                                                     "primal infeasible inaccurate");
  else if (status_val == OSQP_UNSOLVED) c_strcpy(info->status, "unsolved");
  else if (status_val == OSQP_DUAL_INFEASIBLE) c_strcpy(info->status,
                                                        "dual infeasible");
  else if (status_val == OSQP_DUAL_INFEASIBLE_INACCURATE) c_strcpy(info->status,
                                                                   "dual infeasible inaccurate");
  else if (status_val == OSQP_MAX_ITER_REACHED) c_strcpy(info->status,
                                                         "maximum iterations reached");
#ifdef PROFILING
  else if (status_val == OSQP_TIME_LIMIT_REACHED) c_strcpy(info->status,
                                                           "run time limit reached");
#endif /* ifdef PROFILING */
  else if (status_val == OSQP_SIGINT) c_strcpy(info->status, "interrupted");

  else if (status_val == OSQP_NON_CVX) c_strcpy(info->status, "problem non convex");

}

c_int check_termination(OSQPWorkspace *work, c_int approximate) {
  c_float eps_prim, eps_dual, eps_prim_inf, eps_dual_inf;
  c_int   exitflag;
  c_int   prim_res_check, dual_res_check, prim_inf_check, dual_inf_check;
  c_float eps_abs, eps_rel;

  // Initialize variables to 0
  exitflag       = 0;
  prim_res_check = 0; dual_res_check = 0;
  prim_inf_check = 0; dual_inf_check = 0;

  // Initialize tolerances
  eps_abs      = work->settings->eps_abs;
  eps_rel      = work->settings->eps_rel;
  eps_prim_inf = work->settings->eps_prim_inf;
  eps_dual_inf = work->settings->eps_dual_inf;

  // If residuals are too large, the problem is probably non convex
  if ((work->info->pri_res > OSQP_INFTY) ||
      (work->info->dua_res > OSQP_INFTY)){
    // Looks like residuals are diverging. Probably the problem is non convex!
    // Terminate and report it
    update_status(work->info, OSQP_NON_CVX);
    work->info->obj_val = OSQP_NAN;
    return 1;
  }

  // If approximate solution required, increase tolerances by 10
  if (approximate) {
    eps_abs      *= 10;
    eps_rel      *= 10;
    eps_prim_inf *= 10;
    eps_dual_inf *= 10;
  }

  // Check residuals
  if (work->data->m == 0) {
    prim_res_check = 1; // No constraints -> Primal feasibility always satisfied
  }
  else {
    // Compute primal tolerance
    eps_prim = compute_pri_tol(work, eps_abs, eps_rel);

    // Primal feasibility check
    if (work->info->pri_res < eps_prim) {
      prim_res_check = 1;
    } else {
      // Primal infeasibility check
      prim_inf_check = is_primal_infeasible(work, eps_prim_inf);
    }
  } // End check if m == 0

  // Compute dual tolerance
  eps_dual = compute_dua_tol(work, eps_abs, eps_rel);

  // Dual feasibility check
  if (work->info->dua_res < eps_dual) {
    dual_res_check = 1;
  } else {
    // Check dual infeasibility
    dual_inf_check = is_dual_infeasible(work, eps_dual_inf);
  }

  // Compare checks to determine solver status
  if (prim_res_check && dual_res_check) {
    // Update final information
    if (approximate) {
      update_status(work->info, OSQP_SOLVED_INACCURATE);
    } else {
      update_status(work->info, OSQP_SOLVED);
    }
    exitflag = 1;
  }
  else if (prim_inf_check) {
    // Update final information
    if (approximate) {
      update_status(work->info, OSQP_PRIMAL_INFEASIBLE_INACCURATE);
    } else {
      update_status(work->info, OSQP_PRIMAL_INFEASIBLE);
    }

    if (work->settings->scaling && !work->settings->scaled_termination) {
      // Update infeasibility certificate
      vec_ew_prod(work->scaling->E, work->delta_y, work->delta_y, work->data->m);
    }
    work->info->obj_val = OSQP_INFTY;
    exitflag            = 1;
  }
  else if (dual_inf_check) {
    // Update final information
    if (approximate) {
      update_status(work->info, OSQP_DUAL_INFEASIBLE_INACCURATE);
    } else {
      update_status(work->info, OSQP_DUAL_INFEASIBLE);
    }

    if (work->settings->scaling && !work->settings->scaled_termination) {
      // Update infeasibility certificate
      vec_ew_prod(work->scaling->D, work->delta_x, work->delta_x, work->data->n);
    }
    work->info->obj_val = -OSQP_INFTY;
    exitflag            = 1;
  }

  return exitflag;
}


#ifndef EMBEDDED

c_int validate_data(const OSQPData *data) {
  c_int j, ptr;

  if (!data) {
# ifdef PRINTING
    c_eprint("Missing data");
# endif
    return 1;
  }

  if (!(data->P)) {
# ifdef PRINTING
    c_eprint("Missing matrix P");
# endif
    return 1;
  }

  if (!(data->A)) {
# ifdef PRINTING
    c_eprint("Missing matrix A");
# endif
    return 1;
  }

  // General dimensions Tests
  if ((data->n <= 0) || (data->m < 0)) {
# ifdef PRINTING
    c_eprint("n must be positive and m nonnegative; n = %i, m = %i",
             (int)data->n, (int)data->m);
# endif /* ifdef PRINTING */
    return 1;
  }

  // Matrix P
  if (data->P->m != data->n) {
# ifdef PRINTING
    c_eprint("P does not have dimension n x n with n = %i", (int)data->n);
# endif /* ifdef PRINTING */
    return 1;
  }

  if (data->P->m != data->P->n) {
# ifdef PRINTING
    c_eprint("P is not square");
# endif /* ifdef PRINTING */
    return 1;
  }

  for (j = 0; j < data->n; j++) { // COLUMN
    for (ptr = data->P->p[j]; ptr < data->P->p[j + 1]; ptr++) {
      if (data->P->i[ptr] > j) {  // if ROW > COLUMN
# ifdef PRINTING
        c_eprint("P is not upper triangular");
# endif /* ifdef PRINTING */
        return 1;
      }
    }
  }

  // Matrix A
  if ((data->A->m != data->m) || (data->A->n != data->n)) {
# ifdef PRINTING
    c_eprint("A does not have dimension %i x %i", (int)data->m, (int)data->n);
# endif /* ifdef PRINTING */
    return 1;
  }

  // Lower and upper bounds
  for (j = 0; j < data->m; j++) {
    if (data->l[j] > data->u[j]) {
# ifdef PRINTING
      c_eprint("Lower bound at index %d is greater than upper bound: %.4e > %.4e",
               (int)j, data->l[j], data->u[j]);
# endif /* ifdef PRINTING */
      return 1;
    }
  }

  // TODO: Complete with other checks

  return 0;
}

c_int validate_linsys_solver(c_int linsys_solver) {
  if ((linsys_solver != QDLDL_SOLVER) &&
      (linsys_solver != MKL_PARDISO_SOLVER)) {
    return 1;
  }

  // TODO: Add more solvers in case

  // Valid solver
  return 0;
}

c_int validate_settings(const OSQPSettings *settings) {
  if (!settings) {
# ifdef PRINTING
    c_eprint("Missing settings!");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->scaling < 0) {
# ifdef PRINTING
    c_eprint("scaling must be nonnegative");
# endif /* ifdef PRINTING */
    return 1;
  }

  if ((settings->adaptive_rho != 0) && (settings->adaptive_rho != 1)) {
# ifdef PRINTING
    c_eprint("adaptive_rho must be either 0 or 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->adaptive_rho_interval < 0) {
# ifdef PRINTING
    c_eprint("adaptive_rho_interval must be nonnegative");
# endif /* ifdef PRINTING */
    return 1;
  }
# ifdef PROFILING

  if (settings->adaptive_rho_fraction <= 0) {
#  ifdef PRINTING
    c_eprint("adaptive_rho_fraction must be positive");
#  endif /* ifdef PRINTING */
    return 1;
  }
# endif /* ifdef PROFILING */

  if (settings->adaptive_rho_tolerance < 1.0) {
# ifdef PRINTING
    c_eprint("adaptive_rho_tolerance must be >= 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->polish_refine_iter < 0) {
# ifdef PRINTING
    c_eprint("polish_refine_iter must be nonnegative");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->rho <= 0.0) {
# ifdef PRINTING
    c_eprint("rho must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->sigma <= 0.0) {
# ifdef PRINTING
    c_eprint("sigma must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->delta <= 0.0) {
# ifdef PRINTING
    c_eprint("delta must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->max_iter <= 0) {
# ifdef PRINTING
    c_eprint("max_iter must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->eps_abs < 0.0) {
# ifdef PRINTING
    c_eprint("eps_abs must be nonnegative");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->eps_rel < 0.0) {
# ifdef PRINTING
    c_eprint("eps_rel must be nonnegative");
# endif /* ifdef PRINTING */
    return 1;
  }

  if ((settings->eps_rel == 0.0) &&
      (settings->eps_abs == 0.0)) {
# ifdef PRINTING
    c_eprint("at least one of eps_abs and eps_rel must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->eps_prim_inf <= 0.0) {
# ifdef PRINTING
    c_eprint("eps_prim_inf must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->eps_dual_inf <= 0.0) {
# ifdef PRINTING
    c_eprint("eps_dual_inf must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

  if ((settings->alpha <= 0.0) ||
      (settings->alpha >= 2.0)) {
# ifdef PRINTING
    c_eprint("alpha must be strictly between 0 and 2");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (validate_linsys_solver(settings->linsys_solver)) {
# ifdef PRINTING
    c_eprint("linsys_solver not recognized");
# endif /* ifdef PRINTING */
    return 1;
  }

  if ((settings->verbose != 0) &&
      (settings->verbose != 1)) {
# ifdef PRINTING
    c_eprint("verbose must be either 0 or 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  if ((settings->scaled_termination != 0) &&
      (settings->scaled_termination != 1)) {
# ifdef PRINTING
    c_eprint("scaled_termination must be either 0 or 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->check_termination < 0) {
# ifdef PRINTING
    c_eprint("check_termination must be nonnegative");
# endif /* ifdef PRINTING */
    return 1;
  }

  if ((settings->warm_start != 0) &&
      (settings->warm_start != 1)) {
# ifdef PRINTING
    c_eprint("warm_start must be either 0 or 1");
# endif /* ifdef PRINTING */
    return 1;
  }
# ifdef PROFILING

  if (settings->time_limit < 0.0) {
#  ifdef PRINTING
    c_eprint("time_limit must be nonnegative\n");
#  endif /* ifdef PRINTING */
    return 1;
  }
# endif /* ifdef PROFILING */

  return 0;
}

#endif // #ifndef EMBEDDED

const char *OSQP_ERROR_MESSAGE[] = {
  "Problem data validation.",
  "Solver settings validation.",
  "Linear system solver not available.\nTried to obtain it from shared library.",
  "Linear system solver initialization.",
  "KKT matrix factorization.\nThe problem seems to be non-convex.",
  "Memory allocation.",
  "Solver workspace not initialized.",
};


c_int _osqp_error(enum osqp_error_type error_code,
		 const char * function_name) {
# ifdef PRINTING
  c_print("ERROR in %s: %s\n", function_name, OSQP_ERROR_MESSAGE[error_code-1]);
# endif
  return (c_int)error_code;
}

#ifndef EMBEDDED


csc* form_KKT(const csc  *P,
              const  csc *A,
              c_int       format,
              c_float     param1,
              c_float    *param2,
              c_int      *PtoKKT,
              c_int      *AtoKKT,
              c_int     **Pdiag_idx,
              c_int      *Pdiag_n,
              c_int      *param2toKKT) {
  c_int  nKKT, nnzKKTmax; // Size, number of nonzeros and max number of nonzeros
                          // in KKT matrix
  csc   *KKT_trip, *KKT;  // KKT matrix in triplet format and CSC format
  c_int  ptr, i, j;       // Counters for elements (i,j) and index pointer
  c_int  zKKT = 0;        // Counter for total number of elements in P and in
                          // KKT
  c_int *KKT_TtoC;        // Pointer to vector mapping from KKT in triplet form
                          // to CSC

  // Get matrix dimensions
  nKKT = P->m + A->m;

  // Get maximum number of nonzero elements (only upper triangular part)
  nnzKKTmax = P->p[P->n] + // Number of elements in P
              P->m +       // Number of elements in param1 * I
              A->p[A->n] + // Number of nonzeros in A
              A->m;        // Number of elements in - diag(param2)

  // Preallocate KKT matrix in triplet format
  KKT_trip = csc_spalloc(nKKT, nKKT, nnzKKTmax, 1, 1);

  if (!KKT_trip) return OSQP_NULL;  // Failed to preallocate matrix

  // Allocate vector of indices on the diagonal. Worst case it has m elements
  if (Pdiag_idx != OSQP_NULL) {
    (*Pdiag_idx) = c_malloc(P->m * sizeof(c_int));
    *Pdiag_n     = 0; // Set 0 diagonal elements to start
  }

  // Allocate Triplet matrices
  // P + param1 I
  for (j = 0; j < P->n; j++) { // cycle over columns
    // No elements in column j => add diagonal element param1
    if (P->p[j] == P->p[j + 1]) {
      KKT_trip->i[zKKT] = j;
      KKT_trip->p[zKKT] = j;
      KKT_trip->x[zKKT] = param1;
      zKKT++;
    }

    for (ptr = P->p[j]; ptr < P->p[j + 1]; ptr++) { // cycle over rows
      // Get current row
      i = P->i[ptr];

      // Add element of P
      KKT_trip->i[zKKT] = i;
      KKT_trip->p[zKKT] = j;
      KKT_trip->x[zKKT] = P->x[ptr];

      if (PtoKKT != OSQP_NULL) PtoKKT[ptr] = zKKT;  // Update index from P to
                                                    // KKTtrip

      if (i == j) {                                 // P has a diagonal element,
                                                    // add param1
        KKT_trip->x[zKKT] += param1;

        // If index vector pointer supplied -> Store the index
        if (Pdiag_idx != OSQP_NULL) {
          (*Pdiag_idx)[*Pdiag_n] = ptr;
          (*Pdiag_n)++;
        }
      }
      zKKT++;

      // Add diagonal param1 in case
      if ((i < j) &&                  // Diagonal element not reached
          (ptr + 1 == P->p[j + 1])) { // last element of column j
        // Add diagonal element param1
        KKT_trip->i[zKKT] = j;
        KKT_trip->p[zKKT] = j;
        KKT_trip->x[zKKT] = param1;
        zKKT++;
      }
    }
  }

  if (Pdiag_idx != OSQP_NULL) {
    // Realloc Pdiag_idx so that it contains exactly *Pdiag_n diagonal elements
    (*Pdiag_idx) = c_realloc((*Pdiag_idx), (*Pdiag_n) * sizeof(c_int));
  }


  // A' at top right
  for (j = 0; j < A->n; j++) {                      // Cycle over columns of A
    for (ptr = A->p[j]; ptr < A->p[j + 1]; ptr++) {
      KKT_trip->p[zKKT] = P->m + A->i[ptr];         // Assign column index from
                                                    // row index of A
      KKT_trip->i[zKKT] = j;                        // Assign row index from
                                                    // column index of A
      KKT_trip->x[zKKT] = A->x[ptr];                // Assign A value element

      if (AtoKKT != OSQP_NULL) AtoKKT[ptr] = zKKT;  // Update index from A to
                                                    // KKTtrip
      zKKT++;
    }
  }

  // - diag(param2) at bottom right
  for (j = 0; j < A->m; j++) {
    KKT_trip->i[zKKT] = j + P->n;
    KKT_trip->p[zKKT] = j + P->n;
    KKT_trip->x[zKKT] = -param2[j];

    if (param2toKKT != OSQP_NULL) param2toKKT[j] = zKKT;  // Update index from
                                                          // param2 to KKTtrip
    zKKT++;
  }

  // Allocate number of nonzeros
  KKT_trip->nz = zKKT;

  // Convert triplet matrix to csc format
  if (!PtoKKT && !AtoKKT && !param2toKKT) {
    // If no index vectors passed, do not store KKT mapping from Trip to CSC/CSR
    if (format == 0) KKT = triplet_to_csc(KKT_trip, OSQP_NULL);
    else KKT = triplet_to_csr(KKT_trip, OSQP_NULL);
  }
  else {
    // Allocate vector of indices from triplet to csc
    KKT_TtoC = c_malloc((zKKT) * sizeof(c_int));

    if (!KKT_TtoC) {
      // Error in allocating KKT_TtoC vector
      csc_spfree(KKT_trip);
      c_free(*Pdiag_idx);
      return OSQP_NULL;
    }

    // Store KKT mapping from Trip to CSC/CSR
    if (format == 0)
      KKT = triplet_to_csc(KKT_trip, KKT_TtoC);
    else
      KKT = triplet_to_csr(KKT_trip, KKT_TtoC);

    // Update vectors of indices from P, A, param2 to KKT (now in CSC format)
    if (PtoKKT != OSQP_NULL) {
      for (i = 0; i < P->p[P->n]; i++) {
        PtoKKT[i] = KKT_TtoC[PtoKKT[i]];
      }
    }

    if (AtoKKT != OSQP_NULL) {
      for (i = 0; i < A->p[A->n]; i++) {
        AtoKKT[i] = KKT_TtoC[AtoKKT[i]];
      }
    }

    if (param2toKKT != OSQP_NULL) {
      for (i = 0; i < A->m; i++) {
        param2toKKT[i] = KKT_TtoC[param2toKKT[i]];
      }
    }

    // Free mapping
    c_free(KKT_TtoC);
  }

  // Clean matrix in triplet format and return result
  csc_spfree(KKT_trip);

  return KKT;
}

#endif /* ifndef EMBEDDED */


#if EMBEDDED != 1

void update_KKT_P(csc          *KKT,
                  const csc    *P,
                  const c_int  *PtoKKT,
                  const c_float param1,
                  const c_int  *Pdiag_idx,
                  const c_int   Pdiag_n) {
  c_int i, j; // Iterations

  // Update elements of KKT using P
  for (i = 0; i < P->p[P->n]; i++) {
    KKT->x[PtoKKT[i]] = P->x[i];
  }

  // Update diagonal elements of KKT by adding sigma
  for (i = 0; i < Pdiag_n; i++) {
    j                  = Pdiag_idx[i]; // Extract index of the element on the
                                       // diagonal
    KKT->x[PtoKKT[j]] += param1;
  }
}

void update_KKT_A(csc *KKT, const csc *A, const c_int *AtoKKT) {
  c_int i; // Iterations

  // Update elements of KKT using A
  for (i = 0; i < A->p[A->n]; i++) {
    KKT->x[AtoKKT[i]] = A->x[i];
  }
}

void update_KKT_param2(csc *KKT, const c_float *param2,
                       const c_int *param2toKKT, const c_int m) {
  c_int i; // Iterations

  // Update elements of KKT using param2
  for (i = 0; i < m; i++) {
    KKT->x[param2toKKT[i]] = -param2[i];
  }
}

#endif // EMBEDDED != 1

/* VECTOR FUNCTIONS ----------------------------------------------------------*/


void vec_add_scaled(c_float       *c,
                    const c_float *a,
                    const c_float *b,
                    c_int          n,
                    c_float        sc) {
  c_int i;

  for (i = 0; i < n; i++) {
    c[i] =  a[i] + sc * b[i];
  }
}

c_float vec_scaled_norm_inf(const c_float *S, const c_float *v, c_int l) {
  c_int   i;
  c_float abs_Sv_i;
  c_float max = 0.0;

  for (i = 0; i < l; i++) {
    abs_Sv_i = c_absval(S[i] * v[i]);

    if (abs_Sv_i > max) max = abs_Sv_i;
  }
  return max;
}

c_float vec_norm_inf(const c_float *v, c_int l) {
  c_int   i;
  c_float abs_v_i;
  c_float max = 0.0;

  for (i = 0; i < l; i++) {
    abs_v_i = c_absval(v[i]);

    if (abs_v_i > max) max = abs_v_i;
  }
  return max;
}

c_float vec_norm_inf_diff(const c_float *a, const c_float *b, c_int l) {
  c_float nmDiff = 0.0, tmp;
  c_int   i;

  for (i = 0; i < l; i++) {
    tmp = c_absval(a[i] - b[i]);

    if (tmp > nmDiff) nmDiff = tmp;
  }
  return nmDiff;
}

c_float vec_mean(const c_float *a, c_int n) {
  c_float mean = 0.0;
  c_int   i;

  for (i = 0; i < n; i++) {
    mean += a[i];
  }
  mean /= (c_float)n;

  return mean;
}

void int_vec_set_scalar(c_int *a, c_int sc, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    a[i] = sc;
  }
}

void vec_set_scalar(c_float *a, c_float sc, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    a[i] = sc;
  }
}

void vec_add_scalar(c_float *a, c_float sc, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    a[i] += sc;
  }
}

void vec_mult_scalar(c_float *a, c_float sc, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    a[i] *= sc;
  }
}

#ifndef EMBEDDED
c_float* vec_copy(c_float *a, c_int n) {
  c_float *b;
  c_int    i;

  b = c_malloc(n * sizeof(c_float));
  if (!b) return OSQP_NULL;

  for (i = 0; i < n; i++) {
    b[i] = a[i];
  }

  return b;
}

#endif // end EMBEDDED


void prea_int_vec_copy(const c_int *a, c_int *b, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    b[i] = a[i];
  }
}

void prea_vec_copy(const c_float *a, c_float *b, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    b[i] = a[i];
  }
}

void vec_ew_recipr(const c_float *a, c_float *b, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    b[i] = (c_float)1.0 / a[i];
  }
}

c_float vec_prod(const c_float *a, const c_float *b, c_int n) {
  c_float prod = 0.0;
  c_int   i; // Index

  for (i = 0; i < n; i++) {
    prod += a[i] * b[i];
  }

  return prod;
}

void vec_ew_prod(const c_float *a, const c_float *b, c_float *c, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    c[i] = b[i] * a[i];
  }
}

#if EMBEDDED != 1
void vec_ew_sqrt(c_float *a, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    a[i] = c_sqrt(a[i]);
  }
}

void vec_ew_max(c_float *a, c_int n, c_float max_val) {
  c_int i;

  for (i = 0; i < n; i++) {
    a[i] = c_max(a[i], max_val);
  }
}

void vec_ew_min(c_float *a, c_int n, c_float min_val) {
  c_int i;

  for (i = 0; i < n; i++) {
    a[i] = c_min(a[i], min_val);
  }
}

void vec_ew_max_vec(const c_float *a, const c_float *b, c_float *c, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    c[i] = c_max(a[i], b[i]);
  }
}

void vec_ew_min_vec(const c_float *a, const c_float *b, c_float *c, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    c[i] = c_min(a[i], b[i]);
  }
}

#endif // EMBEDDED != 1


/* MATRIX FUNCTIONS ----------------------------------------------------------*/

/* multiply scalar to matrix */
void mat_mult_scalar(csc *A, c_float sc) {
  c_int i, nnzA;

  nnzA = A->p[A->n];

  for (i = 0; i < nnzA; i++) {
    A->x[i] *= sc;
  }
}

void mat_premult_diag(csc *A, const c_float *d) {
  c_int j, i;

  for (j = 0; j < A->n; j++) {                // Cycle over columns
    for (i = A->p[j]; i < A->p[j + 1]; i++) { // Cycle every row in the column
      A->x[i] *= d[A->i[i]];                  // Scale by corresponding element
                                              // of d for row i
    }
  }
}

void mat_postmult_diag(csc *A, const c_float *d) {
  c_int j, i;

  for (j = 0; j < A->n; j++) {                // Cycle over columns j
    for (i = A->p[j]; i < A->p[j + 1]; i++) { // Cycle every row i in column j
      A->x[i] *= d[j];                        // Scale by corresponding element
                                              // of d for column j
    }
  }
}

void mat_vec(const csc *A, const c_float *x, c_float *y, c_int plus_eq) {
  c_int i, j;

  if (!plus_eq) {
    // y = 0
    for (i = 0; i < A->m; i++) {
      y[i] = 0;
    }
  }

  // if A is empty
  if (A->p[A->n] == 0) {
    return;
  }

  if (plus_eq == -1) {
    // y -=  A*x
    for (j = 0; j < A->n; j++) {
      for (i = A->p[j]; i < A->p[j + 1]; i++) {
        y[A->i[i]] -= A->x[i] * x[j];
      }
    }
  } else {
    // y +=  A*x
    for (j = 0; j < A->n; j++) {
      for (i = A->p[j]; i < A->p[j + 1]; i++) {
        y[A->i[i]] += A->x[i] * x[j];
      }
    }
  }
}

void mat_tpose_vec(const csc *A, const c_float *x, c_float *y,
                   c_int plus_eq, c_int skip_diag) {
  c_int i, j, k;

  if (!plus_eq) {
    // y = 0
    for (i = 0; i < A->n; i++) {
      y[i] = 0;
    }
  }

  // if A is empty
  if (A->p[A->n] == 0) {
    return;
  }

  if (plus_eq == -1) {
    // y -=  A*x
    if (skip_diag) {
      for (j = 0; j < A->n; j++) {
        for (k = A->p[j]; k < A->p[j + 1]; k++) {
          i     = A->i[k];
          y[j] -= i == j ? 0 : A->x[k] * x[i];
        }
      }
    } else {
      for (j = 0; j < A->n; j++) {
        for (k = A->p[j]; k < A->p[j + 1]; k++) {
          y[j] -= A->x[k] * x[A->i[k]];
        }
      }
    }
  } else {
    // y +=  A*x
    if (skip_diag) {
      for (j = 0; j < A->n; j++) {
        for (k = A->p[j]; k < A->p[j + 1]; k++) {
          i     = A->i[k];
          y[j] += i == j ? 0 : A->x[k] * x[i];
        }
      }
    } else {
      for (j = 0; j < A->n; j++) {
        for (k = A->p[j]; k < A->p[j + 1]; k++) {
          y[j] += A->x[k] * x[A->i[k]];
        }
      }
    }
  }
}

#if EMBEDDED != 1
void mat_inf_norm_cols(const csc *M, c_float *E) {
  c_int j, ptr;

  // Initialize zero max elements
  for (j = 0; j < M->n; j++) {
    E[j] = 0.;
  }

  // Compute maximum across columns
  for (j = 0; j < M->n; j++) {
    for (ptr = M->p[j]; ptr < M->p[j + 1]; ptr++) {
      E[j] = c_max(c_absval(M->x[ptr]), E[j]);
    }
  }
}

void mat_inf_norm_rows(const csc *M, c_float *E) {
  c_int i, j, ptr;

  // Initialize zero max elements
  for (j = 0; j < M->m; j++) {
    E[j] = 0.;
  }

  // Compute maximum across rows
  for (j = 0; j < M->n; j++) {
    for (ptr = M->p[j]; ptr < M->p[j + 1]; ptr++) {
      i    = M->i[ptr];
      E[i] = c_max(c_absval(M->x[ptr]), E[i]);
    }
  }
}

void mat_inf_norm_cols_sym_triu(const csc *M, c_float *E) {
  c_int   i, j, ptr;
  c_float abs_x;

  // Initialize zero max elements
  for (j = 0; j < M->n; j++) {
    E[j] = 0.;
  }

  // Compute maximum across columns
  // Note that element (i, j) contributes to
  // -> Column j (as expected in any matrices)
  // -> Column i (which is equal to row i for symmetric matrices)
  for (j = 0; j < M->n; j++) {
    for (ptr = M->p[j]; ptr < M->p[j + 1]; ptr++) {
      i     = M->i[ptr];
      abs_x = c_absval(M->x[ptr]);
      E[j]  = c_max(abs_x, E[j]);

      if (i != j) {
        E[i] = c_max(abs_x, E[i]);
      }
    }
  }
}

#endif /* if EMBEDDED != 1 */


c_float quad_form(const csc *P, const c_float *x) {
  c_float quad_form = 0.;
  c_int   i, j, ptr;                                // Pointers to iterate over
                                                    // matrix: (i,j) a element
                                                    // pointer

  for (j = 0; j < P->n; j++) {                      // Iterate over columns
    for (ptr = P->p[j]; ptr < P->p[j + 1]; ptr++) { // Iterate over rows
      i = P->i[ptr];                                // Row index

      if (i == j) {                                 // Diagonal element
        quad_form += (c_float).5 * P->x[ptr] * x[i] * x[i];
      }
      else if (i < j) {                             // Off-diagonal element
        quad_form += P->x[ptr] * x[i] * x[j];
      }
      else {                                        // Element in lower diagonal
                                                    // part
#ifdef PRINTING
        c_eprint("quad_form matrix is not upper triangular");
#endif /* ifdef PRINTING */
        return OSQP_NULL;
      }
    }
  }
  return quad_form;
}

#ifndef EMBEDDED
# include "polish.h"
#endif /* ifndef EMBEDDED */

#ifdef CTRLC
# include "ctrlc.h"
#endif /* ifdef CTRLC */

#ifndef EMBEDDED
# include "lin_sys.h"
#endif /* ifndef EMBEDDED */

/**********************
* Main API Functions *
**********************/
void osqp_set_default_settings(OSQPSettings *settings) {

  settings->rho           = (c_float)RHO;            /* ADMM step */
  settings->sigma         = (c_float)SIGMA;          /* ADMM step */
  settings->scaling = SCALING;                       /* heuristic problem scaling */
#if EMBEDDED != 1
  settings->adaptive_rho           = ADAPTIVE_RHO;
  settings->adaptive_rho_interval  = ADAPTIVE_RHO_INTERVAL;
  settings->adaptive_rho_tolerance = (c_float)ADAPTIVE_RHO_TOLERANCE;

# ifdef PROFILING
  settings->adaptive_rho_fraction = (c_float)ADAPTIVE_RHO_FRACTION;
# endif /* ifdef PROFILING */
#endif  /* if EMBEDDED != 1 */

  settings->max_iter      = MAX_ITER;                /* maximum iterations to
                                                        take */
  settings->eps_abs       = (c_float)EPS_ABS;        /* absolute convergence
                                                        tolerance */
  settings->eps_rel       = (c_float)EPS_REL;        /* relative convergence
                                                        tolerance */
  settings->eps_prim_inf  = (c_float)EPS_PRIM_INF;   /* primal infeasibility
                                                        tolerance */
  settings->eps_dual_inf  = (c_float)EPS_DUAL_INF;   /* dual infeasibility
                                                        tolerance */
  settings->alpha         = (c_float)ALPHA;          /* relaxation parameter */
  settings->linsys_solver = LINSYS_SOLVER;           /* relaxation parameter */

#ifndef EMBEDDED
  settings->delta              = DELTA;              /* regularization parameter
                                                        for polish */
  settings->polish             = POLISH;             /* ADMM solution polish: 1
                                                      */
  settings->polish_refine_iter = POLISH_REFINE_ITER; /* iterative refinement
                                                        steps in polish */
  settings->verbose            = VERBOSE;            /* print output */
#endif /* ifndef EMBEDDED */

  settings->scaled_termination = SCALED_TERMINATION; /* Evaluate scaled
                                                        termination criteria*/
  settings->check_termination  = CHECK_TERMINATION;  /* Interval for evaluating
                                                        termination criteria */
  settings->warm_start         = WARM_START;         /* warm starting */

#ifdef PROFILING
  settings->time_limit = TIME_LIMIT;
#endif /* ifdef PROFILING */
}

#ifndef EMBEDDED


c_int osqp_setup(OSQPWorkspace** workp, const OSQPData *data, const OSQPSettings *settings) {
  c_int exitflag;

  OSQPWorkspace * work;

  // Validate data
  if (validate_data(data)) return osqp_error(OSQP_DATA_VALIDATION_ERROR);

  // Validate settings
  if (validate_settings(settings)) return osqp_error(OSQP_SETTINGS_VALIDATION_ERROR);

  // Allocate empty workspace
  work = c_calloc(1, sizeof(OSQPWorkspace));
  if (!(work)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  *workp = work;

  // Start and allocate directly timer
# ifdef PROFILING
  work->timer = c_malloc(sizeof(OSQPTimer));
  if (!(work->timer)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  osqp_tic(work->timer);
# endif /* ifdef PROFILING */

  // Copy problem data into workspace
  work->data = c_malloc(sizeof(OSQPData));
  if (!(work->data)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->data->n = data->n;
  work->data->m = data->m;

  // Cost function
  work->data->P = copy_csc_mat(data->P);
  work->data->q = vec_copy(data->q, data->n);
  if (!(work->data->P) || !(work->data->q)) return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Constraints
  work->data->A = copy_csc_mat(data->A);
  if (!(work->data->A)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->data->l = vec_copy(data->l, data->m);
  work->data->u = vec_copy(data->u, data->m);
  if ( data->m && (!(work->data->l) || !(work->data->u)) )
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Vectorized rho parameter
  work->rho_vec     = c_malloc(data->m * sizeof(c_float));
  work->rho_inv_vec = c_malloc(data->m * sizeof(c_float));
  if ( data->m && (!(work->rho_vec) || !(work->rho_inv_vec)) )
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Type of constraints
  work->constr_type = c_calloc(data->m, sizeof(c_int));
  if (data->m && !(work->constr_type)) return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Allocate internal solver variables (ADMM steps)
  work->x        = c_calloc(data->n, sizeof(c_float));
  work->z        = c_calloc(data->m, sizeof(c_float));
  work->xz_tilde = c_calloc(data->n + data->m, sizeof(c_float));
  work->x_prev   = c_calloc(data->n, sizeof(c_float));
  work->z_prev   = c_calloc(data->m, sizeof(c_float));
  work->y        = c_calloc(data->m, sizeof(c_float));
  if (!(work->x) || !(work->xz_tilde) || !(work->x_prev))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if ( data->m && (!(work->z) || !(work->z_prev) || !(work->y)) )
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Initialize variables x, y, z to 0
  cold_start(work);

  // Primal and dual residuals variables
  work->Ax  = c_calloc(data->m, sizeof(c_float));
  work->Px  = c_calloc(data->n, sizeof(c_float));
  work->Aty = c_calloc(data->n, sizeof(c_float));

  // Primal infeasibility variables
  work->delta_y   = c_calloc(data->m, sizeof(c_float));
  work->Atdelta_y = c_calloc(data->n, sizeof(c_float));

  // Dual infeasibility variables
  work->delta_x  = c_calloc(data->n, sizeof(c_float));
  work->Pdelta_x = c_calloc(data->n, sizeof(c_float));
  work->Adelta_x = c_calloc(data->m, sizeof(c_float));

  if (!(work->Px) || !(work->Aty) || !(work->Atdelta_y) ||
      !(work->delta_x) || !(work->Pdelta_x))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if ( data->m && (!(work->Ax) || !(work->delta_y) || !(work->Adelta_x)) )
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Copy settings
  work->settings = copy_settings(settings);
  if (!(work->settings)) return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Perform scaling
  if (settings->scaling) {
    // Allocate scaling structure
    work->scaling = c_malloc(sizeof(OSQPScaling));
    if (!(work->scaling)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
    work->scaling->D    = c_malloc(data->n * sizeof(c_float));
    work->scaling->Dinv = c_malloc(data->n * sizeof(c_float));
    work->scaling->E    = c_malloc(data->m * sizeof(c_float));
    work->scaling->Einv = c_malloc(data->m * sizeof(c_float));
    if (!(work->scaling->D) || !(work->scaling->Dinv))
      return osqp_error(OSQP_MEM_ALLOC_ERROR);
    if ( data->m && (!(work->scaling->E) || !(work->scaling->Einv)) )
      return osqp_error(OSQP_MEM_ALLOC_ERROR);


    // Allocate workspace variables used in scaling
    work->D_temp   = c_malloc(data->n * sizeof(c_float));
    work->D_temp_A = c_malloc(data->n * sizeof(c_float));
    work->E_temp   = c_malloc(data->m * sizeof(c_float));
    // if (!(work->D_temp) || !(work->D_temp_A) || !(work->E_temp))
    //   return osqp_error(OSQP_MEM_ALLOC_ERROR);
    if (!(work->D_temp) || !(work->D_temp_A)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
    if (data->m && !(work->E_temp))           return osqp_error(OSQP_MEM_ALLOC_ERROR);

    // Scale data
    scale_data(work);
  } else {
    work->scaling  = OSQP_NULL;
    work->D_temp   = OSQP_NULL;
    work->D_temp_A = OSQP_NULL;
    work->E_temp   = OSQP_NULL;
  }

  // Set type of constraints
  set_rho_vec(work);

  // Load linear system solver
  if (load_linsys_solver(work->settings->linsys_solver)) return osqp_error(OSQP_LINSYS_SOLVER_LOAD_ERROR);

  // Initialize linear system solver structure
  exitflag = init_linsys_solver(&(work->linsys_solver), work->data->P, work->data->A,
                                work->settings->sigma, work->rho_vec,
                                work->settings->linsys_solver, 0);

  if (exitflag) {
    return osqp_error(exitflag);
  }

  // Initialize active constraints structure
  work->pol = c_malloc(sizeof(OSQPPolish));
  if (!(work->pol)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->pol->Alow_to_A = c_malloc(data->m * sizeof(c_int));
  work->pol->Aupp_to_A = c_malloc(data->m * sizeof(c_int));
  work->pol->A_to_Alow = c_malloc(data->m * sizeof(c_int));
  work->pol->A_to_Aupp = c_malloc(data->m * sizeof(c_int));
  work->pol->x         = c_malloc(data->n * sizeof(c_float));
  work->pol->z         = c_malloc(data->m * sizeof(c_float));
  work->pol->y         = c_malloc(data->m * sizeof(c_float));
  if (!(work->pol->x)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if ( data->m && (!(work->pol->Alow_to_A) || !(work->pol->Aupp_to_A) ||
      !(work->pol->A_to_Alow) || !(work->pol->A_to_Aupp) ||
      !(work->pol->z) || !(work->pol->y)) )
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Allocate solution
  work->solution = c_calloc(1, sizeof(OSQPSolution));
  if (!(work->solution)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->solution->x = c_calloc(1, data->n * sizeof(c_float));
  work->solution->y = c_calloc(1, data->m * sizeof(c_float));
  if (!(work->solution->x))            return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if (data->m && !(work->solution->y)) return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Allocate and initialize information
  work->info = c_calloc(1, sizeof(OSQPInfo));
  if (!(work->info)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->info->status_polish = 0;              // Polishing not performed
  update_status(work->info, OSQP_UNSOLVED);
# ifdef PROFILING
  work->info->solve_time  = 0.0;                   // Solve time to zero
  work->info->update_time = 0.0;                   // Update time to zero
  work->info->polish_time = 0.0;                   // Polish time to zero
  work->info->run_time    = 0.0;                   // Total run time to zero
  work->info->setup_time  = osqp_toc(work->timer); // Update timer information

  work->first_run         = 1;
  work->clear_update_time = 0;
  work->rho_update_from_solve = 0;
# endif /* ifdef PROFILING */
  work->info->rho_updates  = 0;                    // Rho updates set to 0
  work->info->rho_estimate = work->settings->rho;  // Best rho estimate

  // Print header
# ifdef PRINTING
  if (work->settings->verbose) print_setup_header(work);
  work->summary_printed = 0; // Initialize last summary  to not printed
# endif /* ifdef PRINTING */


  // If adaptive rho and automatic interval, but profiling disabled, we need to
  // set the interval to a default value
# ifndef PROFILING
  if (work->settings->adaptive_rho && !work->settings->adaptive_rho_interval) {
    if (work->settings->check_termination) {
      // If check_termination is enabled, we set it to a multiple of the check
      // termination interval
      work->settings->adaptive_rho_interval = ADAPTIVE_RHO_MULTIPLE_TERMINATION *
                                              work->settings->check_termination;
    } else {
      // If check_termination is disabled we set it to a predefined fix number
      work->settings->adaptive_rho_interval = ADAPTIVE_RHO_FIXED;
    }
  }
# endif /* ifndef PROFILING */

  // Return exit flag
  return 0;
}

#endif // #ifndef EMBEDDED


c_int osqp_solve(OSQPWorkspace *work) {

  c_int exitflag;
  c_int iter;
  c_int compute_cost_function; // Boolean: compute the cost function in the loop or not
  c_int can_check_termination; // Boolean: check termination or not

#ifdef PROFILING
  c_float temp_run_time;       // Temporary variable to store current run time
#endif /* ifdef PROFILING */

#ifdef PRINTING
  c_int can_print;             // Boolean whether you can print
#endif /* ifdef PRINTING */

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

#ifdef PROFILING
  if (work->clear_update_time == 1)
    work->info->update_time = 0.0;
  work->rho_update_from_solve = 1;
#endif /* ifdef PROFILING */

  // Initialize variables
  exitflag              = 0;
  can_check_termination = 0;
#ifdef PRINTING
  can_print = work->settings->verbose;
#endif /* ifdef PRINTING */
#ifdef PRINTING
  compute_cost_function = work->settings->verbose; // Compute cost function only
                                                   // if verbose is on
#else /* ifdef PRINTING */
  compute_cost_function = 0;                       // Never compute cost
                                                   // function during the
                                                   // iterations if no printing
                                                   // enabled
#endif /* ifdef PRINTING */



#ifdef PROFILING
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */


#ifdef PRINTING

  if (work->settings->verbose) {
    // Print Header for every column
    print_header();
  }
#endif /* ifdef PRINTING */

#ifdef CTRLC

  // initialize Ctrl-C support
  osqp_start_interrupt_listener();
#endif /* ifdef CTRLC */

  // Initialize variables (cold start or warm start depending on settings)
  if (!work->settings->warm_start) cold_start(work);  // If not warm start ->
                                                      // set x, z, y to zero

  // Main ADMM algorithm
  for (iter = 1; iter <= work->settings->max_iter; iter++) {
    // Update x_prev, z_prev (preallocated, no malloc)
    swap_vectors(&(work->x), &(work->x_prev));
    swap_vectors(&(work->z), &(work->z_prev));

    /* ADMM STEPS */
    /* Compute \tilde{x}^{k+1}, \tilde{z}^{k+1} */
    update_xz_tilde(work);

    /* Compute x^{k+1} */
    update_x(work);

    /* Compute z^{k+1} */
    update_z(work);

    /* Compute y^{k+1} */
    update_y(work);

    /* End of ADMM Steps */

#ifdef CTRLC

    // Check the interrupt signal
    if (osqp_is_interrupted()) {
      update_status(work->info, OSQP_SIGINT);
# ifdef PRINTING
      c_print("Solver interrupted\n");
# endif /* ifdef PRINTING */
      exitflag = 1;
      goto exit;
    }
#endif /* ifdef CTRLC */

#ifdef PROFILING

    // Check if solver time_limit is enabled. In case, check if the current
    // run time is more than the time_limit option.
    if (work->first_run) {
      temp_run_time = work->info->setup_time + osqp_toc(work->timer);
    }
    else {
      temp_run_time = work->info->update_time + osqp_toc(work->timer);
    }

    if (work->settings->time_limit &&
        (temp_run_time >= work->settings->time_limit)) {
      update_status(work->info, OSQP_TIME_LIMIT_REACHED);
# ifdef PRINTING
      if (work->settings->verbose) c_print("run time limit reached\n");
      can_print = 0;  // Not printing at this iteration
# endif /* ifdef PRINTING */
      break;
    }
#endif /* ifdef PROFILING */


    // Can we check for termination ?
    can_check_termination = work->settings->check_termination &&
                            (iter % work->settings->check_termination == 0);

#ifdef PRINTING

    // Can we print ?
    can_print = work->settings->verbose &&
                ((iter % PRINT_INTERVAL == 0) || (iter == 1));

    if (can_check_termination || can_print) { // Update status in either of
                                              // these cases
      // Update information
      update_info(work, iter, compute_cost_function, 0);

      if (can_print) {
        // Print summary
        print_summary(work);
      }

      if (can_check_termination) {
        // Check algorithm termination
        if (check_termination(work, 0)) {
          // Terminate algorithm
          break;
        }
      }
    }
#else /* ifdef PRINTING */

    if (can_check_termination) {
      // Update information and compute also objective value
      update_info(work, iter, compute_cost_function, 0);

      // Check algorithm termination
      if (check_termination(work, 0)) {
        // Terminate algorithm
        break;
      }
    }
#endif /* ifdef PRINTING */


#if EMBEDDED != 1
# ifdef PROFILING

    // If adaptive rho with automatic interval, check if the solve time is a
    // certain fraction
    // of the setup time.
    if (work->settings->adaptive_rho && !work->settings->adaptive_rho_interval) {
      // Check time
      if (osqp_toc(work->timer) >
          work->settings->adaptive_rho_fraction * work->info->setup_time) {
        // Enough time has passed. We now get the number of iterations between
        // the updates.
        if (work->settings->check_termination) {
          // If check_termination is enabled, we round the number of iterations
          // between
          // rho updates to the closest multiple of check_termination
          work->settings->adaptive_rho_interval = (c_int)c_roundmultiple(iter,
                                                                         work->settings->check_termination);
        } else {
          // If check_termination is disabled, we round the number of iterations
          // between
          // updates to the closest multiple of the default check_termination
          // interval.
          work->settings->adaptive_rho_interval = (c_int)c_roundmultiple(iter,
                                                                         CHECK_TERMINATION);
        }

        // Make sure the interval is not 0 and at least check_termination times
        work->settings->adaptive_rho_interval = c_max(
          work->settings->adaptive_rho_interval,
          work->settings->check_termination);
      } // If time condition is met
    }   // If adaptive rho enabled and interval set to auto
# endif // PROFILING

    // Adapt rho
    if (work->settings->adaptive_rho &&
        work->settings->adaptive_rho_interval &&
        (iter % work->settings->adaptive_rho_interval == 0)) {
      // Update info with the residuals if it hasn't been done before
# ifdef PRINTING

      if (!can_check_termination && !can_print) {
        // Information has not been computed neither for termination or printing
        // reasons
        update_info(work, iter, compute_cost_function, 0);
      }
# else /* ifdef PRINTING */

      if (!can_check_termination) {
        // Information has not been computed before for termination check
        update_info(work, iter, compute_cost_function, 0);
      }
# endif /* ifdef PRINTING */

      // Actually update rho
      if (adapt_rho(work)) {
# ifdef PRINTING
        c_eprint("Failed rho update");
# endif // PRINTING
        exitflag = 1;
        goto exit;
      }
    }
#endif // EMBEDDED != 1

  }        // End of ADMM for loop


  // Update information and check termination condition if it hasn't been done
  // during last iteration (max_iter reached or check_termination disabled)
  if (!can_check_termination) {
    /* Update information */
#ifdef PRINTING

    if (!can_print) {
      // Update info only if it hasn't been updated before for printing
      // reasons
      update_info(work, iter - 1, compute_cost_function, 0);
    }
#else /* ifdef PRINTING */

    // If no printing is enabled, update info directly
    update_info(work, iter - 1, compute_cost_function, 0);
#endif /* ifdef PRINTING */

#ifdef PRINTING

    /* Print summary */
    if (work->settings->verbose && !work->summary_printed) print_summary(work);
#endif /* ifdef PRINTING */

    /* Check whether a termination criterion is triggered */
    check_termination(work, 0);
  }

  // Compute objective value in case it was not
  // computed during the iterations
  if (!compute_cost_function && has_solution(work->info)){
    work->info->obj_val = compute_obj_val(work, work->x);
  }


#ifdef PRINTING
  /* Print summary for last iteration */
  if (work->settings->verbose && !work->summary_printed) {
    print_summary(work);
  }
#endif /* ifdef PRINTING */

  /* if max iterations reached, change status accordingly */
  if (work->info->status_val == OSQP_UNSOLVED) {
    if (!check_termination(work, 1)) { // Try to check for approximate
      update_status(work->info, OSQP_MAX_ITER_REACHED);
    }
  }

#ifdef PROFILING
  /* if time-limit reached check termination and update status accordingly */
 if (work->info->status_val == OSQP_TIME_LIMIT_REACHED) {
    if (!check_termination(work, 1)) { // Try for approximate solutions
      update_status(work->info, OSQP_TIME_LIMIT_REACHED); /* Change update status back to OSQP_TIME_LIMIT_REACHED */
    }
  }
#endif /* ifdef PROFILING */


#if EMBEDDED != 1
  /* Update rho estimate */
  work->info->rho_estimate = compute_rho_estimate(work);
#endif /* if EMBEDDED != 1 */

  /* Update solve time */
#ifdef PROFILING
  work->info->solve_time = osqp_toc(work->timer);
#endif /* ifdef PROFILING */


#ifndef EMBEDDED
  // Polish the obtained solution
  if (work->settings->polish && (work->info->status_val == OSQP_SOLVED))
    polish(work);
#endif /* ifndef EMBEDDED */

#ifdef PROFILING
  /* Update total time */
  if (work->first_run) {
    // total time: setup + solve + polish
    work->info->run_time = work->info->setup_time +
                           work->info->solve_time +
                           work->info->polish_time;
  } else {
    // total time: update + solve + polish
    work->info->run_time = work->info->update_time +
                           work->info->solve_time +
                           work->info->polish_time;
  }

  // Indicate that the solve function has already been executed
  if (work->first_run) work->first_run = 0;

  // Indicate that the update_time should be set to zero
  work->clear_update_time = 1;

  // Indicate that osqp_update_rho is not called from osqp_solve
  work->rho_update_from_solve = 0;
#endif /* ifdef PROFILING */

#ifdef PRINTING
  /* Print final footer */
  if (work->settings->verbose) print_footer(work->info, work->settings->polish);
#endif /* ifdef PRINTING */

  // Store solution
  store_solution(work);


// Define exit flag for quitting function
#if defined(PROFILING) || defined(CTRLC) || EMBEDDED != 1
exit:
#endif /* if defined(PROFILING) || defined(CTRLC) || EMBEDDED != 1 */

#ifdef CTRLC
  // Restore previous signal handler
  osqp_end_interrupt_listener();
#endif /* ifdef CTRLC */

  return exitflag;
}


#ifndef EMBEDDED

c_int osqp_cleanup(OSQPWorkspace *work) {
  c_int exitflag = 0;

  if (work) { // If workspace has been allocated
    // Free Data
    if (work->data) {
      if (work->data->P) csc_spfree(work->data->P);
      if (work->data->A) csc_spfree(work->data->A);
      if (work->data->q) c_free(work->data->q);
      if (work->data->l) c_free(work->data->l);
      if (work->data->u) c_free(work->data->u);
      c_free(work->data);
    }

    // Free scaling variables
    if (work->scaling){
      if (work->scaling->D)    c_free(work->scaling->D);
      if (work->scaling->Dinv) c_free(work->scaling->Dinv);
      if (work->scaling->E)    c_free(work->scaling->E);
      if (work->scaling->Einv) c_free(work->scaling->Einv);
      c_free(work->scaling);
    }

    // Free temp workspace variables for scaling
    if (work->D_temp)   c_free(work->D_temp);
    if (work->D_temp_A) c_free(work->D_temp_A);
    if (work->E_temp)   c_free(work->E_temp);

    // Free linear system solver structure
    if (work->linsys_solver) {
      if (work->linsys_solver->free) {
        work->linsys_solver->free(work->linsys_solver);
      }
    }

    // Unload linear system solver after free
    if (work->settings) {
      exitflag = unload_linsys_solver(work->settings->linsys_solver);
    }

#ifndef EMBEDDED
    // Free active constraints structure
    if (work->pol) {
      if (work->pol->Alow_to_A) c_free(work->pol->Alow_to_A);
      if (work->pol->Aupp_to_A) c_free(work->pol->Aupp_to_A);
      if (work->pol->A_to_Alow) c_free(work->pol->A_to_Alow);
      if (work->pol->A_to_Aupp) c_free(work->pol->A_to_Aupp);
      if (work->pol->x)         c_free(work->pol->x);
      if (work->pol->z)         c_free(work->pol->z);
      if (work->pol->y)         c_free(work->pol->y);
      c_free(work->pol);
    }
#endif /* ifndef EMBEDDED */

    // Free other Variables
    if (work->rho_vec)     c_free(work->rho_vec);
    if (work->rho_inv_vec) c_free(work->rho_inv_vec);
#if EMBEDDED != 1
    if (work->constr_type) c_free(work->constr_type);
#endif
    if (work->x)           c_free(work->x);
    if (work->z)           c_free(work->z);
    if (work->xz_tilde)    c_free(work->xz_tilde);
    if (work->x_prev)      c_free(work->x_prev);
    if (work->z_prev)      c_free(work->z_prev);
    if (work->y)           c_free(work->y);
    if (work->Ax)          c_free(work->Ax);
    if (work->Px)          c_free(work->Px);
    if (work->Aty)         c_free(work->Aty);
    if (work->delta_y)     c_free(work->delta_y);
    if (work->Atdelta_y)   c_free(work->Atdelta_y);
    if (work->delta_x)     c_free(work->delta_x);
    if (work->Pdelta_x)    c_free(work->Pdelta_x);
    if (work->Adelta_x)    c_free(work->Adelta_x);

    // Free Settings
    if (work->settings) c_free(work->settings);

    // Free solution
    if (work->solution) {
      if (work->solution->x) c_free(work->solution->x);
      if (work->solution->y) c_free(work->solution->y);
      c_free(work->solution);
    }

    // Free information
    if (work->info) c_free(work->info);

# ifdef PROFILING
    // Free timer
    if (work->timer) c_free(work->timer);
# endif /* ifdef PROFILING */

    // Free work
    c_free(work);
  }

  return exitflag;
}

#endif // #ifndef EMBEDDED


/************************
* Update problem data  *
************************/
c_int osqp_update_lin_cost(OSQPWorkspace *work, const c_float *q_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    work->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  // Replace q by the new vector
  prea_vec_copy(q_new, work->data->q, work->data->n);

  // Scaling
  if (work->settings->scaling) {
    vec_ew_prod(work->scaling->D, work->data->q, work->data->q, work->data->n);
    vec_mult_scalar(work->data->q, work->scaling->c, work->data->n);
  }

  // Reset solver information
  reset_info(work->info);

#ifdef PROFILING
  work->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return 0;
}

c_int osqp_update_bounds(OSQPWorkspace *work,
                         const c_float *l_new,
                         const c_float *u_new) {
  c_int i, exitflag = 0;

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    work->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  // Check if lower bound is smaller than upper bound
  for (i = 0; i < work->data->m; i++) {
    if (l_new[i] > u_new[i]) {
#ifdef PRINTING
      c_eprint("lower bound must be lower than or equal to upper bound");
#endif /* ifdef PRINTING */
      return 1;
    }
  }

  // Replace l and u by the new vectors
  prea_vec_copy(l_new, work->data->l, work->data->m);
  prea_vec_copy(u_new, work->data->u, work->data->m);

  // Scaling
  if (work->settings->scaling) {
    vec_ew_prod(work->scaling->E, work->data->l, work->data->l, work->data->m);
    vec_ew_prod(work->scaling->E, work->data->u, work->data->u, work->data->m);
  }

  // Reset solver information
  reset_info(work->info);

#if EMBEDDED != 1
  // Update rho_vec and refactor if constraints type changes
  exitflag = update_rho_vec(work);
#endif // EMBEDDED != 1

#ifdef PROFILING
  work->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}

c_int osqp_update_lower_bound(OSQPWorkspace *work, const c_float *l_new) {
  c_int i, exitflag = 0;

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    work->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  // Replace l by the new vector
  prea_vec_copy(l_new, work->data->l, work->data->m);

  // Scaling
  if (work->settings->scaling) {
    vec_ew_prod(work->scaling->E, work->data->l, work->data->l, work->data->m);
  }

  // Check if lower bound is smaller than upper bound
  for (i = 0; i < work->data->m; i++) {
    if (work->data->l[i] > work->data->u[i]) {
#ifdef PRINTING
      c_eprint("upper bound must be greater than or equal to lower bound");
#endif /* ifdef PRINTING */
      return 1;
    }
  }

  // Reset solver information
  reset_info(work->info);

#if EMBEDDED != 1
  // Update rho_vec and refactor if constraints type changes
  exitflag = update_rho_vec(work);
#endif // EMBEDDED ! =1

#ifdef PROFILING
  work->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}

c_int osqp_update_upper_bound(OSQPWorkspace *work, const c_float *u_new) {
  c_int i, exitflag = 0;

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    work->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  // Replace u by the new vector
  prea_vec_copy(u_new, work->data->u, work->data->m);

  // Scaling
  if (work->settings->scaling) {
    vec_ew_prod(work->scaling->E, work->data->u, work->data->u, work->data->m);
  }

  // Check if upper bound is greater than lower bound
  for (i = 0; i < work->data->m; i++) {
    if (work->data->u[i] < work->data->l[i]) {
#ifdef PRINTING
      c_eprint("lower bound must be lower than or equal to upper bound");
#endif /* ifdef PRINTING */
      return 1;
    }
  }

  // Reset solver information
  reset_info(work->info);

#if EMBEDDED != 1
  // Update rho_vec and refactor if constraints type changes
  exitflag = update_rho_vec(work);
#endif // EMBEDDED != 1

#ifdef PROFILING
  work->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}

c_int osqp_warm_start(OSQPWorkspace *work, const c_float *x, const c_float *y) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Update warm_start setting to true
  if (!work->settings->warm_start) work->settings->warm_start = 1;

  // Copy primal and dual variables into the iterates
  prea_vec_copy(x, work->x, work->data->n);
  prea_vec_copy(y, work->y, work->data->m);

  // Scale iterates
  if (work->settings->scaling) {
    vec_ew_prod(work->scaling->Dinv, work->x, work->x, work->data->n);
    vec_ew_prod(work->scaling->Einv, work->y, work->y, work->data->m);
    vec_mult_scalar(work->y, work->scaling->c, work->data->m);
  }

  // Compute Ax = z and store it in z
  mat_vec(work->data->A, work->x, work->z, 0);

  return 0;
}

c_int osqp_warm_start_x(OSQPWorkspace *work, const c_float *x) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Update warm_start setting to true
  if (!work->settings->warm_start) work->settings->warm_start = 1;

  // Copy primal variable into the iterate x
  prea_vec_copy(x, work->x, work->data->n);

  // Scale iterate
  if (work->settings->scaling) {
    vec_ew_prod(work->scaling->Dinv, work->x, work->x, work->data->n);
  }

  // Compute Ax = z and store it in z
  mat_vec(work->data->A, work->x, work->z, 0);

  return 0;
}

c_int osqp_warm_start_y(OSQPWorkspace *work, const c_float *y) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Update warm_start setting to true
  if (!work->settings->warm_start) work->settings->warm_start = 1;

  // Copy primal variable into the iterate y
  prea_vec_copy(y, work->y, work->data->m);

  // Scale iterate
  if (work->settings->scaling) {
    vec_ew_prod(work->scaling->Einv, work->y, work->y, work->data->m);
    vec_mult_scalar(work->y, work->scaling->c, work->data->m);
  }

  return 0;
}


#if EMBEDDED != 1

c_int osqp_update_P(OSQPWorkspace *work,
                    const c_float *Px_new,
                    const c_int   *Px_new_idx,
                    c_int          P_new_n) {
  c_int i;        // For indexing
  c_int exitflag; // Exit flag
  c_int nnzP;     // Number of nonzeros in P

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    work->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  nnzP = work->data->P->p[work->data->P->n];

  if (Px_new_idx) { // Passing the index of elements changed
    // Check if number of elements is less or equal than the total number of
    // nonzeros in P
    if (P_new_n > nnzP) {
# ifdef PRINTING
      c_eprint("new number of elements (%i) greater than elements in P (%i)",
               (int)P_new_n,
               (int)nnzP);
# endif /* ifdef PRINTING */
      return 1;
    }
  }

  if (work->settings->scaling) {
    // Unscale data
    unscale_data(work);
  }

  // Update P elements
  if (Px_new_idx) { // Change only Px_new_idx
    for (i = 0; i < P_new_n; i++) {
      work->data->P->x[Px_new_idx[i]] = Px_new[i];
    }
  }
  else // Change whole P
  {
    for (i = 0; i < nnzP; i++) {
      work->data->P->x[i] = Px_new[i];
    }
  }

  if (work->settings->scaling) {
    // Scale data
    scale_data(work);
  }

  // Update linear system structure with new data
  exitflag = work->linsys_solver->update_matrices(work->linsys_solver,
                                                  work->data->P,
                                                  work->data->A);

  // Reset solver information
  reset_info(work->info);

# ifdef PRINTING

  if (exitflag < 0) {
    c_eprint("new KKT matrix is not quasidefinite");
  }
# endif /* ifdef PRINTING */

#ifdef PROFILING
  work->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}


c_int osqp_update_A(OSQPWorkspace *work,
                    const c_float *Ax_new,
                    const c_int   *Ax_new_idx,
                    c_int          A_new_n) {
  c_int i;        // For indexing
  c_int exitflag; // Exit flag
  c_int nnzA;     // Number of nonzeros in A

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    work->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  nnzA = work->data->A->p[work->data->A->n];

  if (Ax_new_idx) { // Passing the index of elements changed
    // Check if number of elements is less or equal than the total number of
    // nonzeros in A
    if (A_new_n > nnzA) {
# ifdef PRINTING
      c_eprint("new number of elements (%i) greater than elements in A (%i)",
               (int)A_new_n,
               (int)nnzA);
# endif /* ifdef PRINTING */
      return 1;
    }
  }

  if (work->settings->scaling) {
    // Unscale data
    unscale_data(work);
  }

  // Update A elements
  if (Ax_new_idx) { // Change only Ax_new_idx
    for (i = 0; i < A_new_n; i++) {
      work->data->A->x[Ax_new_idx[i]] = Ax_new[i];
    }
  }
  else { // Change whole A
    for (i = 0; i < nnzA; i++) {
      work->data->A->x[i] = Ax_new[i];
    }
  }

  if (work->settings->scaling) {
    // Scale data
    scale_data(work);
  }

  // Update linear system structure with new data
  exitflag = work->linsys_solver->update_matrices(work->linsys_solver,
                                                  work->data->P,
                                                  work->data->A);

  // Reset solver information
  reset_info(work->info);

# ifdef PRINTING

  if (exitflag < 0) {
    c_eprint("new KKT matrix is not quasidefinite");
  }
# endif /* ifdef PRINTING */

#ifdef PROFILING
  work->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}


c_int osqp_update_P_A(OSQPWorkspace *work,
                      const c_float *Px_new,
                      const c_int   *Px_new_idx,
                      c_int          P_new_n,
                      const c_float *Ax_new,
                      const c_int   *Ax_new_idx,
                      c_int          A_new_n) {
  c_int i;          // For indexing
  c_int exitflag;   // Exit flag
  c_int nnzP, nnzA; // Number of nonzeros in P and A

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    work->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  nnzP = work->data->P->p[work->data->P->n];
  nnzA = work->data->A->p[work->data->A->n];


  if (Px_new_idx) { // Passing the index of elements changed
    // Check if number of elements is less or equal than the total number of
    // nonzeros in P
    if (P_new_n > nnzP) {
# ifdef PRINTING
      c_eprint("new number of elements (%i) greater than elements in P (%i)",
               (int)P_new_n,
               (int)nnzP);
# endif /* ifdef PRINTING */
      return 1;
    }
  }


  if (Ax_new_idx) { // Passing the index of elements changed
    // Check if number of elements is less or equal than the total number of
    // nonzeros in A
    if (A_new_n > nnzA) {
# ifdef PRINTING
      c_eprint("new number of elements (%i) greater than elements in A (%i)",
               (int)A_new_n,
               (int)nnzA);
# endif /* ifdef PRINTING */
      return 2;
    }
  }

  if (work->settings->scaling) {
    // Unscale data
    unscale_data(work);
  }

  // Update P elements
  if (Px_new_idx) { // Change only Px_new_idx
    for (i = 0; i < P_new_n; i++) {
      work->data->P->x[Px_new_idx[i]] = Px_new[i];
    }
  }
  else // Change whole P
  {
    for (i = 0; i < nnzP; i++) {
      work->data->P->x[i] = Px_new[i];
    }
  }

  // Update A elements
  if (Ax_new_idx) { // Change only Ax_new_idx
    for (i = 0; i < A_new_n; i++) {
      work->data->A->x[Ax_new_idx[i]] = Ax_new[i];
    }
  }
  else { // Change whole A
    for (i = 0; i < nnzA; i++) {
      work->data->A->x[i] = Ax_new[i];
    }
  }

  if (work->settings->scaling) {
    // Scale data
    scale_data(work);
  }

  // Update linear system structure with new data
  exitflag = work->linsys_solver->update_matrices(work->linsys_solver,
                                                  work->data->P,
                                                  work->data->A);

  // Reset solver information
  reset_info(work->info);

# ifdef PRINTING

  if (exitflag < 0) {
    c_eprint("new KKT matrix is not quasidefinite");
  }
# endif /* ifdef PRINTING */

#ifdef PROFILING
  work->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}

c_int osqp_update_rho(OSQPWorkspace *work, c_float rho_new) {
  c_int exitflag, i;

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check value of rho
  if (rho_new <= 0) {
# ifdef PRINTING
    c_eprint("rho must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

#ifdef PROFILING
  if (work->rho_update_from_solve == 0) {
    if (work->clear_update_time == 1) {
      work->clear_update_time = 0;
      work->info->update_time = 0.0;
    }
    osqp_tic(work->timer); // Start timer
  }
#endif /* ifdef PROFILING */

  // Update rho in settings
  work->settings->rho = c_min(c_max(rho_new, RHO_MIN), RHO_MAX);

  // Update rho_vec and rho_inv_vec
  for (i = 0; i < work->data->m; i++) {
    if (work->constr_type[i] == 0) {
      // Inequalities
      work->rho_vec[i]     = work->settings->rho;
      work->rho_inv_vec[i] = 1. / work->settings->rho;
    }
    else if (work->constr_type[i] == 1) {
      // Equalities
      work->rho_vec[i]     = RHO_EQ_OVER_RHO_INEQ * work->settings->rho;
      work->rho_inv_vec[i] = 1. / work->rho_vec[i];
    }
  }

  // Update rho_vec in KKT matrix
  exitflag = work->linsys_solver->update_rho_vec(work->linsys_solver,
                                                 work->rho_vec);

#ifdef PROFILING
  if (work->rho_update_from_solve == 0)
    work->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}

#endif // EMBEDDED != 1

/****************************
* Update problem settings  *
****************************/
c_int osqp_update_max_iter(OSQPWorkspace *work, c_int max_iter_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that max_iter is positive
  if (max_iter_new <= 0) {
#ifdef PRINTING
    c_eprint("max_iter must be positive");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update max_iter
  work->settings->max_iter = max_iter_new;

  return 0;
}

c_int osqp_update_eps_abs(OSQPWorkspace *work, c_float eps_abs_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that eps_abs is positive
  if (eps_abs_new < 0.) {
#ifdef PRINTING
    c_eprint("eps_abs must be nonnegative");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update eps_abs
  work->settings->eps_abs = eps_abs_new;

  return 0;
}

c_int osqp_update_eps_rel(OSQPWorkspace *work, c_float eps_rel_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that eps_rel is positive
  if (eps_rel_new < 0.) {
#ifdef PRINTING
    c_eprint("eps_rel must be nonnegative");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update eps_rel
  work->settings->eps_rel = eps_rel_new;

  return 0;
}

c_int osqp_update_eps_prim_inf(OSQPWorkspace *work, c_float eps_prim_inf_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that eps_prim_inf is positive
  if (eps_prim_inf_new < 0.) {
#ifdef PRINTING
    c_eprint("eps_prim_inf must be nonnegative");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update eps_prim_inf
  work->settings->eps_prim_inf = eps_prim_inf_new;

  return 0;
}

c_int osqp_update_eps_dual_inf(OSQPWorkspace *work, c_float eps_dual_inf_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that eps_dual_inf is positive
  if (eps_dual_inf_new < 0.) {
#ifdef PRINTING
    c_eprint("eps_dual_inf must be nonnegative");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update eps_dual_inf
  work->settings->eps_dual_inf = eps_dual_inf_new;


  return 0;
}

c_int osqp_update_alpha(OSQPWorkspace *work, c_float alpha_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that alpha is between 0 and 2
  if ((alpha_new <= 0.) || (alpha_new >= 2.)) {
#ifdef PRINTING
    c_eprint("alpha must be between 0 and 2");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update alpha
  work->settings->alpha = alpha_new;

  return 0;
}

c_int osqp_update_warm_start(OSQPWorkspace *work, c_int warm_start_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that warm_start is either 0 or 1
  if ((warm_start_new != 0) && (warm_start_new != 1)) {
#ifdef PRINTING
    c_eprint("warm_start should be either 0 or 1");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update warm_start
  work->settings->warm_start = warm_start_new;

  return 0;
}

c_int osqp_update_scaled_termination(OSQPWorkspace *work, c_int scaled_termination_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that scaled_termination is either 0 or 1
  if ((scaled_termination_new != 0) && (scaled_termination_new != 1)) {
#ifdef PRINTING
    c_eprint("scaled_termination should be either 0 or 1");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update scaled_termination
  work->settings->scaled_termination = scaled_termination_new;

  return 0;
}

c_int osqp_update_check_termination(OSQPWorkspace *work, c_int check_termination_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that check_termination is nonnegative
  if (check_termination_new < 0) {
#ifdef PRINTING
    c_eprint("check_termination should be nonnegative");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update check_termination
  work->settings->check_termination = check_termination_new;

  return 0;
}

#ifndef EMBEDDED

c_int osqp_update_delta(OSQPWorkspace *work, c_float delta_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that delta is positive
  if (delta_new <= 0.) {
# ifdef PRINTING
    c_eprint("delta must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

  // Update delta
  work->settings->delta = delta_new;

  return 0;
}

c_int osqp_update_polish(OSQPWorkspace *work, c_int polish_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that polish is either 0 or 1
  if ((polish_new != 0) && (polish_new != 1)) {
# ifdef PRINTING
    c_eprint("polish should be either 0 or 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  // Update polish
  work->settings->polish = polish_new;

# ifdef PROFILING

  // Reset polish time to zero
  work->info->polish_time = 0.0;
# endif /* ifdef PROFILING */

  return 0;
}

c_int osqp_update_polish_refine_iter(OSQPWorkspace *work, c_int polish_refine_iter_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that polish_refine_iter is nonnegative
  if (polish_refine_iter_new < 0) {
# ifdef PRINTING
    c_eprint("polish_refine_iter must be nonnegative");
# endif /* ifdef PRINTING */
    return 1;
  }

  // Update polish_refine_iter
  work->settings->polish_refine_iter = polish_refine_iter_new;

  return 0;
}

c_int osqp_update_verbose(OSQPWorkspace *work, c_int verbose_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that verbose is either 0 or 1
  if ((verbose_new != 0) && (verbose_new != 1)) {
# ifdef PRINTING
    c_eprint("verbose should be either 0 or 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  // Update verbose
  work->settings->verbose = verbose_new;

  return 0;
}

#endif // EMBEDDED

#ifdef PROFILING

c_int osqp_update_time_limit(OSQPWorkspace *work, c_float time_limit_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that time_limit is nonnegative
  if (time_limit_new < 0.) {
# ifdef PRINTING
    c_print("time_limit must be nonnegative\n");
# endif /* ifdef PRINTING */
    return 1;
  }

  // Update time_limit
  work->settings->time_limit = time_limit_new;

  return 0;
}
#endif /* ifdef PROFILING */



void project(OSQPWorkspace *work, c_float *z) {
  c_int i, m;

  m = work->data->m;

  for (i = 0; i < m; i++) {
    z[i] = c_min(c_max(z[i],
                       work->data->l[i]), // Between lower
                 work->data->u[i]);       // and upper bounds
  }
}

void project_normalcone(OSQPWorkspace *work, c_float *z, c_float *y) {
  c_int i, m;

  // NB: Use z_prev as temporary vector

  m = work->data->m;

  for (i = 0; i < m; i++) {
    work->z_prev[i] = z[i] + y[i];
    z[i]            = c_min(c_max(work->z_prev[i], work->data->l[i]),
                            work->data->u[i]);
    y[i] = work->z_prev[i] - z[i];
  }
}


#define QDLDL_UNKNOWN (-1)
#define QDLDL_USED (1)
#define QDLDL_UNUSED (0)

// //DEBUG
// #include <stdio.h>
// void qdprint_arrayi(const QDLDL_int* data, QDLDL_int n,char* varName){

//   QDLDL_int i;
//   printf("%s = [",varName);
//   for(i=0; i< n; i++){
//     printf("%lli,",data[i]);
//   }
//   printf("]\n");

// }

// void qdprint_arrayf(const QDLDL_float* data, QDLDL_int n, char* varName){

//   QDLDL_int i;
//   printf("%s = [",varName);
//   for(i=0; i< n; i++){
//     printf("%.3g,",data[i]);
//   }
//   printf("]\n");

// }
// // END DEBUG

/* Compute the elimination tree for a quasidefinite matrix
   in compressed sparse column form.
*/

QDLDL_int QDLDL_etree(const QDLDL_int  n,
                      const QDLDL_int* Ap,
                      const QDLDL_int* Ai,
                      QDLDL_int* work,
                      QDLDL_int* Lnz,
                      QDLDL_int* etree){

  QDLDL_int sumLnz = 0;
  QDLDL_int i,j,p;


  for(i = 0; i < n; i++){
  // zero out Lnz and work.  Set all etree values to unknown
    work[i]  = 0;
    Lnz[i]   = 0;
    etree[i] = QDLDL_UNKNOWN;

    //Abort if A doesn't have at least one entry
    //one entry in every column
    if(Ap[i] == Ap[i+1]){
      return -1;
    }

  }

  for(j = 0; j < n; j++){
    work[j] = j;
    for(p = Ap[j]; p < Ap[j+1]; p++){
      i = Ai[p];
      if(i > j){return -1;}; //abort if entries on lower triangle
      while(work[i] != j){
        if(etree[i] == QDLDL_UNKNOWN){
          etree[i] = j;
        }
        Lnz[i]++;         //nonzeros in this column
        work[i] = j;
        i = etree[i];
      }
    }
  }

  //compute the total nonzeros in L.  This much
  //space is required to store Li and Lx
  for(i = 0; i < n; i++){sumLnz += Lnz[i];}

  return sumLnz;
}



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
                  QDLDL_bool*  bwork,
                  QDLDL_int*   iwork,
                  QDLDL_float* fwork){

  QDLDL_int i,j,k,nnzY, bidx, cidx, nextIdx, nnzE, tmpIdx;
  QDLDL_int *yIdx, *elimBuffer, *LNextSpaceInCol;
  QDLDL_float *yVals;
  QDLDL_float yVals_cidx;
  QDLDL_bool  *yMarkers;
  QDLDL_int   positiveValuesInD = 0;

  //partition working memory into pieces
  yMarkers        = bwork;
  yIdx            = iwork;
  elimBuffer      = iwork + n;
  LNextSpaceInCol = iwork + n*2;
  yVals           = fwork;


  Lp[0] = 0; //first column starts at index zero

  for(i = 0; i < n; i++){

    //compute L column indices
    Lp[i+1] = Lp[i] + Lnz[i];   //cumsum, total at the end

    // set all Yidx to be 'unused' initially
    //in each column of L, the next available space
    //to start is just the first space in the column
    yMarkers[i]  = QDLDL_UNUSED;
    yVals[i]     = 0.0;
    D[i]         = 0.0;
    LNextSpaceInCol[i] = Lp[i];
  }

  // First element of the diagonal D.
  D[0]     = Ax[0];
  if(D[0] == 0.0){return -1;}
  if(D[0]  > 0.0){positiveValuesInD++;}
  Dinv[0] = 1/D[0];

  //Start from 1 here. The upper LH corner is trivially 0
  //in L b/c we are only computing the subdiagonal elements
  for(k = 1; k < n; k++){

    //NB : For each k, we compute a solution to
    //y = L(0:(k-1),0:k-1))\b, where b is the kth
    //column of A that sits above the diagonal.
    //The solution y is then the kth row of L,
    //with an implied '1' at the diagonal entry.

    //number of nonzeros in this row of L
    nnzY = 0;  //number of elements in this row

    //This loop determines where nonzeros
    //will go in the kth row of L, but doesn't
    //compute the actual values
    tmpIdx = Ap[k+1];

    for(i = Ap[k]; i < tmpIdx; i++){

      bidx = Ai[i];   // we are working on this element of b

      //Initialize D[k] as the element of this column
      //corresponding to the diagonal place.  Don't use
      //this element as part of the elimination step
      //that computes the k^th row of L
      if(bidx == k){
        D[k] = Ax[i];
        continue;
      }

      yVals[bidx] = Ax[i];   // initialise y(bidx) = b(bidx)

      // use the forward elimination tree to figure
      // out which elements must be eliminated after
      // this element of b
      nextIdx = bidx;

      if(yMarkers[nextIdx] == QDLDL_UNUSED){   //this y term not already visited

        yMarkers[nextIdx] = QDLDL_USED;     //I touched this one
        elimBuffer[0]     = nextIdx;  // It goes at the start of the current list
        nnzE              = 1;         //length of unvisited elimination path from here

        nextIdx = etree[bidx];

        while(nextIdx != QDLDL_UNKNOWN && nextIdx < k){
          if(yMarkers[nextIdx] == QDLDL_USED) break;

          yMarkers[nextIdx] = QDLDL_USED;   //I touched this one
          elimBuffer[nnzE] = nextIdx; //It goes in the current list
          nnzE++;                     //the list is one longer than before
          nextIdx = etree[nextIdx];   //one step further along tree

        } //end while

        // now I put the buffered elimination list into
        // my current ordering in reverse order
        while(nnzE){
          yIdx[nnzY++] = elimBuffer[--nnzE];
        } //end while
      } //end if

    } //end for i

    //This for loop places nonzeros values in the k^th row
    for(i = (nnzY-1); i >=0; i--){

      //which column are we working on?
      cidx = yIdx[i];

      // loop along the elements in this
      // column of L and subtract to solve to y
      tmpIdx = LNextSpaceInCol[cidx];
      yVals_cidx = yVals[cidx];
      for(j = Lp[cidx]; j < tmpIdx; j++){
        yVals[Li[j]] -= Lx[j]*yVals_cidx;
      }

      //Now I have the cidx^th element of y = L\b.
      //so compute the corresponding element of
      //this row of L and put it into the right place
      Li[tmpIdx] = k;
      Lx[tmpIdx] = yVals_cidx *Dinv[cidx];

      //D[k] -= yVals[cidx]*yVals[cidx]*Dinv[cidx];
      D[k] -= yVals_cidx*Lx[tmpIdx];
      LNextSpaceInCol[cidx]++;

      //reset the yvalues and indices back to zero and QDLDL_UNUSED
      //once I'm done with them
      yVals[cidx]     = 0.0;
      yMarkers[cidx]  = QDLDL_UNUSED;

    } //end for i

    //Maintain a count of the positive entries
    //in D.  If we hit a zero, we can't factor
    //this matrix, so abort
    if(D[k] == 0.0){return -1;}
    if(D[k]  > 0.0){positiveValuesInD++;}

    //compute the inverse of the diagonal
    Dinv[k]= 1/D[k];

  } //end for k

  return positiveValuesInD;

}

// Solves (L+I)x = b
void QDLDL_Lsolve(const QDLDL_int    n,
                  const QDLDL_int*   Lp,
                  const QDLDL_int*   Li,
                  const QDLDL_float* Lx,
                  QDLDL_float* x){

QDLDL_int i,j;
  for(i = 0; i < n; i++){
      for(j = Lp[i]; j < Lp[i+1]; j++){
          x[Li[j]] -= Lx[j]*x[i];
      }
  }
}

// Solves (L+I)'x = b
void QDLDL_Ltsolve(const QDLDL_int    n,
                   const QDLDL_int*   Lp,
                   const QDLDL_int*   Li,
                   const QDLDL_float* Lx,
                   QDLDL_float* x){

QDLDL_int i,j;
  for(i = n-1; i>=0; i--){
      for(j = Lp[i]; j < Lp[i+1]; j++){
          x[i] -= Lx[j]*x[Li[j]];
      }
  }
}

// Solves Ax = b where A has given LDL factors
void QDLDL_solve(const QDLDL_int       n,
                    const QDLDL_int*   Lp,
                    const QDLDL_int*   Li,
                    const QDLDL_float* Lx,
                    const QDLDL_float* Dinv,
                    QDLDL_float* x){

QDLDL_int i;

QDLDL_Lsolve(n,Lp,Li,Lx,x);
for(i = 0; i < n; i++) x[i] *= Dinv[i];
QDLDL_Ltsolve(n,Lp,Li,Lx,x);

}

#ifndef EMBEDDED
#include "amd.h"
#endif


#ifndef EMBEDDED

// Free LDL Factorization structure
void free_linsys_solver_qdldl(qdldl_solver *s) {
    if (s) {
        if (s->L)           csc_spfree(s->L);
        if (s->P)           c_free(s->P);
        if (s->Dinv)        c_free(s->Dinv);
        if (s->bp)          c_free(s->bp);
        if (s->sol)         c_free(s->sol);
        if (s->rho_inv_vec) c_free(s->rho_inv_vec);

        // These are required for matrix updates
        if (s->Pdiag_idx) c_free(s->Pdiag_idx);
        if (s->KKT)       csc_spfree(s->KKT);
        if (s->PtoKKT)    c_free(s->PtoKKT);
        if (s->AtoKKT)    c_free(s->AtoKKT);
        if (s->rhotoKKT)  c_free(s->rhotoKKT);

        // QDLDL workspace
        if (s->D)         c_free(s->D);
        if (s->etree)     c_free(s->etree);
        if (s->Lnz)       c_free(s->Lnz);
        if (s->iwork)     c_free(s->iwork);
        if (s->bwork)     c_free(s->bwork);
        if (s->fwork)     c_free(s->fwork);
        c_free(s);

    }
}


/**
 * Compute LDL factorization of matrix A
 * @param  A    Matrix to be factorized
 * @param  p    Private workspace
 * @param  nvar Number of QP variables
 * @return      exitstatus (0 is good)
 */
static c_int LDL_factor(csc *A,  qdldl_solver * p, c_int nvar){

    c_int sum_Lnz;
    c_int factor_status;

    // Compute elimination tree
    sum_Lnz = QDLDL_etree(A->n, A->p, A->i, p->iwork, p->Lnz, p->etree);

    if (sum_Lnz < 0){
      // Error
#ifdef PRINTING
      c_eprint("Error in KKT matrix LDL factorization when computing the elimination tree. A is not perfectly upper triangular");
#endif
      return sum_Lnz;
    }

    // Allocate memory for Li and Lx
    p->L->i = (c_int *)c_malloc(sizeof(c_int)*sum_Lnz);
    p->L->x = (c_float *)c_malloc(sizeof(c_float)*sum_Lnz);

    // Factor matrix
    factor_status = QDLDL_factor(A->n, A->p, A->i, A->x,
                                 p->L->p, p->L->i, p->L->x,
                                 p->D, p->Dinv, p->Lnz,
                                 p->etree, p->bwork, p->iwork, p->fwork);


    if (factor_status < 0){
      // Error
#ifdef PRINTING
      c_eprint("Error in KKT matrix LDL factorization when computing the nonzero elements. There are zeros in the diagonal matrix");
#endif
      return factor_status;
    } else if (factor_status < nvar) {
      // Error: Number of positive elements of D should be equal to nvar
#ifdef PRINTING
      c_eprint("Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex");
#endif
      return -2;
    }

    return 0;

}


static c_int permute_KKT(csc ** KKT, qdldl_solver * p, c_int Pnz, c_int Anz, c_int m, c_int * PtoKKT, c_int * AtoKKT, c_int * rhotoKKT){
    c_float *info;
    c_int amd_status;
    c_int * Pinv;
    csc *KKT_temp;
    c_int * KtoPKPt;
    c_int i; // Indexing

    info = (c_float *)c_malloc(AMD_INFO * sizeof(c_float));

    // Compute permutation matrix P using AMD
#ifdef DLONG
    amd_status = amd_l_order((*KKT)->n, (*KKT)->p, (*KKT)->i, p->P, (c_float *)OSQP_NULL, info);
#else
    amd_status = amd_order((*KKT)->n, (*KKT)->p, (*KKT)->i, p->P, (c_float *)OSQP_NULL, info);
#endif
    if (amd_status < 0) {
        // Free Amd info and return an error
        c_free(info);
        return amd_status;
    }


    // Inverse of the permutation vector
    Pinv = csc_pinv(p->P, (*KKT)->n);

    // Permute KKT matrix
    if (!PtoKKT && !AtoKKT && !rhotoKKT){  // No vectors to be stored
        // Assign values of mapping
        KKT_temp = csc_symperm((*KKT), Pinv, OSQP_NULL, 1);
    }
    else {
        // Allocate vector of mappings from unpermuted to permuted
        KtoPKPt = c_malloc((*KKT)->p[(*KKT)->n] * sizeof(c_int));
        KKT_temp = csc_symperm((*KKT), Pinv, KtoPKPt, 1);

        // Update vectors PtoKKT, AtoKKT and rhotoKKT
        if (PtoKKT){
            for (i = 0; i < Pnz; i++){
                PtoKKT[i] = KtoPKPt[PtoKKT[i]];
            }
        }
        if (AtoKKT){
            for (i = 0; i < Anz; i++){
                AtoKKT[i] = KtoPKPt[AtoKKT[i]];
            }
        }
        if (rhotoKKT){
            for (i = 0; i < m; i++){
                rhotoKKT[i] = KtoPKPt[rhotoKKT[i]];
            }
        }

        // Cleanup vector of mapping
        c_free(KtoPKPt);
    }

    // Cleanup
    // Free previous KKT matrix and assign pointer to new one
    csc_spfree((*KKT));
    (*KKT) = KKT_temp;
    // Free Pinv
    c_free(Pinv);
    // Free Amd info
    c_free(info);

    return 0;
}


// Initialize LDL Factorization structure
c_int init_linsys_solver_qdldl(qdldl_solver ** sp, const csc * P, const csc * A, c_float sigma, const c_float * rho_vec, c_int polish){

    // Define Variables
    csc * KKT_temp;     // Temporary KKT pointer
    c_int i;            // Loop counter
    c_int n_plus_m;     // Define n_plus_m dimension

    // Allocate private structure to store KKT factorization
    qdldl_solver *s;
    s = c_calloc(1, sizeof(qdldl_solver));
    *sp = s;

    // Size of KKT
    s->n = P->n;
    s->m = A->m;
    n_plus_m = s->n + s->m;

    // Sigma parameter
    s->sigma = sigma;

    // Polishing flag
    s->polish = polish;

    // Link Functions
    s->solve = &solve_linsys_qdldl;

#ifndef EMBEDDED
    s->free = &free_linsys_solver_qdldl;
#endif

#if EMBEDDED != 1
    s->update_matrices = &update_linsys_solver_matrices_qdldl;
    s->update_rho_vec = &update_linsys_solver_rho_vec_qdldl;
#endif

    // Assign type
    s->type = QDLDL_SOLVER;

    // Set number of threads to 1 (single threaded)
    s->nthreads = 1;

    // Sparse matrix L (lower triangular)
    // NB: We don not allocate L completely (CSC elements)
    //      L will be allocated during the factorization depending on the
    //      resulting number of elements.
    s->L = c_malloc(sizeof(csc));
    s->L->m = n_plus_m;
    s->L->n = n_plus_m;
    s->L->nz = -1;

    // Diagonal matrix stored as a vector D
    s->Dinv = (QDLDL_float *)c_malloc(sizeof(QDLDL_float) * n_plus_m);
    s->D    = (QDLDL_float *)c_malloc(sizeof(QDLDL_float) * n_plus_m);

    // Permutation vector P
    s->P    = (QDLDL_int *)c_malloc(sizeof(QDLDL_int) * n_plus_m);

    // Working vector
    s->bp   = (QDLDL_float *)c_malloc(sizeof(QDLDL_float) * n_plus_m);

    // Solution vector
    s->sol  = (QDLDL_float *)c_malloc(sizeof(QDLDL_float) * n_plus_m);

    // Parameter vector
    s->rho_inv_vec = (c_float *)c_malloc(sizeof(c_float) * s->m);

    // Elimination tree workspace
    s->etree = (QDLDL_int *)c_malloc(n_plus_m * sizeof(QDLDL_int));
    s->Lnz   = (QDLDL_int *)c_malloc(n_plus_m * sizeof(QDLDL_int));

    // Preallocate L matrix (Lx and Li are sparsity dependent)
    s->L->p = (c_int *)c_malloc((n_plus_m+1) * sizeof(QDLDL_int));

    // Lx and Li are sparsity dependent, so set them to
    // null initially so we don't try to free them prematurely
    s->L->i = OSQP_NULL;
    s->L->x = OSQP_NULL;

    // Preallocate workspace
    s->iwork = (QDLDL_int *)c_malloc(sizeof(QDLDL_int)*(3*n_plus_m));
    s->bwork = (QDLDL_bool *)c_malloc(sizeof(QDLDL_bool)*n_plus_m);
    s->fwork = (QDLDL_float *)c_malloc(sizeof(QDLDL_float)*n_plus_m);

    // Form and permute KKT matrix
    if (polish){ // Called from polish()
        // Use s->rho_inv_vec for storing param2 = vec(delta)
        for (i = 0; i < A->m; i++){
            s->rho_inv_vec[i] = sigma;
        }

        KKT_temp = form_KKT(P, A, 0, sigma, s->rho_inv_vec, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL);

        // Permute matrix
        if (KKT_temp)
            permute_KKT(&KKT_temp, s, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL);
    }
    else { // Called from ADMM algorithm

        // Allocate vectors of indices
        s->PtoKKT = c_malloc((P->p[P->n]) * sizeof(c_int));
        s->AtoKKT = c_malloc((A->p[A->n]) * sizeof(c_int));
        s->rhotoKKT = c_malloc((A->m) * sizeof(c_int));

        // Use p->rho_inv_vec for storing param2 = rho_inv_vec
        for (i = 0; i < A->m; i++){
            s->rho_inv_vec[i] = 1. / rho_vec[i];
        }

        KKT_temp = form_KKT(P, A, 0, sigma, s->rho_inv_vec,
                            s->PtoKKT, s->AtoKKT,
                            &(s->Pdiag_idx), &(s->Pdiag_n), s->rhotoKKT);

        // Permute matrix
        if (KKT_temp)
            permute_KKT(&KKT_temp, s, P->p[P->n], A->p[A->n], A->m, s->PtoKKT, s->AtoKKT, s->rhotoKKT);
    }

    // Check if matrix has been created
    if (!KKT_temp){
#ifdef PRINTING
        c_eprint("Error forming and permuting KKT matrix");
#endif
        free_linsys_solver_qdldl(s);
        *sp = OSQP_NULL;
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    // Factorize the KKT matrix
    if (LDL_factor(KKT_temp, s, P->n) < 0) {
        csc_spfree(KKT_temp);
        free_linsys_solver_qdldl(s);
        *sp = OSQP_NULL;
        return OSQP_NONCVX_ERROR;
    }

    if (polish){ // If KKT passed, assign it to KKT_temp
        // Polish, no need for KKT_temp
        csc_spfree(KKT_temp);
    }
    else { // If not embedded option 1 copy pointer to KKT_temp. Do not free it.
        s->KKT = KKT_temp;
    }


    // No error
    return 0;
}

#endif  // EMBEDDED


// Permute x = P*b using P
void permute_x(c_int n, c_float * x, const c_float * b, const c_int * P) {
    c_int j;
    for (j = 0 ; j < n ; j++) x[j] = b[P[j]];
}

// Permute x = P'*b using P
void permutet_x(c_int n, c_float * x, const c_float * b, const c_int * P) {
    c_int j;
    for (j = 0 ; j < n ; j++) x[P[j]] = b[j];
}


static void LDLSolve(c_float *x, c_float *b, const csc *L, const c_float *Dinv, const c_int *P, c_float *bp) {
    /* solves P'LDL'P x = b for x */
    permute_x(L->n, bp, b, P);
    QDLDL_solve(L->n, L->p, L->i, L->x, Dinv, bp);
    permutet_x(L->n, x, bp, P);

}


c_int solve_linsys_qdldl(qdldl_solver * s, c_float * b) {
    c_int j;

#ifndef EMBEDDED
    if (s->polish) {
        /* stores solution to the KKT system in b */
        LDLSolve(b, b, s->L, s->Dinv, s->P, s->bp);
    } else {
#endif
        /* stores solution to the KKT system in s->sol */
        LDLSolve(s->sol, b, s->L, s->Dinv, s->P, s->bp);

        /* copy x_tilde from s->sol */
        for (j = 0 ; j < s->n ; j++) {
            b[j] = s->sol[j];
        }

        /* compute z_tilde from b and s->sol */
        for (j = 0 ; j < s->m ; j++) {
            b[j + s->n] += s->rho_inv_vec[j] * s->sol[j + s->n];
        }
#ifndef EMBEDDED
    }
#endif

    return 0;
}


#if EMBEDDED != 1
// Update private structure with new P and A
c_int update_linsys_solver_matrices_qdldl(qdldl_solver * s, const csc *P, const csc *A) {

    // Update KKT matrix with new P
    update_KKT_P(s->KKT, P, s->PtoKKT, s->sigma, s->Pdiag_idx, s->Pdiag_n);

    // Update KKT matrix with new A
    update_KKT_A(s->KKT, A, s->AtoKKT);

    return (QDLDL_factor(s->KKT->n, s->KKT->p, s->KKT->i, s->KKT->x,
        s->L->p, s->L->i, s->L->x, s->D, s->Dinv, s->Lnz,
        s->etree, s->bwork, s->iwork, s->fwork) < 0);

}


c_int update_linsys_solver_rho_vec_qdldl(qdldl_solver * s, const c_float * rho_vec){
    c_int i;

    // Update internal rho_inv_vec
    for (i = 0; i < s->m; i++){
        s->rho_inv_vec[i] = 1. / rho_vec[i];
    }

    // Update KKT matrix with new rho_vec
    update_KKT_param2(s->KKT, s->rho_inv_vec, s->rhotoKKT, s->m);

    return (QDLDL_factor(s->KKT->n, s->KKT->p, s->KKT->i, s->KKT->x,
        s->L->p, s->L->i, s->L->x, s->D, s->Dinv, s->Lnz,
        s->etree, s->bwork, s->iwork, s->fwork) < 0);
}


#endif

#if EMBEDDED != 1


// Set values lower than threshold SCALING_REG to 1
void limit_scaling(c_float *D, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    D[i] = D[i] < MIN_SCALING ? 1.0 : D[i];
    D[i] = D[i] > MAX_SCALING ? MAX_SCALING : D[i];
  }
}

/**
 * Compute infinite norm of the columns of the KKT matrix without forming it
 *
 * The norm is stored in the vector v = (D, E)
 *
 * @param P        Cost matrix
 * @param A        Constraints matrix
 * @param D        Norm of columns related to variables
 * @param D_temp_A Temporary vector for norm of columns of A
 * @param E        Norm of columns related to constraints
 * @param n        Dimension of KKT matrix
 */
void compute_inf_norm_cols_KKT(const csc *P, const csc *A,
                               c_float *D, c_float *D_temp_A,
                               c_float *E, c_int n) {
  // First half
  //  [ P ]
  //  [ A ]
  mat_inf_norm_cols_sym_triu(P, D);
  mat_inf_norm_cols(A, D_temp_A);
  vec_ew_max_vec(D, D_temp_A, D, n);

  // Second half
  //  [ A']
  //  [ 0 ]
  mat_inf_norm_rows(A, E);
}

c_int scale_data(OSQPWorkspace *work) {
  // Scale KKT matrix
  //
  //    [ P   A']
  //    [ A   0 ]
  //
  // with diagonal matrix
  //
  //  S = [ D    ]
  //      [    E ]
  //

  c_int   i;          // Iterations index
  c_int   n, m;       // Number of constraints and variables
  c_float c_temp;     // Cost function scaling
  c_float inf_norm_q; // Infinity norm of q

  n = work->data->n;
  m = work->data->m;

  // Initialize scaling to 1
  work->scaling->c = 1.0;
  vec_set_scalar(work->scaling->D,    1., work->data->n);
  vec_set_scalar(work->scaling->Dinv, 1., work->data->n);
  vec_set_scalar(work->scaling->E,    1., work->data->m);
  vec_set_scalar(work->scaling->Einv, 1., work->data->m);


  for (i = 0; i < work->settings->scaling; i++) {
    //
    // First Ruiz step
    //

    // Compute norm of KKT columns
    compute_inf_norm_cols_KKT(work->data->P, work->data->A,
                              work->D_temp, work->D_temp_A,
                              work->E_temp, n);

    // Set to 1 values with 0 norms (avoid crazy scaling)
    limit_scaling(work->D_temp, n);
    limit_scaling(work->E_temp, m);

    // Take square root of norms
    vec_ew_sqrt(work->D_temp, n);
    vec_ew_sqrt(work->E_temp, m);

    // Divide scalings D and E by themselves
    vec_ew_recipr(work->D_temp, work->D_temp, n);
    vec_ew_recipr(work->E_temp, work->E_temp, m);

    // Equilibrate matrices P and A and vector q
    // P <- DPD
    mat_premult_diag(work->data->P, work->D_temp);
    mat_postmult_diag(work->data->P, work->D_temp);

    // A <- EAD
    mat_premult_diag(work->data->A, work->E_temp);
    mat_postmult_diag(work->data->A, work->D_temp);

    // q <- Dq
    vec_ew_prod(work->D_temp,     work->data->q, work->data->q,    n);

    // Update equilibration matrices D and E
    vec_ew_prod(work->scaling->D, work->D_temp,  work->scaling->D, n);
    vec_ew_prod(work->scaling->E, work->E_temp,  work->scaling->E, m);

    //
    // Cost normalization step
    //

    // Compute avg norm of cols of P
    mat_inf_norm_cols_sym_triu(work->data->P, work->D_temp);
    c_temp = vec_mean(work->D_temp, n);

    // Compute inf norm of q
    inf_norm_q = vec_norm_inf(work->data->q, n);

    // If norm_q == 0, set it to 1 (ignore it in the scaling)
    // NB: Using the same function as with vectors here
    limit_scaling(&inf_norm_q, 1);

    // Compute max between avg norm of cols of P and inf norm of q
    c_temp = c_max(c_temp, inf_norm_q);

    // Limit scaling (use same function as with vectors)
    limit_scaling(&c_temp, 1);

    // Invert scaling c = 1 / cost_measure
    c_temp = 1. / c_temp;

    // Scale P
    mat_mult_scalar(work->data->P, c_temp);

    // Scale q
    vec_mult_scalar(work->data->q, c_temp, n);

    // Update cost scaling
    work->scaling->c *= c_temp;
  }


  // Store cinv, Dinv, Einv
  work->scaling->cinv = 1. / work->scaling->c;
  vec_ew_recipr(work->scaling->D, work->scaling->Dinv, work->data->n);
  vec_ew_recipr(work->scaling->E, work->scaling->Einv, work->data->m);


  // Scale problem vectors l, u
  vec_ew_prod(work->scaling->E, work->data->l, work->data->l, work->data->m);
  vec_ew_prod(work->scaling->E, work->data->u, work->data->u, work->data->m);

  return 0;
}

#endif // EMBEDDED

c_int unscale_data(OSQPWorkspace *work) {
  // Unscale cost
  mat_mult_scalar(work->data->P, work->scaling->cinv);
  mat_premult_diag(work->data->P, work->scaling->Dinv);
  mat_postmult_diag(work->data->P, work->scaling->Dinv);
  vec_mult_scalar(work->data->q, work->scaling->cinv, work->data->n);
  vec_ew_prod(work->scaling->Dinv, work->data->q, work->data->q, work->data->n);

  // Unscale constraints
  mat_premult_diag(work->data->A, work->scaling->Einv);
  mat_postmult_diag(work->data->A, work->scaling->Dinv);
  vec_ew_prod(work->scaling->Einv, work->data->l, work->data->l, work->data->m);
  vec_ew_prod(work->scaling->Einv, work->data->u, work->data->u, work->data->m);

  return 0;
}

c_int unscale_solution(OSQPWorkspace *work) {
  // primal
  vec_ew_prod(work->scaling->D,
              work->solution->x,
              work->solution->x,
              work->data->n);

  // dual
  vec_ew_prod(work->scaling->E,
              work->solution->y,
              work->solution->y,
              work->data->m);
  vec_mult_scalar(work->solution->y, work->scaling->cinv, work->data->m);

  return 0;
}


/***************
* Versioning  *
***************/
const char* osqp_version(void) {
  return OSQP_VERSION;
}

/************************************
* Printing Constants to set Layout *
************************************/
#ifdef PRINTING
# define HEADER_LINE_LEN 65
#endif /* ifdef PRINTING */

/**********************
* Utility Functions  *
**********************/
void c_strcpy(char dest[], const char source[]) {
  int i = 0;

  while (1) {
    dest[i] = source[i];

    if (dest[i] == '\0') break;
    i++;
  }
}

#ifdef PRINTING

static void print_line(void) {
  char  the_line[HEADER_LINE_LEN + 1];
  c_int i;

  for (i = 0; i < HEADER_LINE_LEN; ++i) the_line[i] = '-';
  the_line[HEADER_LINE_LEN] = '\0';
  c_print("%s\n", the_line);
}

void print_header(void) {
  // Different indentation required for windows
#if defined(IS_WINDOWS) && !defined(PYTHON)
  c_print("iter  ");
#else
  c_print("iter   ");
#endif

  // Main information
  c_print("objective    pri res    dua res    rho");
# ifdef PROFILING
  c_print("        time");
# endif /* ifdef PROFILING */
  c_print("\n");
}

void print_setup_header(const OSQPWorkspace *work) {
  OSQPData *data;
  OSQPSettings *settings;
  c_int nnz; // Number of nonzeros in the problem

  data     = work->data;
  settings = work->settings;

  // Number of nonzeros
  nnz = data->P->p[data->P->n] + data->A->p[data->A->n];

  print_line();
  c_print("           OSQP v%s  -  Operator Splitting QP Solver\n"
          "              (c) Bartolomeo Stellato,  Goran Banjac\n"
          "        University of Oxford  -  Stanford University 2019\n",
          OSQP_VERSION);
  print_line();

  // Print variables and constraints
  c_print("problem:  ");
  c_print("variables n = %i, constraints m = %i\n          ",
                                    (int)data->n,
          (int)data->m);
  c_print("nnz(P) + nnz(A) = %i\n", (int)nnz);

  // Print Settings
  c_print("settings: ");
  c_print("linear system solver = %s",
          LINSYS_SOLVER_NAME[settings->linsys_solver]);

  if (work->linsys_solver->nthreads != 1) {
    c_print(" (%d threads)", (int)work->linsys_solver->nthreads);
  }
  c_print(",\n          ");

  c_print("eps_abs = %.1e, eps_rel = %.1e,\n          ",
          settings->eps_abs, settings->eps_rel);
  c_print("eps_prim_inf = %.1e, eps_dual_inf = %.1e,\n          ",
          settings->eps_prim_inf, settings->eps_dual_inf);
  c_print("rho = %.2e ", settings->rho);

  if (settings->adaptive_rho) c_print("(adaptive)");
  c_print(",\n          ");
  c_print("sigma = %.2e, alpha = %.2f, ",
          settings->sigma, settings->alpha);
  c_print("max_iter = %i\n", (int)settings->max_iter);

  if (settings->check_termination) c_print(
      "          check_termination: on (interval %i),\n",
      (int)settings->check_termination);
  else c_print("          check_termination: off,\n");

# ifdef PROFILING
  if (settings->time_limit) c_print("          time_limit: %.2e sec,\n",
                                    settings->time_limit);
# endif /* ifdef PROFILING */

  if (settings->scaling) c_print("          scaling: on, ");
  else c_print("          scaling: off, ");

  if (settings->scaled_termination) c_print("scaled_termination: on\n");
  else c_print("scaled_termination: off\n");

  if (settings->warm_start) c_print("          warm start: on, ");
  else c_print("          warm start: off, ");

  if (settings->polish) c_print("polish: on, ");
  else c_print("polish: off, ");

  if (settings->time_limit) c_print("time_limit: %.2e sec\n", settings->time_limit);
  else c_print("time_limit: off\n");

  c_print("\n");
}

void print_summary(OSQPWorkspace *work) {
  OSQPInfo *info;

  info = work->info;

  c_print("%4i",     (int)info->iter);
  c_print(" %12.4e", info->obj_val);
  c_print("  %9.2e", info->pri_res);
  c_print("  %9.2e", info->dua_res);
  c_print("  %9.2e", work->settings->rho);
# ifdef PROFILING

  if (work->first_run) {
    // total time: setup + solve
    c_print("  %9.2es", info->setup_time + info->solve_time);
  } else {
    // total time: update + solve
    c_print("  %9.2es", info->update_time + info->solve_time);
  }
# endif /* ifdef PROFILING */
  c_print("\n");

  work->summary_printed = 1; // Summary has been printed
}

void print_polish(OSQPWorkspace *work) {
  OSQPInfo *info;

  info = work->info;

  c_print("%4s",     "plsh");
  c_print(" %12.4e", info->obj_val);
  c_print("  %9.2e", info->pri_res);
  c_print("  %9.2e", info->dua_res);

  // Different characters for windows/unix
#if defined(IS_WINDOWS) && !defined(PYTHON)
  c_print("  ---------");
#else
  c_print("   --------");
#endif

# ifdef PROFILING
  if (work->first_run) {
    // total time: setup + solve
    c_print("  %9.2es", info->setup_time + info->solve_time +
            info->polish_time);
  } else {
    // total time: update + solve
    c_print("  %9.2es", info->update_time + info->solve_time +
            info->polish_time);
  }
# endif /* ifdef PROFILING */
  c_print("\n");
}

void print_footer(OSQPInfo *info, c_int polish) {
  c_print("\n"); // Add space after iterations

  c_print("status:               %s\n", info->status);

  if (polish && (info->status_val == OSQP_SOLVED)) {
    if (info->status_polish == 1) {
      c_print("solution polish:      successful\n");
    } else if (info->status_polish < 0) {
      c_print("solution polish:      unsuccessful\n");
    }
  }

  c_print("number of iterations: %i\n", (int)info->iter);

  if ((info->status_val == OSQP_SOLVED) ||
      (info->status_val == OSQP_SOLVED_INACCURATE)) {
    c_print("optimal objective:    %.4f\n", info->obj_val);
  }

# ifdef PROFILING
  c_print("run time:             %.2es\n", info->run_time);
# endif /* ifdef PROFILING */

# if EMBEDDED != 1
  c_print("optimal rho estimate: %.2e\n", info->rho_estimate);
# endif /* if EMBEDDED != 1 */
  c_print("\n");
}

#endif /* End #ifdef PRINTING */


#ifndef EMBEDDED

OSQPSettings* copy_settings(const OSQPSettings *settings) {
  OSQPSettings *new = c_malloc(sizeof(OSQPSettings));

  if (!new) return OSQP_NULL;

  // Copy settings
  // NB. Copying them explicitly because memcpy is not
  // defined when PRINTING is disabled (appears in string.h)
  new->rho = settings->rho;
  new->sigma = settings->sigma;
  new->scaling = settings->scaling;

# if EMBEDDED != 1
  new->adaptive_rho = settings->adaptive_rho;
  new->adaptive_rho_interval = settings->adaptive_rho_interval;
  new->adaptive_rho_tolerance = settings->adaptive_rho_tolerance;
# ifdef PROFILING
  new->adaptive_rho_fraction = settings->adaptive_rho_fraction;
# endif
# endif // EMBEDDED != 1
  new->max_iter = settings->max_iter;
  new->eps_abs = settings->eps_abs;
  new->eps_rel = settings->eps_rel;
  new->eps_prim_inf = settings->eps_prim_inf;
  new->eps_dual_inf = settings->eps_dual_inf;
  new->alpha = settings->alpha;
  new->linsys_solver = settings->linsys_solver;
  new->delta = settings->delta;
  new->polish = settings->polish;
  new->polish_refine_iter = settings->polish_refine_iter;
  new->verbose = settings->verbose;
  new->scaled_termination = settings->scaled_termination;
  new->check_termination = settings->check_termination;
  new->warm_start = settings->warm_start;
# ifdef PROFILING
  new->time_limit = settings->time_limit;
# endif

  return new;
}

#endif // #ifndef EMBEDDED


/*******************
* Timer Functions *
*******************/

#ifdef PROFILING

// Windows
# ifdef IS_WINDOWS

void osqp_tic(OSQPTimer *t)
{
  QueryPerformanceFrequency(&t->freq);
  QueryPerformanceCounter(&t->tic);
}

c_float osqp_toc(OSQPTimer *t)
{
  QueryPerformanceCounter(&t->toc);
  return (t->toc.QuadPart - t->tic.QuadPart) / (c_float)t->freq.QuadPart;
}

// Mac
# elif defined IS_MAC

void osqp_tic(OSQPTimer *t)
{
  /* read current clock cycles */
  t->tic = mach_absolute_time();
}

c_float osqp_toc(OSQPTimer *t)
{
  uint64_t duration; /* elapsed time in clock cycles*/

  t->toc   = mach_absolute_time();
  duration = t->toc - t->tic;

  /*conversion from clock cycles to nanoseconds*/
  mach_timebase_info(&(t->tinfo));
  duration *= t->tinfo.numer;
  duration /= t->tinfo.denom;

  return (c_float)duration / 1e9;
}

// Linux
# else  /* ifdef IS_WINDOWS */

/* read current time */
void osqp_tic(OSQPTimer *t)
{
  clock_gettime(CLOCK_MONOTONIC, &t->tic);
}

/* return time passed since last call to tic on this timer */
c_float osqp_toc(OSQPTimer *t)
{
  struct timespec temp;

  clock_gettime(CLOCK_MONOTONIC, &t->toc);

  if ((t->toc.tv_nsec - t->tic.tv_nsec) < 0) {
    temp.tv_sec  = t->toc.tv_sec - t->tic.tv_sec - 1;
    temp.tv_nsec = 1e9 + t->toc.tv_nsec - t->tic.tv_nsec;
  } else {
    temp.tv_sec  = t->toc.tv_sec - t->tic.tv_sec;
    temp.tv_nsec = t->toc.tv_nsec - t->tic.tv_nsec;
  }
  return (c_float)temp.tv_sec + (c_float)temp.tv_nsec / 1e9;
}

# endif /* ifdef IS_WINDOWS */

#endif // If Profiling end


/* ==================== DEBUG FUNCTIONS ======================= */



// If debug mode enabled
#ifdef DDEBUG

#ifdef PRINTING

void print_csc_matrix(csc *M, const char *name)
{
  c_int j, i, row_start, row_stop;
  c_int k = 0;

  // Print name
  c_print("%s :\n", name);

  for (j = 0; j < M->n; j++) {
    row_start = M->p[j];
    row_stop  = M->p[j + 1];

    if (row_start == row_stop) continue;
    else {
      for (i = row_start; i < row_stop; i++) {
        c_print("\t[%3u,%3u] = %.3g\n", (int)M->i[i], (int)j, M->x[k++]);
      }
    }
  }
}

void dump_csc_matrix(csc *M, const char *file_name) {
  c_int j, i, row_strt, row_stop;
  c_int k = 0;
  FILE *f = fopen(file_name, "w");

  if (f != NULL) {
    for (j = 0; j < M->n; j++) {
      row_strt = M->p[j];
      row_stop = M->p[j + 1];

      if (row_strt == row_stop) continue;
      else {
        for (i = row_strt; i < row_stop; i++) {
          fprintf(f, "%d\t%d\t%20.18e\n",
                  (int)M->i[i] + 1, (int)j + 1, M->x[k++]);
        }
      }
    }
    fprintf(f, "%d\t%d\t%20.18e\n", (int)M->m, (int)M->n, 0.0);
    fclose(f);
    c_print("File %s successfully written.\n", file_name);
  } else {
    c_eprint("Error during writing file %s.\n", file_name);
  }
}

void print_trip_matrix(csc *M, const char *name)
{
  c_int k = 0;

  // Print name
  c_print("%s :\n", name);

  for (k = 0; k < M->nz; k++) {
    c_print("\t[%3u, %3u] = %.3g\n", (int)M->i[k], (int)M->p[k], M->x[k]);
  }
}

void print_dns_matrix(c_float *M, c_int m, c_int n, const char *name)
{
  c_int i, j;

  c_print("%s : \n\t", name);

  for (i = 0; i < m; i++) {   // Cycle over rows
    for (j = 0; j < n; j++) { // Cycle over columns
      if (j < n - 1)
        // c_print("% 14.12e,  ", M[j*m+i]);
        c_print("% .3g,  ", M[j * m + i]);

      else
        // c_print("% 14.12e;  ", M[j*m+i]);
        c_print("% .3g;  ", M[j * m + i]);
    }

    if (i < m - 1) {
      c_print("\n\t");
    }
  }
  c_print("\n");
}

void print_vec(c_float *v, c_int n, const char *name) {
  print_dns_matrix(v, 1, n, name);
}

void dump_vec(c_float *v, c_int len, const char *file_name) {
  c_int i;
  FILE *f = fopen(file_name, "w");

  if (f != NULL) {
    for (i = 0; i < len; i++) {
      fprintf(f, "%20.18e\n", v[i]);
    }
    fclose(f);
    c_print("File %s successfully written.\n", file_name);
  } else {
    c_print("Error during writing file %s.\n", file_name);
  }
}

void print_vec_int(c_int *x, c_int n, const char *name) {
  c_int i;

  c_print("%s = [", name);

  for (i = 0; i < n; i++) {
    c_print(" %i ", (int)x[i]);
  }
  c_print("]\n");
}

#endif // PRINTING

#endif // DEBUG MODE

// Define data structure
c_int Pdata_i[117] = {
2,
3,
4,
5,
9,
10,
11,
14,
15,
16,
17,
21,
22,
23,
26,
27,
28,
29,
33,
34,
35,
38,
39,
40,
41,
45,
46,
47,
50,
51,
52,
53,
57,
58,
59,
62,
63,
64,
65,
69,
70,
71,
74,
75,
76,
77,
81,
82,
83,
86,
87,
88,
89,
93,
94,
95,
98,
99,
100,
101,
105,
106,
107,
110,
111,
112,
113,
117,
118,
119,
122,
123,
124,
125,
129,
130,
131,
132,
133,
134,
135,
136,
137,
138,
139,
140,
141,
142,
143,
144,
145,
146,
147,
148,
149,
150,
151,
152,
153,
154,
155,
156,
157,
158,
159,
160,
161,
162,
163,
164,
165,
166,
167,
168,
169,
170,
171,
};
c_int Pdata_p[173] = {
0,
0,
0,
1,
2,
3,
4,
4,
4,
4,
5,
6,
7,
7,
7,
8,
9,
10,
11,
11,
11,
11,
12,
13,
14,
14,
14,
15,
16,
17,
18,
18,
18,
18,
19,
20,
21,
21,
21,
22,
23,
24,
25,
25,
25,
25,
26,
27,
28,
28,
28,
29,
30,
31,
32,
32,
32,
32,
33,
34,
35,
35,
35,
36,
37,
38,
39,
39,
39,
39,
40,
41,
42,
42,
42,
43,
44,
45,
46,
46,
46,
46,
47,
48,
49,
49,
49,
50,
51,
52,
53,
53,
53,
53,
54,
55,
56,
56,
56,
57,
58,
59,
60,
60,
60,
60,
61,
62,
63,
63,
63,
64,
65,
66,
67,
67,
67,
67,
68,
69,
70,
70,
70,
71,
72,
73,
74,
74,
74,
74,
75,
76,
77,
78,
79,
80,
81,
82,
83,
84,
85,
86,
87,
88,
89,
90,
91,
92,
93,
94,
95,
96,
97,
98,
99,
100,
101,
102,
103,
104,
105,
106,
107,
108,
109,
110,
111,
112,
113,
114,
115,
116,
117,
};
c_float Pdata_x[117] = {
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.39763536438352536928,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
(c_float)0.01225385481008078002,
};
csc Pdata = {117, 172, 172, Pdata_p, Pdata_i, Pdata_x, -1};
c_int Adata_i[884] = {
0,
12,
15,
21,
132,
1,
13,
16,
22,
133,
2,
14,
134,
3,
15,
135,
4,
16,
136,
5,
17,
137,
6,
12,
15,
18,
21,
138,
7,
13,
16,
19,
22,
139,
8,
14,
20,
140,
9,
15,
21,
141,
10,
16,
22,
142,
11,
17,
23,
143,
12,
24,
27,
33,
144,
13,
25,
28,
34,
145,
14,
26,
146,
15,
27,
147,
16,
28,
148,
17,
29,
149,
18,
24,
27,
30,
33,
150,
19,
25,
28,
31,
34,
151,
20,
26,
32,
152,
21,
27,
33,
153,
22,
28,
34,
154,
23,
29,
35,
155,
24,
36,
39,
45,
156,
25,
37,
40,
46,
157,
26,
38,
158,
27,
39,
159,
28,
40,
160,
29,
41,
161,
30,
36,
39,
42,
45,
162,
31,
37,
40,
43,
46,
163,
32,
38,
44,
164,
33,
39,
45,
165,
34,
40,
46,
166,
35,
41,
47,
167,
36,
48,
51,
57,
168,
37,
49,
52,
58,
169,
38,
50,
170,
39,
51,
171,
40,
52,
172,
41,
53,
173,
42,
48,
51,
54,
57,
174,
43,
49,
52,
55,
58,
175,
44,
50,
56,
176,
45,
51,
57,
177,
46,
52,
58,
178,
47,
53,
59,
179,
48,
60,
63,
69,
180,
49,
61,
64,
70,
181,
50,
62,
182,
51,
63,
183,
52,
64,
184,
53,
65,
185,
54,
60,
63,
66,
69,
186,
55,
61,
64,
67,
70,
187,
56,
62,
68,
188,
57,
63,
69,
189,
58,
64,
70,
190,
59,
65,
71,
191,
60,
72,
75,
81,
192,
61,
73,
76,
82,
193,
62,
74,
194,
63,
75,
195,
64,
76,
196,
65,
77,
197,
66,
72,
75,
78,
81,
198,
67,
73,
76,
79,
82,
199,
68,
74,
80,
200,
69,
75,
81,
201,
70,
76,
82,
202,
71,
77,
83,
203,
72,
84,
87,
93,
204,
73,
85,
88,
94,
205,
74,
86,
206,
75,
87,
207,
76,
88,
208,
77,
89,
209,
78,
84,
87,
90,
93,
210,
79,
85,
88,
91,
94,
211,
80,
86,
92,
212,
81,
87,
93,
213,
82,
88,
94,
214,
83,
89,
95,
215,
84,
96,
99,
105,
216,
85,
97,
100,
106,
217,
86,
98,
218,
87,
99,
219,
88,
100,
220,
89,
101,
221,
90,
96,
99,
102,
105,
222,
91,
97,
100,
103,
106,
223,
92,
98,
104,
224,
93,
99,
105,
225,
94,
100,
106,
226,
95,
101,
107,
227,
96,
108,
111,
117,
228,
97,
109,
112,
118,
229,
98,
110,
230,
99,
111,
231,
100,
112,
232,
101,
113,
233,
102,
108,
111,
114,
117,
234,
103,
109,
112,
115,
118,
235,
104,
110,
116,
236,
105,
111,
117,
237,
106,
112,
118,
238,
107,
113,
119,
239,
108,
120,
123,
129,
240,
109,
121,
124,
130,
241,
110,
122,
242,
111,
123,
243,
112,
124,
244,
113,
125,
245,
114,
120,
123,
126,
129,
246,
115,
121,
124,
127,
130,
247,
116,
122,
128,
248,
117,
123,
129,
249,
118,
124,
130,
250,
119,
125,
131,
251,
120,
252,
121,
253,
122,
254,
123,
255,
124,
256,
125,
257,
126,
258,
127,
259,
128,
260,
129,
261,
130,
262,
131,
263,
13,
14,
16,
17,
19,
20,
22,
23,
264,
12,
14,
15,
17,
18,
20,
21,
23,
265,
13,
14,
16,
17,
19,
20,
22,
23,
266,
12,
14,
15,
17,
18,
20,
21,
23,
267,
25,
26,
28,
29,
31,
32,
34,
35,
268,
24,
26,
27,
29,
30,
32,
33,
35,
269,
25,
26,
28,
29,
31,
32,
34,
35,
270,
24,
26,
27,
29,
30,
32,
33,
35,
271,
37,
38,
40,
41,
43,
44,
46,
47,
272,
36,
38,
39,
41,
42,
44,
45,
47,
273,
37,
38,
40,
41,
43,
44,
46,
47,
274,
36,
38,
39,
41,
42,
44,
45,
47,
275,
49,
50,
52,
53,
55,
56,
58,
59,
276,
48,
50,
51,
53,
54,
56,
57,
59,
277,
49,
50,
52,
53,
55,
56,
58,
59,
278,
48,
50,
51,
53,
54,
56,
57,
59,
279,
61,
62,
64,
65,
67,
68,
70,
71,
280,
60,
62,
63,
65,
66,
68,
69,
71,
281,
61,
62,
64,
65,
67,
68,
70,
71,
282,
60,
62,
63,
65,
66,
68,
69,
71,
283,
73,
74,
76,
77,
79,
80,
82,
83,
284,
72,
74,
75,
77,
78,
80,
81,
83,
285,
73,
74,
76,
77,
79,
80,
82,
83,
286,
72,
74,
75,
77,
78,
80,
81,
83,
287,
85,
86,
88,
89,
91,
92,
94,
95,
288,
84,
86,
87,
89,
90,
92,
93,
95,
289,
85,
86,
88,
89,
91,
92,
94,
95,
290,
84,
86,
87,
89,
90,
92,
93,
95,
291,
97,
98,
100,
101,
103,
104,
106,
107,
292,
96,
98,
99,
101,
102,
104,
105,
107,
293,
97,
98,
100,
101,
103,
104,
106,
107,
294,
96,
98,
99,
101,
102,
104,
105,
107,
295,
109,
110,
112,
113,
115,
116,
118,
119,
296,
108,
110,
111,
113,
114,
116,
117,
119,
297,
109,
110,
112,
113,
115,
116,
118,
119,
298,
108,
110,
111,
113,
114,
116,
117,
119,
299,
121,
122,
124,
125,
127,
128,
130,
131,
300,
120,
122,
123,
125,
126,
128,
129,
131,
301,
121,
122,
124,
125,
127,
128,
130,
131,
302,
120,
122,
123,
125,
126,
128,
129,
131,
303,
};
c_int Adata_p[173] = {
0,
5,
10,
13,
16,
19,
22,
28,
34,
38,
42,
46,
50,
55,
60,
63,
66,
69,
72,
78,
84,
88,
92,
96,
100,
105,
110,
113,
116,
119,
122,
128,
134,
138,
142,
146,
150,
155,
160,
163,
166,
169,
172,
178,
184,
188,
192,
196,
200,
205,
210,
213,
216,
219,
222,
228,
234,
238,
242,
246,
250,
255,
260,
263,
266,
269,
272,
278,
284,
288,
292,
296,
300,
305,
310,
313,
316,
319,
322,
328,
334,
338,
342,
346,
350,
355,
360,
363,
366,
369,
372,
378,
384,
388,
392,
396,
400,
405,
410,
413,
416,
419,
422,
428,
434,
438,
442,
446,
450,
455,
460,
463,
466,
469,
472,
478,
484,
488,
492,
496,
500,
502,
504,
506,
508,
510,
512,
514,
516,
518,
520,
522,
524,
533,
542,
551,
560,
569,
578,
587,
596,
605,
614,
623,
632,
641,
650,
659,
668,
677,
686,
695,
704,
713,
722,
731,
740,
749,
758,
767,
776,
785,
794,
803,
812,
821,
830,
839,
848,
857,
866,
875,
884,
};
c_float Adata_x[884] = {
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.08678003520989943653,
(c_float)0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.08678003520989943653,
(c_float)-0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.10000000000000000555,
(c_float)0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.10000000000000000555,
(c_float)-0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)-0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)0.98459999999999991971,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.08678003520989943653,
(c_float)0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.08678003520989943653,
(c_float)-0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)-0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)-0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)0.98459999999999991971,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.08678003520989943653,
(c_float)0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.08678003520989943653,
(c_float)-0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)-0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)-0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)0.98459999999999991971,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.08678003520989943653,
(c_float)0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.08678003520989943653,
(c_float)-0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)-0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)-0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)0.98459999999999991971,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.08678003520989943653,
(c_float)0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.08678003520989943653,
(c_float)-0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)-0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)-0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)0.98459999999999991971,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.08678003520989943653,
(c_float)0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.08678003520989943653,
(c_float)-0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)-0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)-0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)0.98459999999999991971,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.08678003520989943653,
(c_float)0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.08678003520989943653,
(c_float)-0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)-0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)-0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)0.98459999999999991971,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.08678003520989943653,
(c_float)0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.08678003520989943653,
(c_float)-0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)-0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)-0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)0.98459999999999991971,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.08678003520989943653,
(c_float)0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.08678003520989943653,
(c_float)-0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)-0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)-0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)0.98459999999999991971,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.08678003520989943653,
(c_float)0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.08678003520989943653,
(c_float)-0.99994734478168090241,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)-0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)-0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)0.11796934580826992645,
(c_float)0.67639925404571177303,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)0.98459999999999991971,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)1.00000000000000000000,
(c_float)-0.68697872643277657634,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
};
csc Adata = {884, 304, 172, Adata_p, Adata_i, Adata_x, -1};
c_float qdata[172] = {
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
};
c_float ldata[304] = {
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.52359877559829881566,
(c_float)-0.52359877559829881566,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1.77827941003892275873,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-0.52359877559829881566,
(c_float)-0.52359877559829881566,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1.77827941003892275873,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-0.52359877559829881566,
(c_float)-0.52359877559829881566,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1.77827941003892275873,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-0.52359877559829881566,
(c_float)-0.52359877559829881566,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1.77827941003892275873,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-0.52359877559829881566,
(c_float)-0.52359877559829881566,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1.77827941003892275873,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-0.52359877559829881566,
(c_float)-0.52359877559829881566,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1.77827941003892275873,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-0.52359877559829881566,
(c_float)-0.52359877559829881566,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1.77827941003892275873,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-0.52359877559829881566,
(c_float)-0.52359877559829881566,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1.77827941003892275873,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-0.52359877559829881566,
(c_float)-0.52359877559829881566,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1.77827941003892275873,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-0.52359877559829881566,
(c_float)-0.52359877559829881566,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1.77827941003892275873,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-0.52359877559829881566,
(c_float)-0.52359877559829881566,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1778279410038922838185969975296.00000000000000000000,
(c_float)-1.77827941003892275873,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1000000000000000019884624838656.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1495348781221220583518584700928.00000000000000000000,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
(c_float)-1.19410417329134777020,
};
c_float udata[304] = {
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.52359877559829881566,
(c_float)0.52359877559829881566,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)0.52359877559829881566,
(c_float)0.52359877559829881566,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)0.52359877559829881566,
(c_float)0.52359877559829881566,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)0.52359877559829881566,
(c_float)0.52359877559829881566,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)0.52359877559829881566,
(c_float)0.52359877559829881566,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)0.52359877559829881566,
(c_float)0.52359877559829881566,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)0.52359877559829881566,
(c_float)0.52359877559829881566,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)0.52359877559829881566,
(c_float)0.52359877559829881566,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)0.52359877559829881566,
(c_float)0.52359877559829881566,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)0.52359877559829881566,
(c_float)0.52359877559829881566,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)0.52359877559829881566,
(c_float)0.52359877559829881566,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1778279410038922838185969975296.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1000000000000000019884624838656.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)1495348781221220583518584700928.00000000000000000000,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
(c_float)2.90024252819169214845,
};
OSQPData data = {172, 304, &Pdata, &Adata, qdata, ldata, udata};

// Define settings structure
OSQPSettings settings = {(c_float)0.10000000000000000555, (c_float)0.00000100000000000000, 10, 1, 0, (c_float)5.00000000000000000000,4000, (c_float)0.00100000000000000002, (c_float)0.00100000000000000002, (c_float)0.00010000000000000000, (c_float)0.00010000000000000000, (c_float)1.60000000000000008882, (enum linsys_solver_type) LINSYS_SOLVER, 0, 25, 1, };

// Define scaling structure
c_float Dscaling[172] = {
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
};
c_float Dinvscaling[172] = {
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
};
c_float Escaling[304] = {
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)1.02727280129615849980,
(c_float)1.02727280129615849980,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)1.02727280129615849980,
(c_float)1.02727280129615849980,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)1.02727280129615849980,
(c_float)1.02727280129615849980,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)1.02727280129615849980,
(c_float)1.02727280129615849980,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)1.02727280129615849980,
(c_float)1.02727280129615849980,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)1.02727280129615849980,
(c_float)1.02727280129615849980,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)1.02727280129615849980,
(c_float)1.02727280129615849980,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)1.02727280129615849980,
(c_float)1.02727280129615849980,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)1.02727280129615849980,
(c_float)1.02727280129615849980,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)1.02727280129615849980,
(c_float)1.02727280129615849980,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.77827941003892275873,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.49534878122122050215,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
(c_float)1.20421961808324695653,
};
c_float Einvscaling[304] = {
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.00000000000000000000,
(c_float)0.97345125728847570201,
(c_float)0.97345125728847570201,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.00000000000000000000,
(c_float)0.97345125728847570201,
(c_float)0.97345125728847570201,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.00000000000000000000,
(c_float)0.97345125728847570201,
(c_float)0.97345125728847570201,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.00000000000000000000,
(c_float)0.97345125728847570201,
(c_float)0.97345125728847570201,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.00000000000000000000,
(c_float)0.97345125728847570201,
(c_float)0.97345125728847570201,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.00000000000000000000,
(c_float)0.97345125728847570201,
(c_float)0.97345125728847570201,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.00000000000000000000,
(c_float)0.97345125728847570201,
(c_float)0.97345125728847570201,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.00000000000000000000,
(c_float)0.97345125728847570201,
(c_float)0.97345125728847570201,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.00000000000000000000,
(c_float)0.97345125728847570201,
(c_float)0.97345125728847570201,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.20465762771004780696,
(c_float)1.20465762771004780696,
(c_float)1.00000000000000000000,
(c_float)0.97345125728847570201,
(c_float)0.97345125728847570201,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)0.56234132519034907283,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.66874030497642200643,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
(c_float)0.83041331081426594807,
};
OSQPScaling scaling = {(c_float)0.17782794100389229253, Dscaling, Escaling, (c_float)5.62341325190349028418, Dinvscaling, Einvscaling};

// Define linsys_solver structure
c_int linsys_solver_L_i[2200] = {
1,
9,
465,
3,
8,
9,
468,
5,
9,
471,
7,
8,
471,
473,
9,
459,
460,
461,
462,
468,
471,
473,
459,
460,
461,
462,
465,
468,
471,
473,
11,
60,
466,
13,
59,
60,
469,
15,
58,
59,
60,
464,
458,
460,
462,
57,
458,
460,
462,
21,
54,
60,
23,
53,
54,
59,
25,
52,
53,
54,
58,
57,
55,
56,
51,
55,
56,
57,
31,
50,
33,
50,
54,
35,
49,
37,
49,
50,
53,
39,
43,
51,
463,
475,
51,
463,
475,
45,
48,
47,
48,
49,
50,
52,
49,
50,
51,
52,
463,
475,
50,
51,
52,
53,
463,
475,
51,
52,
53,
54,
463,
475,
52,
53,
54,
55,
56,
57,
463,
475,
53,
54,
55,
56,
57,
58,
463,
475,
54,
55,
56,
57,
58,
59,
463,
475,
55,
56,
57,
58,
59,
60,
463,
475,
56,
57,
58,
59,
60,
463,
471,
472,
473,
474,
475,
57,
58,
59,
60,
463,
471,
472,
473,
474,
475,
58,
59,
60,
458,
460,
462,
463,
471,
472,
473,
474,
475,
59,
60,
458,
460,
462,
463,
464,
471,
472,
473,
474,
475,
60,
458,
460,
462,
463,
464,
469,
471,
472,
473,
474,
475,
458,
460,
462,
463,
464,
466,
469,
471,
472,
473,
474,
475,
62,
70,
467,
64,
69,
70,
470,
66,
70,
472,
68,
69,
472,
474,
70,
459,
460,
461,
462,
470,
472,
474,
459,
460,
461,
462,
467,
470,
472,
474,
72,
80,
74,
80,
471,
76,
79,
78,
79,
80,
473,
80,
142,
143,
463,
473,
475,
142,
143,
463,
471,
473,
475,
82,
90,
84,
90,
472,
86,
89,
88,
89,
90,
474,
90,
142,
143,
463,
474,
475,
142,
143,
463,
472,
474,
475,
92,
141,
455,
94,
140,
141,
456,
96,
139,
140,
141,
454,
457,
459,
461,
138,
457,
459,
461,
102,
135,
141,
104,
134,
135,
140,
106,
133,
134,
135,
139,
138,
136,
137,
132,
136,
137,
138,
112,
131,
114,
131,
135,
116,
130,
118,
130,
131,
134,
120,
124,
132,
142,
143,
132,
142,
143,
126,
129,
128,
129,
130,
131,
133,
130,
131,
132,
133,
142,
143,
131,
132,
133,
134,
142,
143,
132,
133,
134,
135,
142,
143,
133,
134,
135,
136,
137,
138,
142,
143,
134,
135,
136,
137,
138,
139,
142,
143,
135,
136,
137,
138,
139,
140,
142,
143,
136,
137,
138,
139,
140,
141,
142,
143,
137,
138,
139,
140,
141,
142,
143,
471,
472,
473,
474,
138,
139,
140,
141,
142,
143,
471,
472,
473,
474,
139,
140,
141,
142,
143,
457,
459,
461,
471,
472,
473,
474,
140,
141,
142,
143,
454,
457,
459,
461,
471,
472,
473,
474,
141,
142,
143,
454,
456,
457,
459,
461,
471,
472,
473,
474,
142,
143,
454,
455,
456,
457,
459,
461,
471,
472,
473,
474,
143,
454,
455,
456,
457,
459,
461,
463,
471,
472,
473,
474,
475,
454,
455,
456,
457,
459,
461,
463,
471,
472,
473,
474,
475,
451,
452,
453,
152,
451,
452,
453,
152,
153,
154,
152,
153,
154,
457,
153,
154,
451,
452,
453,
454,
455,
456,
457,
154,
451,
452,
453,
454,
455,
456,
457,
465,
467,
468,
470,
451,
452,
453,
454,
455,
456,
457,
465,
467,
468,
470,
156,
169,
449,
158,
168,
169,
450,
160,
167,
168,
169,
448,
162,
169,
455,
164,
168,
455,
456,
166,
167,
454,
455,
456,
168,
169,
448,
451,
452,
453,
454,
455,
456,
169,
448,
450,
451,
452,
453,
454,
455,
456,
448,
449,
450,
451,
452,
453,
454,
455,
456,
445,
446,
447,
178,
445,
446,
447,
178,
179,
180,
178,
179,
180,
458,
179,
180,
445,
446,
447,
458,
464,
466,
469,
180,
445,
446,
447,
458,
464,
465,
466,
467,
468,
469,
470,
445,
446,
447,
458,
464,
465,
466,
467,
468,
469,
470,
182,
190,
443,
184,
189,
190,
444,
186,
190,
467,
188,
189,
467,
470,
190,
444,
446,
447,
452,
453,
467,
470,
443,
444,
446,
447,
452,
453,
467,
470,
192,
200,
441,
194,
199,
200,
442,
196,
200,
465,
198,
199,
465,
468,
200,
442,
446,
447,
452,
453,
465,
468,
441,
442,
446,
447,
452,
453,
465,
468,
202,
215,
439,
204,
214,
215,
440,
206,
213,
214,
215,
438,
208,
215,
466,
210,
214,
466,
469,
212,
213,
464,
466,
469,
214,
215,
438,
445,
446,
447,
464,
466,
469,
215,
438,
440,
445,
446,
447,
464,
466,
469,
438,
439,
440,
445,
446,
447,
464,
466,
469,
435,
436,
437,
224,
435,
436,
437,
224,
225,
226,
224,
225,
226,
445,
225,
226,
435,
436,
437,
438,
439,
440,
445,
226,
435,
436,
437,
438,
439,
440,
441,
442,
443,
444,
445,
435,
436,
437,
438,
439,
440,
441,
442,
443,
444,
445,
228,
241,
434,
230,
240,
241,
433,
232,
239,
240,
241,
432,
234,
241,
439,
236,
240,
439,
440,
238,
239,
438,
439,
440,
240,
241,
432,
435,
436,
437,
438,
439,
440,
241,
432,
433,
435,
436,
437,
438,
439,
440,
432,
433,
434,
435,
436,
437,
438,
439,
440,
429,
430,
431,
250,
429,
430,
431,
250,
251,
252,
250,
251,
252,
451,
251,
252,
429,
430,
431,
448,
449,
450,
451,
252,
429,
430,
431,
441,
442,
443,
444,
448,
449,
450,
451,
429,
430,
431,
441,
442,
443,
444,
448,
449,
450,
451,
254,
262,
427,
256,
261,
262,
428,
258,
262,
441,
260,
261,
441,
442,
262,
428,
430,
431,
436,
437,
441,
442,
427,
428,
430,
431,
436,
437,
441,
442,
264,
272,
425,
266,
271,
272,
426,
268,
272,
443,
270,
271,
443,
444,
272,
426,
430,
431,
436,
437,
443,
444,
425,
426,
430,
431,
436,
437,
443,
444,
274,
287,
424,
276,
286,
287,
423,
278,
285,
286,
287,
422,
280,
287,
449,
282,
286,
449,
450,
284,
285,
448,
449,
450,
286,
287,
422,
429,
430,
431,
448,
449,
450,
287,
422,
423,
429,
430,
431,
448,
449,
450,
422,
423,
424,
429,
430,
431,
448,
449,
450,
421,
292,
293,
292,
293,
421,
435,
293,
421,
425,
426,
427,
428,
432,
433,
434,
435,
421,
425,
426,
427,
428,
432,
433,
434,
435,
420,
298,
299,
298,
299,
420,
429,
299,
420,
422,
423,
424,
425,
426,
427,
428,
429,
420,
422,
423,
424,
425,
426,
427,
428,
429,
301,
419,
425,
426,
303,
418,
425,
305,
417,
427,
428,
307,
416,
427,
309,
319,
416,
417,
312,
312,
318,
315,
315,
318,
319,
317,
318,
416,
319,
372,
373,
414,
415,
416,
372,
373,
414,
415,
416,
417,
321,
331,
418,
419,
324,
324,
330,
327,
327,
330,
331,
329,
330,
418,
331,
372,
373,
414,
415,
418,
372,
373,
414,
415,
418,
419,
369,
370,
371,
369,
370,
371,
421,
337,
368,
432,
433,
434,
339,
367,
433,
434,
341,
366,
434,
343,
365,
366,
367,
368,
345,
364,
366,
367,
347,
363,
366,
372,
373,
361,
361,
360,
360,
356,
356,
363,
359,
359,
363,
364,
363,
364,
365,
362,
363,
364,
365,
363,
364,
365,
369,
372,
373,
364,
365,
366,
369,
372,
373,
365,
366,
367,
369,
372,
373,
366,
367,
368,
369,
372,
373,
367,
368,
369,
370,
371,
372,
373,
434,
368,
369,
370,
371,
372,
373,
433,
434,
369,
370,
371,
372,
373,
432,
433,
434,
370,
371,
372,
373,
421,
432,
433,
434,
371,
372,
373,
416,
417,
418,
419,
421,
432,
433,
434,
372,
373,
416,
417,
418,
419,
421,
432,
433,
434,
373,
414,
415,
416,
417,
418,
419,
421,
432,
433,
434,
414,
415,
416,
417,
418,
419,
421,
432,
433,
434,
411,
412,
413,
411,
412,
413,
420,
379,
410,
422,
423,
424,
381,
409,
423,
424,
383,
408,
424,
385,
407,
408,
409,
410,
387,
406,
408,
409,
389,
405,
408,
414,
415,
403,
403,
402,
402,
398,
398,
405,
401,
401,
405,
406,
405,
406,
407,
404,
405,
406,
407,
405,
406,
407,
411,
414,
415,
406,
407,
408,
411,
414,
415,
407,
408,
409,
411,
414,
415,
408,
409,
410,
411,
414,
415,
409,
410,
411,
412,
413,
414,
415,
424,
410,
411,
412,
413,
414,
415,
423,
424,
411,
412,
413,
414,
415,
422,
423,
424,
412,
413,
414,
415,
420,
422,
423,
424,
413,
414,
415,
416,
417,
418,
419,
420,
422,
423,
424,
414,
415,
416,
417,
418,
419,
420,
422,
423,
424,
415,
416,
417,
418,
419,
420,
421,
422,
423,
424,
432,
433,
434,
416,
417,
418,
419,
420,
421,
422,
423,
424,
432,
433,
434,
417,
418,
419,
420,
421,
422,
423,
424,
427,
432,
433,
434,
418,
419,
420,
421,
422,
423,
424,
427,
428,
432,
433,
434,
419,
420,
421,
422,
423,
424,
425,
427,
428,
432,
433,
434,
420,
421,
422,
423,
424,
425,
426,
427,
428,
432,
433,
434,
421,
422,
423,
424,
425,
426,
427,
428,
429,
432,
433,
434,
422,
423,
424,
425,
426,
427,
428,
429,
432,
433,
434,
435,
423,
424,
425,
426,
427,
428,
429,
430,
431,
432,
433,
434,
435,
448,
449,
450,
424,
425,
426,
427,
428,
429,
430,
431,
432,
433,
434,
435,
448,
449,
450,
425,
426,
427,
428,
429,
430,
431,
432,
433,
434,
435,
448,
449,
450,
426,
427,
428,
429,
430,
431,
432,
433,
434,
435,
436,
437,
443,
444,
448,
449,
450,
427,
428,
429,
430,
431,
432,
433,
434,
435,
436,
437,
443,
444,
448,
449,
450,
428,
429,
430,
431,
432,
433,
434,
435,
436,
437,
441,
442,
443,
444,
448,
449,
450,
429,
430,
431,
432,
433,
434,
435,
436,
437,
441,
442,
443,
444,
448,
449,
450,
430,
431,
432,
433,
434,
435,
436,
437,
441,
442,
443,
444,
448,
449,
450,
451,
431,
432,
433,
434,
435,
436,
437,
441,
442,
443,
444,
448,
449,
450,
451,
432,
433,
434,
435,
436,
437,
441,
442,
443,
444,
448,
449,
450,
451,
433,
434,
435,
436,
437,
438,
439,
440,
441,
442,
443,
444,
448,
449,
450,
451,
434,
435,
436,
437,
438,
439,
440,
441,
442,
443,
444,
448,
449,
450,
451,
435,
436,
437,
438,
439,
440,
441,
442,
443,
444,
448,
449,
450,
451,
436,
437,
438,
439,
440,
441,
442,
443,
444,
445,
448,
449,
450,
451,
437,
438,
439,
440,
441,
442,
443,
444,
445,
448,
449,
450,
451,
438,
439,
440,
441,
442,
443,
444,
445,
448,
449,
450,
451,
439,
440,
441,
442,
443,
444,
445,
446,
447,
448,
449,
450,
451,
464,
466,
469,
440,
441,
442,
443,
444,
445,
446,
447,
448,
449,
450,
451,
464,
466,
469,
441,
442,
443,
444,
445,
446,
447,
448,
449,
450,
451,
464,
466,
469,
442,
443,
444,
445,
446,
447,
448,
449,
450,
451,
452,
453,
464,
465,
466,
468,
469,
443,
444,
445,
446,
447,
448,
449,
450,
451,
452,
453,
464,
465,
466,
468,
469,
444,
445,
446,
447,
448,
449,
450,
451,
452,
453,
464,
465,
466,
467,
468,
469,
470,
445,
446,
447,
448,
449,
450,
451,
452,
453,
464,
465,
466,
467,
468,
469,
470,
446,
447,
448,
449,
450,
451,
452,
453,
458,
464,
465,
466,
467,
468,
469,
470,
447,
448,
449,
450,
451,
452,
453,
458,
464,
465,
466,
467,
468,
469,
470,
448,
449,
450,
451,
452,
453,
458,
464,
465,
466,
467,
468,
469,
470,
449,
450,
451,
452,
453,
454,
455,
456,
458,
464,
465,
466,
467,
468,
469,
470,
450,
451,
452,
453,
454,
455,
456,
458,
464,
465,
466,
467,
468,
469,
470,
451,
452,
453,
454,
455,
456,
458,
464,
465,
466,
467,
468,
469,
470,
452,
453,
454,
455,
456,
457,
458,
464,
465,
466,
467,
468,
469,
470,
453,
454,
455,
456,
457,
458,
464,
465,
466,
467,
468,
469,
470,
454,
455,
456,
457,
458,
464,
465,
466,
467,
468,
469,
470,
455,
456,
457,
458,
459,
461,
463,
464,
465,
466,
467,
468,
469,
470,
471,
472,
473,
474,
475,
456,
457,
458,
459,
461,
463,
464,
465,
466,
467,
468,
469,
470,
471,
472,
473,
474,
475,
457,
458,
459,
461,
463,
464,
465,
466,
467,
468,
469,
470,
471,
472,
473,
474,
475,
458,
459,
461,
463,
464,
465,
466,
467,
468,
469,
470,
471,
472,
473,
474,
475,
459,
460,
461,
462,
463,
464,
465,
466,
467,
468,
469,
470,
471,
472,
473,
474,
475,
460,
461,
462,
463,
464,
465,
466,
467,
468,
469,
470,
471,
472,
473,
474,
475,
461,
462,
463,
464,
465,
466,
467,
468,
469,
470,
471,
472,
473,
474,
475,
462,
463,
464,
465,
466,
467,
468,
469,
470,
471,
472,
473,
474,
475,
463,
464,
465,
466,
467,
468,
469,
470,
471,
472,
473,
474,
475,
464,
465,
466,
467,
468,
469,
470,
471,
472,
473,
474,
475,
465,
466,
467,
468,
469,
470,
471,
472,
473,
474,
475,
466,
467,
468,
469,
470,
471,
472,
473,
474,
475,
467,
468,
469,
470,
471,
472,
473,
474,
475,
468,
469,
470,
471,
472,
473,
474,
475,
469,
470,
471,
472,
473,
474,
475,
470,
471,
472,
473,
474,
475,
471,
472,
473,
474,
475,
472,
473,
474,
475,
473,
474,
475,
474,
475,
475,
};
c_int linsys_solver_L_p[477] = {
0,
1,
3,
4,
7,
8,
10,
11,
14,
22,
30,
31,
33,
34,
37,
38,
42,
43,
44,
45,
49,
50,
52,
53,
56,
57,
61,
62,
63,
64,
68,
69,
70,
71,
73,
74,
75,
76,
79,
80,
81,
82,
83,
84,
87,
88,
89,
90,
94,
100,
106,
112,
120,
128,
136,
144,
155,
165,
177,
189,
201,
213,
214,
216,
217,
220,
221,
223,
224,
227,
235,
243,
244,
245,
246,
248,
249,
250,
251,
254,
260,
266,
267,
268,
269,
271,
272,
273,
274,
277,
283,
289,
290,
292,
293,
296,
297,
301,
302,
303,
304,
308,
309,
311,
312,
315,
316,
320,
321,
322,
323,
327,
328,
329,
330,
332,
333,
334,
335,
338,
339,
340,
341,
342,
343,
346,
347,
348,
349,
353,
359,
365,
371,
379,
387,
395,
403,
414,
424,
436,
448,
460,
472,
485,
497,
498,
499,
500,
504,
505,
506,
507,
511,
520,
532,
543,
544,
546,
547,
550,
551,
555,
556,
558,
559,
562,
563,
567,
576,
585,
594,
595,
596,
597,
601,
602,
603,
604,
608,
617,
629,
640,
641,
643,
644,
647,
648,
650,
651,
654,
662,
670,
671,
673,
674,
677,
678,
680,
681,
684,
692,
700,
701,
703,
704,
707,
708,
712,
713,
715,
716,
719,
720,
724,
733,
742,
751,
752,
753,
754,
758,
759,
760,
761,
765,
774,
786,
797,
798,
800,
801,
804,
805,
809,
810,
812,
813,
816,
817,
821,
830,
839,
848,
849,
850,
851,
855,
856,
857,
858,
862,
871,
883,
894,
895,
897,
898,
901,
902,
904,
905,
908,
916,
924,
925,
927,
928,
931,
932,
934,
935,
938,
946,
954,
955,
957,
958,
961,
962,
966,
967,
969,
970,
973,
974,
978,
987,
996,
1005,
1006,
1007,
1008,
1012,
1022,
1031,
1032,
1033,
1034,
1038,
1048,
1057,
1058,
1061,
1062,
1064,
1065,
1068,
1069,
1071,
1072,
1075,
1076,
1077,
1078,
1079,
1080,
1082,
1083,
1085,
1091,
1097,
1098,
1101,
1102,
1103,
1104,
1105,
1106,
1108,
1109,
1111,
1117,
1123,
1124,
1125,
1126,
1130,
1131,
1135,
1136,
1139,
1140,
1142,
1143,
1147,
1148,
1151,
1152,
1154,
1155,
1156,
1157,
1158,
1159,
1160,
1161,
1162,
1163,
1164,
1165,
1167,
1170,
1174,
1180,
1186,
1192,
1198,
1206,
1214,
1222,
1230,
1241,
1251,
1262,
1272,
1273,
1274,
1275,
1279,
1280,
1284,
1285,
1288,
1289,
1291,
1292,
1296,
1297,
1300,
1301,
1303,
1304,
1305,
1306,
1307,
1308,
1309,
1310,
1311,
1312,
1313,
1314,
1316,
1319,
1323,
1329,
1335,
1341,
1347,
1355,
1363,
1371,
1379,
1390,
1400,
1413,
1425,
1437,
1449,
1461,
1473,
1485,
1497,
1513,
1528,
1542,
1559,
1575,
1592,
1608,
1624,
1639,
1653,
1669,
1684,
1698,
1712,
1725,
1737,
1753,
1768,
1782,
1799,
1815,
1832,
1848,
1864,
1879,
1893,
1909,
1924,
1938,
1952,
1965,
1977,
1996,
2014,
2031,
2047,
2064,
2080,
2095,
2109,
2122,
2134,
2145,
2155,
2164,
2172,
2179,
2185,
2190,
2194,
2197,
2199,
2200,
2200,
};
c_float linsys_solver_L_x[2200] = {
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)500000.00000000000000000000,
(c_float)88913.97050194613984785974,
(c_float)-500000.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-1.77827308550609597404,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)-500000.00000000000000000000,
(c_float)88913.97050194613984785974,
(c_float)500000.00000000000000000000,
(c_float)0.08891396961280643996,
(c_float)0.00000025310095577163,
(c_float)-0.00000025310095577163,
(c_float)0.00000025310095577163,
(c_float)-0.00000025310095577163,
(c_float)-0.49999999500000008590,
(c_float)-0.08891396961280643996,
(c_float)-0.49999999500000008590,
(c_float)-0.00000000839868925027,
(c_float)0.00000000839868925027,
(c_float)-0.00000000839868925027,
(c_float)0.00000000839868925027,
(c_float)-0.00022483429690181136,
(c_float)-5.62087746966410062299,
(c_float)0.99932421277765215439,
(c_float)5.62087735724655246372,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)1.70104551189338870643,
(c_float)0.29667570599448817026,
(c_float)-1.72765134256895058762,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)9.99937345408226718746,
(c_float)0.86779167418225255926,
(c_float)-9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)83.01113752136491541478,
(c_float)-83.01113752136491541478,
(c_float)100.00000000000000000000,
(c_float)-100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)1.70104551189338870643,
(c_float)0.29667570599448817026,
(c_float)-1.72765134256895058762,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)9.99937345408226718746,
(c_float)0.86779167418225255926,
(c_float)-9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)83.01113752136491541478,
(c_float)100.00000000000000000000,
(c_float)-100.00000000000000000000,
(c_float)-83.01113752136491541478,
(c_float)-0.00000100000000000000,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)-1.72765134256895058762,
(c_float)-0.00000100000000000000,
(c_float)1.70104551189338870643,
(c_float)0.29667570599448817026,
(c_float)-1.72765134256895058762,
(c_float)-0.00000100000000000000,
(c_float)-415055.68760682462016120553,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)-0.00000240931518549213,
(c_float)0.00000290239991576074,
(c_float)-0.00000290239991576074,
(c_float)-0.10000000000000000555,
(c_float)-9.99990000099998965766,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)9.99937345408226718746,
(c_float)0.86779167418225255926,
(c_float)-9.99990000099998965766,
(c_float)0.49972380798822735315,
(c_float)0.04336833322149731551,
(c_float)-0.00499755119941253882,
(c_float)-0.49975012244002942063,
(c_float)0.00301182849576144386,
(c_float)-0.00301182849576144386,
(c_float)0.08636947859454861820,
(c_float)-0.00002157082259167717,
(c_float)-0.68062799762033932005,
(c_float)-0.15900444020054996241,
(c_float)-0.00135950175511003564,
(c_float)0.00135950175511003564,
(c_float)0.00041995353857996658,
(c_float)-0.00057932779696479779,
(c_float)-0.02870222926923442913,
(c_float)-0.49611552489411114841,
(c_float)-0.00024131810357049503,
(c_float)0.00024131810357049503,
(c_float)0.00072678526561517805,
(c_float)-0.00000026117888669834,
(c_float)-0.00001083736726331657,
(c_float)1.20464881002089807538,
(c_float)-1.20464881002089807538,
(c_float)-0.99999268033593369243,
(c_float)-0.00000440784103886633,
(c_float)0.00000440784103886633,
(c_float)0.79291326602267664381,
(c_float)0.07467083966653552396,
(c_float)0.01038937687320654751,
(c_float)-0.01038937687320654751,
(c_float)-0.01292942991943504642,
(c_float)-0.86148429768067258383,
(c_float)0.00200872425740108605,
(c_float)-0.00200872425740108605,
(c_float)0.06792956852148668290,
(c_float)-0.01554037317999317110,
(c_float)0.01554037317999317110,
(c_float)0.01417582277700450699,
(c_float)-0.42602680534493903020,
(c_float)-0.24046875350773239122,
(c_float)-0.00413649269109931089,
(c_float)0.00413649269109931089,
(c_float)-0.00143638705438013030,
(c_float)0.00143638705438013030,
(c_float)0.00163572046074196152,
(c_float)0.00721897408554852439,
(c_float)-0.04595740879038931470,
(c_float)-0.65680083620601703576,
(c_float)-0.00029721766293356298,
(c_float)0.00029721766293356298,
(c_float)-0.02743126186835682934,
(c_float)-0.02812218398603424582,
(c_float)-0.62201643117465854615,
(c_float)0.15899026276826799808,
(c_float)0.02214453113475870621,
(c_float)0.00799064233871897740,
(c_float)0.19452548408949441039,
(c_float)0.13565592969398951317,
(c_float)2.19427139292989581421,
(c_float)2.26746166574380270120,
(c_float)-0.00799064233871897740,
(c_float)0.02737135322794713790,
(c_float)0.60540929039236957010,
(c_float)-0.15474540114649459888,
(c_float)-0.02155329700060855044,
(c_float)-0.00777730115413094013,
(c_float)0.20001206748963407112,
(c_float)0.13948209969671845898,
(c_float)2.25616072869585515548,
(c_float)2.33141533019015101402,
(c_float)0.00777730115413094013,
(c_float)0.00139379506962521182,
(c_float)-0.00022877407655285642,
(c_float)-0.00004018080088620058,
(c_float)-0.99995279347723065033,
(c_float)1.20460076001231608522,
(c_float)-1.20460076001231608522,
(c_float)-0.00001220443638820604,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000001,
(c_float)0.00001220443638820604,
(c_float)0.89436692756898927836,
(c_float)0.08278687438821914768,
(c_float)-0.01849492620302541773,
(c_float)0.01660078393955453230,
(c_float)-0.01660078393955453230,
(c_float)0.00098225621406022130,
(c_float)-0.94339595863275493848,
(c_float)0.00000000000000000033,
(c_float)0.00000000000000000033,
(c_float)0.00000000000000000262,
(c_float)0.00000000000000000786,
(c_float)-0.00098225621406022130,
(c_float)0.05650335314331271508,
(c_float)0.03930579970354779878,
(c_float)-0.04356499224335439596,
(c_float)0.04356499224335439596,
(c_float)-0.00387493693575350273,
(c_float)-0.29438230393417608166,
(c_float)-0.32583061711662680837,
(c_float)-0.00000000000000000099,
(c_float)-0.00000000000000000099,
(c_float)-0.00000000000000000692,
(c_float)-0.00000000000000002270,
(c_float)0.00387493693575350273,
(c_float)0.00337673530234294068,
(c_float)-0.00337711237045438531,
(c_float)0.00337711237045438531,
(c_float)-0.00023319971558094255,
(c_float)0.00813424588142395120,
(c_float)-0.73334104637942965077,
(c_float)-0.05681946803317799999,
(c_float)-0.00000000000000000008,
(c_float)-0.00000000000000000008,
(c_float)-0.00000000000000000108,
(c_float)-0.00000000000000000166,
(c_float)0.00023319971558094255,
(c_float)-0.10000000000000000555,
(c_float)1.50979329263400496863,
(c_float)-1.50979329263400496863,
(c_float)-0.00000100000000000000,
(c_float)2.47612545547994100659,
(c_float)0.29667570599448817026,
(c_float)-2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)-1.50979329263400496863,
(c_float)1.50979329263400496863,
(c_float)-0.00000100000000000000,
(c_float)-2.51485421031885136856,
(c_float)0.29667570599448817026,
(c_float)2.47612545547994100659,
(c_float)0.05885873178763046526,
(c_float)-0.05270022702805166742,
(c_float)-0.05270022702805166742,
(c_float)-0.05270022702805166742,
(c_float)-0.05270022702805166742,
(c_float)-0.49893242506651530599,
(c_float)-0.05977933352389850491,
(c_float)-0.49893242506651530599,
(c_float)-0.00008312080549299120,
(c_float)-0.00008312080549299120,
(c_float)-0.00008312080549299120,
(c_float)-0.00008312080549299120,
(c_float)-0.49543780458827424429,
(c_float)-0.04952893913784821994,
(c_float)-0.48970767378754592736,
(c_float)0.04782502393093367166,
(c_float)-0.00000100000000000000,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)-500000.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)500000.00000000000000000000,
(c_float)88913.97050194613984785974,
(c_float)-500000.00000000000000000000,
(c_float)0.08891396961280643996,
(c_float)0.00000025310095577163,
(c_float)0.00000025310095577163,
(c_float)-0.00000025310095577163,
(c_float)-0.49999999500000008590,
(c_float)-0.00000025310095577163,
(c_float)-0.00000000839868925027,
(c_float)-0.00000000839868925027,
(c_float)0.00000000839868925027,
(c_float)-0.00022483429690181136,
(c_float)-5.62087746966410062299,
(c_float)0.00000000839868925027,
(c_float)-0.10000000000000000555,
(c_float)-1.50979329263400496863,
(c_float)-0.10000000000000000555,
(c_float)1.50979329263400496863,
(c_float)-1.50979329263400496863,
(c_float)-0.00000100000000000000,
(c_float)-2.51485421031885136856,
(c_float)-0.00000100000000000000,
(c_float)2.47612545547994100659,
(c_float)0.29667570599448817026,
(c_float)-2.51485421031885136856,
(c_float)0.05885873178763046526,
(c_float)-0.05270022702805166742,
(c_float)-0.05270022702805166742,
(c_float)-0.05270022702805166742,
(c_float)-0.49893242506651530599,
(c_float)-0.05270022702805166742,
(c_float)-0.00008312080549299120,
(c_float)-0.00008312080549299120,
(c_float)-0.00008312080549299120,
(c_float)-0.49543780458827424429,
(c_float)-0.04952893913784821994,
(c_float)-0.00008312080549299120,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)1.70104551189338870643,
(c_float)0.29667570599448817026,
(c_float)-1.72765134256895058762,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-9.99937345408226718746,
(c_float)-0.86779167418225255926,
(c_float)-9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)83.01113752136491541478,
(c_float)-83.01113752136491541478,
(c_float)100.00000000000000000000,
(c_float)-100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)1.70104551189338870643,
(c_float)0.29667570599448817026,
(c_float)-1.72765134256895058762,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-9.99937345408226718746,
(c_float)-0.86779167418225255926,
(c_float)-9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)83.01113752136491541478,
(c_float)100.00000000000000000000,
(c_float)-100.00000000000000000000,
(c_float)-83.01113752136491541478,
(c_float)-0.00000100000000000000,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)-1.72765134256895058762,
(c_float)-0.00000100000000000000,
(c_float)1.70104551189338870643,
(c_float)0.29667570599448817026,
(c_float)-1.72765134256895058762,
(c_float)-0.00000100000000000000,
(c_float)-415055.68760682462016120553,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)-0.00000240931518549213,
(c_float)0.00000290239991576074,
(c_float)-0.00000290239991576074,
(c_float)-0.10000000000000000555,
(c_float)-9.99990000099998965766,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-9.99937345408226718746,
(c_float)-0.86779167418225255926,
(c_float)-9.99990000099998965766,
(c_float)-0.49972380798822735315,
(c_float)-0.04336833322149731551,
(c_float)-0.00499755119941253882,
(c_float)-0.49975012244002942063,
(c_float)0.00301182849576144386,
(c_float)-0.00301182849576144386,
(c_float)0.08636947859454861820,
(c_float)0.00002157082259167717,
(c_float)0.68062799762033932005,
(c_float)-0.15900444020054996241,
(c_float)0.00135950175511003564,
(c_float)-0.00135950175511003564,
(c_float)-0.00041995353857996658,
(c_float)0.00057932779696479779,
(c_float)-0.02870222926923442913,
(c_float)-0.49611552489411114841,
(c_float)0.00024131810357049503,
(c_float)-0.00024131810357049503,
(c_float)0.00072678526561517805,
(c_float)0.00000026117888669834,
(c_float)0.00001083736726331657,
(c_float)1.20464881002089807538,
(c_float)-1.20464881002089807538,
(c_float)-0.99999268033593369243,
(c_float)-0.00000440784103886633,
(c_float)0.00000440784103886633,
(c_float)-0.79291326602267664381,
(c_float)-0.07467083966653552396,
(c_float)0.01038937687320654751,
(c_float)-0.01038937687320654751,
(c_float)-0.01292942991943504642,
(c_float)-0.86148429768067258383,
(c_float)0.00200872425740108605,
(c_float)-0.00200872425740108605,
(c_float)0.06792956852148668290,
(c_float)0.01554037317999317110,
(c_float)-0.01554037317999317110,
(c_float)-0.01417582277700450699,
(c_float)0.42602680534493903020,
(c_float)-0.24046875350773239122,
(c_float)0.00413649269109931089,
(c_float)-0.00413649269109931089,
(c_float)0.00143638705438013030,
(c_float)-0.00143638705438013030,
(c_float)-0.00163572046074196152,
(c_float)-0.00721897408554852439,
(c_float)-0.04595740879038931470,
(c_float)-0.65680083620601703576,
(c_float)0.00029721766293356298,
(c_float)-0.00029721766293356298,
(c_float)-0.02743126186835682934,
(c_float)-0.02812218398603424582,
(c_float)-0.62201643117465854615,
(c_float)-0.15899026276826799808,
(c_float)-0.02214453113475870621,
(c_float)0.00799064233871897740,
(c_float)-0.00799064233871897740,
(c_float)-0.19452548408949441039,
(c_float)0.13565592969398951317,
(c_float)-2.19427139292989581421,
(c_float)2.26746166574380270120,
(c_float)0.02737135322794713790,
(c_float)0.60540929039236957010,
(c_float)0.15474540114649459888,
(c_float)0.02155329700060855044,
(c_float)-0.00777730115413094013,
(c_float)0.00777730115413094013,
(c_float)-0.20001206748963407112,
(c_float)0.13948209969671845898,
(c_float)-2.25616072869585515548,
(c_float)2.33141533019015101402,
(c_float)0.00139379506962521182,
(c_float)0.00022877407655285642,
(c_float)0.00004018080088620058,
(c_float)-0.00001220443638820604,
(c_float)0.00001220443638820604,
(c_float)-0.99995279347723065033,
(c_float)1.20460076001231608522,
(c_float)-1.20460076001231608522,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000001,
(c_float)-0.89436692756898927836,
(c_float)-0.08278687438821914768,
(c_float)0.00098225621406022130,
(c_float)-0.00098225621406022130,
(c_float)-0.94339595863275493848,
(c_float)-0.01849492620302541773,
(c_float)0.01660078393955453230,
(c_float)-0.01660078393955453230,
(c_float)-0.00000000000000000033,
(c_float)0.00000000000000000033,
(c_float)-0.00000000000000000262,
(c_float)0.00000000000000000786,
(c_float)0.05650335314331271508,
(c_float)0.00387493693575350273,
(c_float)-0.00387493693575350273,
(c_float)0.29438230393417608166,
(c_float)-0.32583061711662680837,
(c_float)-0.03930579970354779878,
(c_float)0.04356499224335439596,
(c_float)-0.04356499224335439596,
(c_float)-0.00000000000000000099,
(c_float)0.00000000000000000099,
(c_float)-0.00000000000000000692,
(c_float)0.00000000000000002270,
(c_float)0.00023319971558094255,
(c_float)-0.00023319971558094255,
(c_float)-0.00813424588142395120,
(c_float)-0.73334104637942965077,
(c_float)-0.05681946803317799999,
(c_float)-0.00337673530234294068,
(c_float)0.00337711237045438531,
(c_float)-0.00337711237045438531,
(c_float)-0.00000000000000000008,
(c_float)0.00000000000000000008,
(c_float)-0.00000000000000000108,
(c_float)0.00000000000000000166,
(c_float)0.10610068377058416711,
(c_float)-0.04539933803492164632,
(c_float)-0.00328225370523674721,
(c_float)-0.03609443662222060895,
(c_float)-0.01251909089715313485,
(c_float)0.01419376683259560186,
(c_float)-0.01419376683259560186,
(c_float)0.10909383486647726558,
(c_float)0.00000011821038821612,
(c_float)0.00099328278879940498,
(c_float)-0.99867966822287002060,
(c_float)1.03293396528202530149,
(c_float)0.10909383486647726558,
(c_float)0.05078797713642067835,
(c_float)0.00367183825476198253,
(c_float)0.04037863769095569738,
(c_float)0.01400503464971905067,
(c_float)-0.01587848494220441345,
(c_float)0.01587848494220441345,
(c_float)0.09862920841400037164,
(c_float)0.00000010687127306930,
(c_float)0.00089800395513128567,
(c_float)-0.90288314877310549278,
(c_float)0.93385166507705152927,
(c_float)0.09862920841400037164,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)83.01113752136491541478,
(c_float)-83.01113752136491541478,
(c_float)100.00000000000000000000,
(c_float)-100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)-83.01113752136491541478,
(c_float)100.00000000000000000000,
(c_float)-100.00000000000000000000,
(c_float)83.01113752136491541478,
(c_float)-0.60232880511402830948,
(c_float)0.60232880511402830948,
(c_float)-0.49999999274400014349,
(c_float)0.60232880511402830948,
(c_float)-0.60232880511402830948,
(c_float)0.00072559998947009305,
(c_float)-0.00002064511233918601,
(c_float)-0.00036374989729606207,
(c_float)-0.49999999274400014349,
(c_float)-0.99776137972362799466,
(c_float)-0.82825304701823243558,
(c_float)0.99776135076460259477,
(c_float)-0.99776135076460259477,
(c_float)-0.00000066261989008549,
(c_float)-0.00001652437056395644,
(c_float)-0.00020095658177428745,
(c_float)0.82825307105744883085,
(c_float)-0.00044775106980137830,
(c_float)0.00031224745657201374,
(c_float)-0.00505068715401275209,
(c_float)0.00521915363080817634,
(c_float)0.41459057894734363536,
(c_float)-0.49943970330561709980,
(c_float)0.49943970330561709980,
(c_float)0.00000033168119917162,
(c_float)0.00000827144359264912,
(c_float)0.00010059088328260846,
(c_float)-0.41459059098039613422,
(c_float)-0.20001206748963504256,
(c_float)0.13948209969671915287,
(c_float)-2.25616072869586625771,
(c_float)2.33141533019016211625,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)1.70104551189338870643,
(c_float)0.29667570599448817026,
(c_float)-1.72765134256895058762,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-9.99937345408226718746,
(c_float)-0.86779167418225255926,
(c_float)-9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)-1.77827308550609597404,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)-1.72765134256895058762,
(c_float)0.29667570599448817026,
(c_float)1.70104551189338870643,
(c_float)-0.10000000000000000555,
(c_float)-9.99990000099998965766,
(c_float)9.99990000099998965766,
(c_float)-0.86779167418225255926,
(c_float)-9.99937345408226718746,
(c_float)-0.49972380798822735315,
(c_float)-0.04336833322149731551,
(c_float)-0.49975012244002942063,
(c_float)-0.00499755119941253882,
(c_float)0.00301182849576144386,
(c_float)-0.00301182849576144386,
(c_float)-0.49975012244002942063,
(c_float)0.04336833322149731551,
(c_float)0.49972380798822735315,
(c_float)0.08636947859454861820,
(c_float)0.68062799762033932005,
(c_float)-0.15900444020054996241,
(c_float)0.00002157082259167717,
(c_float)0.00135950175511003564,
(c_float)-0.00135950175511003564,
(c_float)-0.67994804277808096327,
(c_float)0.03127429108728092128,
(c_float)0.52090779976489287595,
(c_float)0.00057932779696479779,
(c_float)-0.49611552489411114841,
(c_float)-0.02870222926923442913,
(c_float)-0.00041995353857996658,
(c_float)0.00024131810357049503,
(c_float)-0.00024131810357049503,
(c_float)-0.00057874904213516605,
(c_float)-0.49115430104422874713,
(c_float)0.02873685325534826654,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)83.01113752136491541478,
(c_float)-83.01113752136491541478,
(c_float)100.00000000000000000000,
(c_float)-100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)-83.01113752136491541478,
(c_float)100.00000000000000000000,
(c_float)-100.00000000000000000000,
(c_float)83.01113752136491541478,
(c_float)-0.60232880511402830948,
(c_float)0.60232880511402830948,
(c_float)-0.49999999274400014349,
(c_float)0.60232880511402830948,
(c_float)-0.60232880511402830948,
(c_float)-0.49999999274400014349,
(c_float)0.00072559998947009305,
(c_float)0.00002064511233918601,
(c_float)0.00036374989729606207,
(c_float)-0.99776137972362799466,
(c_float)-0.82825304701823243558,
(c_float)0.99776135076460259477,
(c_float)-0.99776135076460259477,
(c_float)0.82825307105744883085,
(c_float)-0.00000066261989008549,
(c_float)0.00044775106980137830,
(c_float)0.00001652437056395644,
(c_float)0.00031224745657201374,
(c_float)0.00505068715401275209,
(c_float)0.00020095658177428745,
(c_float)0.00521915363080817634,
(c_float)0.41459057894734363536,
(c_float)-0.49943970330561709980,
(c_float)0.49943970330561709980,
(c_float)-0.41459059098039613422,
(c_float)0.00000033168119917162,
(c_float)0.20001206748963504256,
(c_float)-0.00000827144359264912,
(c_float)0.13948209969671915287,
(c_float)2.25616072869586625771,
(c_float)-0.00010059088328260846,
(c_float)2.33141533019016211625,
(c_float)-0.10000000000000000555,
(c_float)1.50979329263400496863,
(c_float)-1.50979329263400496863,
(c_float)-0.00000100000000000000,
(c_float)2.47612545547994100659,
(c_float)0.29667570599448817026,
(c_float)-2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)-1.50979329263400496863,
(c_float)1.50979329263400496863,
(c_float)-0.00000100000000000000,
(c_float)-2.51485421031885136856,
(c_float)0.29667570599448817026,
(c_float)2.47612545547994100659,
(c_float)0.05885873178763046526,
(c_float)-0.49893242506651530599,
(c_float)-0.05270022702805166742,
(c_float)-0.05270022702805166742,
(c_float)-0.05270022702805166742,
(c_float)-0.05270022702805166742,
(c_float)-0.05977933352389850491,
(c_float)-0.49893242506651530599,
(c_float)-0.49543780458827424429,
(c_float)-0.04952893913784821994,
(c_float)-0.00008312080549299120,
(c_float)-0.00008312080549299120,
(c_float)-0.00008312080549299120,
(c_float)-0.00008312080549299120,
(c_float)-0.48970767378754592736,
(c_float)0.04782502393093367166,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)500000.00000000000000000000,
(c_float)88913.97050194613984785974,
(c_float)-500000.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-1.77827308550609597404,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)-500000.00000000000000000000,
(c_float)88913.97050194613984785974,
(c_float)500000.00000000000000000000,
(c_float)0.08891396961280643996,
(c_float)-0.49999999500000008590,
(c_float)-0.00000025310095577163,
(c_float)-0.00000025310095577163,
(c_float)0.00000025310095577163,
(c_float)0.00000025310095577163,
(c_float)-0.08891396961280643996,
(c_float)-0.49999999500000008590,
(c_float)-0.00022483429690181136,
(c_float)-5.62087746966410062299,
(c_float)0.00000000839868925027,
(c_float)0.00000000839868925027,
(c_float)-0.00000000839868925027,
(c_float)-0.00000000839868925027,
(c_float)0.99932421277765215439,
(c_float)5.62087735724655246372,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)1.70104551189338870643,
(c_float)0.29667570599448817026,
(c_float)-1.72765134256895058762,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)9.99937345408226718746,
(c_float)0.86779167418225255926,
(c_float)-9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)-1.77827308550609597404,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)-1.72765134256895058762,
(c_float)0.29667570599448817026,
(c_float)1.70104551189338870643,
(c_float)-0.10000000000000000555,
(c_float)-9.99990000099998965766,
(c_float)9.99990000099998965766,
(c_float)0.86779167418225255926,
(c_float)9.99937345408226718746,
(c_float)0.49972380798822735315,
(c_float)0.04336833322149731551,
(c_float)-0.49975012244002942063,
(c_float)-0.00499755119941253882,
(c_float)0.00301182849576144386,
(c_float)-0.00301182849576144386,
(c_float)-0.49975012244002942063,
(c_float)-0.04336833322149731551,
(c_float)-0.49972380798822735315,
(c_float)0.08636947859454861820,
(c_float)-0.68062799762033932005,
(c_float)-0.15900444020054996241,
(c_float)-0.00002157082259167717,
(c_float)-0.00135950175511003564,
(c_float)0.00135950175511003564,
(c_float)0.67994804277808096327,
(c_float)0.03127429108728092128,
(c_float)0.52090779976489287595,
(c_float)-0.00057932779696479779,
(c_float)-0.49611552489411114841,
(c_float)-0.02870222926923442913,
(c_float)0.00041995353857996658,
(c_float)-0.00024131810357049503,
(c_float)0.00024131810357049503,
(c_float)0.00057874904213516605,
(c_float)-0.49115430104422874713,
(c_float)0.02873685325534826654,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)83.01113752136491541478,
(c_float)-83.01113752136491541478,
(c_float)100.00000000000000000000,
(c_float)-100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)-83.01113752136491541478,
(c_float)100.00000000000000000000,
(c_float)-100.00000000000000000000,
(c_float)83.01113752136491541478,
(c_float)-0.60232880511402830948,
(c_float)0.60232880511402830948,
(c_float)-0.49999999274400014349,
(c_float)0.60232880511402830948,
(c_float)-0.60232880511402830948,
(c_float)0.00072559998947009305,
(c_float)0.00002064511233918601,
(c_float)0.00036374989729606207,
(c_float)-0.49999999274400014349,
(c_float)-0.99776137972362799466,
(c_float)-0.82825304701823243558,
(c_float)0.99776135076460259477,
(c_float)-0.99776135076460259477,
(c_float)-0.00000066261989008549,
(c_float)0.00001652437056395644,
(c_float)0.00020095658177428745,
(c_float)0.00044775106980137830,
(c_float)0.00505068715401275209,
(c_float)0.00031224745657201374,
(c_float)0.00521915363080817634,
(c_float)0.82825307105744883085,
(c_float)0.41459057894734363536,
(c_float)-0.49943970330561709980,
(c_float)0.49943970330561709980,
(c_float)0.00000033168119917162,
(c_float)-0.00000827144359264912,
(c_float)-0.00010059088328260846,
(c_float)0.20001206748963504256,
(c_float)2.25616072869586625771,
(c_float)0.13948209969671915287,
(c_float)2.33141533019016211625,
(c_float)-0.41459059098039613422,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)1.70104551189338870643,
(c_float)0.29667570599448817026,
(c_float)-1.72765134256895058762,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)9.99937345408226718746,
(c_float)0.86779167418225255926,
(c_float)-9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)-1.77827308550609597404,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)-1.72765134256895058762,
(c_float)0.29667570599448817026,
(c_float)1.70104551189338870643,
(c_float)-0.10000000000000000555,
(c_float)-9.99990000099998965766,
(c_float)9.99990000099998965766,
(c_float)0.86779167418225255926,
(c_float)9.99937345408226718746,
(c_float)0.49972380798822735315,
(c_float)0.04336833322149731551,
(c_float)-0.49975012244002942063,
(c_float)-0.00499755119941253882,
(c_float)0.00301182849576144386,
(c_float)-0.00301182849576144386,
(c_float)-0.49975012244002942063,
(c_float)-0.04336833322149731551,
(c_float)-0.49972380798822735315,
(c_float)0.08636947859454861820,
(c_float)-0.68062799762033932005,
(c_float)-0.15900444020054996241,
(c_float)-0.00002157082259167717,
(c_float)-0.00135950175511003564,
(c_float)0.00135950175511003564,
(c_float)0.67994804277808096327,
(c_float)0.03127429108728092128,
(c_float)0.52090779976489287595,
(c_float)-0.00057932779696479779,
(c_float)-0.02870222926923442913,
(c_float)-0.49611552489411114841,
(c_float)0.00041995353857996658,
(c_float)-0.00024131810357049503,
(c_float)0.00024131810357049503,
(c_float)0.00057874904213516605,
(c_float)-0.49115430104422874713,
(c_float)0.02873685325534826654,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)83.01113752136491541478,
(c_float)-83.01113752136491541478,
(c_float)100.00000000000000000000,
(c_float)-100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)-83.01113752136491541478,
(c_float)100.00000000000000000000,
(c_float)-100.00000000000000000000,
(c_float)83.01113752136491541478,
(c_float)-0.60232880511402830948,
(c_float)0.60232880511402830948,
(c_float)-0.49999999274400014349,
(c_float)0.60232880511402830948,
(c_float)-0.60232880511402830948,
(c_float)0.00072559998947009305,
(c_float)-0.00002064511233918601,
(c_float)-0.00036374989729606207,
(c_float)-0.49999999274400014349,
(c_float)-0.99776137972362799466,
(c_float)-0.82825304701823243558,
(c_float)0.99776135076460259477,
(c_float)-0.99776135076460259477,
(c_float)-0.00044775106980137830,
(c_float)-0.00505068715401275209,
(c_float)0.00031224745657201374,
(c_float)0.00521915363080817634,
(c_float)-0.00000066261989008549,
(c_float)-0.00001652437056395644,
(c_float)-0.00020095658177428745,
(c_float)0.82825307105744883085,
(c_float)0.41459057894734363536,
(c_float)-0.49943970330561709980,
(c_float)0.49943970330561709980,
(c_float)-0.20001206748963504256,
(c_float)-2.25616072869586625771,
(c_float)0.13948209969671915287,
(c_float)2.33141533019016211625,
(c_float)0.00000033168119917162,
(c_float)0.00000827144359264912,
(c_float)0.00010059088328260846,
(c_float)-0.41459059098039613422,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)500000.00000000000000000000,
(c_float)88913.97050194613984785974,
(c_float)-500000.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-1.77827308550609597404,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)-500000.00000000000000000000,
(c_float)88913.97050194613984785974,
(c_float)500000.00000000000000000000,
(c_float)0.08891396961280643996,
(c_float)-0.49999999500000008590,
(c_float)0.00000025310095577163,
(c_float)0.00000025310095577163,
(c_float)-0.00000025310095577163,
(c_float)-0.00000025310095577163,
(c_float)-0.08891396961280643996,
(c_float)-0.49999999500000008590,
(c_float)-0.00022483429690181136,
(c_float)-5.62087746966410062299,
(c_float)-0.00000000839868925027,
(c_float)-0.00000000839868925027,
(c_float)0.00000000839868925027,
(c_float)0.00000000839868925027,
(c_float)0.99932421277765215439,
(c_float)5.62087735724655246372,
(c_float)-0.10000000000000000555,
(c_float)1.50979329263400496863,
(c_float)-1.50979329263400496863,
(c_float)-0.00000100000000000000,
(c_float)2.47612545547994100659,
(c_float)0.29667570599448817026,
(c_float)-2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)-1.50979329263400496863,
(c_float)1.50979329263400496863,
(c_float)-0.00000100000000000000,
(c_float)-2.51485421031885136856,
(c_float)0.29667570599448817026,
(c_float)2.47612545547994100659,
(c_float)0.05885873178763046526,
(c_float)-0.49893242506651530599,
(c_float)-0.05270022702805166742,
(c_float)-0.05270022702805166742,
(c_float)-0.05270022702805166742,
(c_float)-0.05270022702805166742,
(c_float)-0.05977933352389850491,
(c_float)-0.49893242506651530599,
(c_float)-0.49543780458827424429,
(c_float)-0.04952893913784821994,
(c_float)-0.00008312080549299120,
(c_float)-0.00008312080549299120,
(c_float)-0.00008312080549299120,
(c_float)-0.00008312080549299120,
(c_float)-0.48970767378754592736,
(c_float)0.04782502393093367166,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)1.70104551189338870643,
(c_float)0.29667570599448817026,
(c_float)-1.72765134256895058762,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-9.99937345408226718746,
(c_float)-0.86779167418225255926,
(c_float)-9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)-1.77827308550609597404,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)-1.72765134256895058762,
(c_float)0.29667570599448817026,
(c_float)1.70104551189338870643,
(c_float)-0.10000000000000000555,
(c_float)-9.99990000099998965766,
(c_float)9.99990000099998965766,
(c_float)-0.86779167418225255926,
(c_float)-9.99937345408226718746,
(c_float)-0.49972380798822735315,
(c_float)-0.04336833322149731551,
(c_float)-0.49975012244002942063,
(c_float)-0.00499755119941253882,
(c_float)0.00301182849576144386,
(c_float)-0.00301182849576144386,
(c_float)-0.49975012244002942063,
(c_float)0.04336833322149731551,
(c_float)0.49972380798822735315,
(c_float)0.08636947859454861820,
(c_float)0.68062799762033932005,
(c_float)-0.15900444020054996241,
(c_float)0.00002157082259167717,
(c_float)0.00135950175511003564,
(c_float)-0.00135950175511003564,
(c_float)-0.67994804277808096327,
(c_float)0.03127429108728092128,
(c_float)0.52090779976489287595,
(c_float)0.00057932779696479779,
(c_float)-0.02870222926923442913,
(c_float)-0.49611552489411114841,
(c_float)-0.00041995353857996658,
(c_float)0.00024131810357049503,
(c_float)-0.00024131810357049503,
(c_float)-0.00057874904213516605,
(c_float)-0.49115430104422874713,
(c_float)0.02873685325534826654,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)100.00000000000000000000,
(c_float)-100.00000000000000000000,
(c_float)-83.01113752136491541478,
(c_float)83.01113752136491541478,
(c_float)-0.99887943558648017994,
(c_float)-0.82918118194732670645,
(c_float)0.00015629867724315888,
(c_float)0.00261250105214551852,
(c_float)0.00022412640510339769,
(c_float)0.00252817342375739927,
(c_float)-0.00060198553809375920,
(c_float)-0.00020102368503241731,
(c_float)-0.00000884709493829201,
(c_float)0.82918118194732670645,
(c_float)0.41482300892454415653,
(c_float)0.13948209969671537811,
(c_float)2.33141533019009905559,
(c_float)0.20001206748962963022,
(c_float)2.25616072869580541749,
(c_float)0.00030116150447920427,
(c_float)0.00010056818908311829,
(c_float)0.00000442602729348530,
(c_float)-0.41482300892454415653,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)100.00000000000000000000,
(c_float)-100.00000000000000000000,
(c_float)-83.01113752136491541478,
(c_float)83.01113752136491541478,
(c_float)-0.99887943558648017994,
(c_float)-0.82918118194732670645,
(c_float)-0.00060198553809375920,
(c_float)0.00020102368503241731,
(c_float)0.00000884709493829201,
(c_float)0.00015629867724315888,
(c_float)0.00261250105214551852,
(c_float)-0.00022412640510339769,
(c_float)-0.00252817342375739927,
(c_float)0.82918118194732670645,
(c_float)0.41482300892454415653,
(c_float)0.00030116150447920427,
(c_float)-0.00010056818908311829,
(c_float)-0.00000442602729348530,
(c_float)0.13948209969671537811,
(c_float)2.33141533019009905559,
(c_float)-0.20001206748962963022,
(c_float)-2.25616072869580541749,
(c_float)-0.41482300892454415653,
(c_float)-0.00000100000000000000,
(c_float)-2.51485421031885136856,
(c_float)0.29667570599448817026,
(c_float)2.47612545547994100659,
(c_float)-0.10000000000000000555,
(c_float)-1.50979329263400496863,
(c_float)1.50979329263400496863,
(c_float)-0.00000100000000000000,
(c_float)-500000.00000000000000000000,
(c_float)88913.97050194613984785974,
(c_float)500000.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-1.77827308550609597404,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)-500000.00000000000000000000,
(c_float)88913.97050194613984785974,
(c_float)500000.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)100.00000000000000000000,
(c_float)0.00994408012914218924,
(c_float)-0.00000100000000000000,
(c_float)100.00000000000000000000,
(c_float)0.00177827937447333560,
(c_float)0.00999999980000000448,
(c_float)-0.00000100000000000000,
(c_float)-1.77827308550609597404,
(c_float)1.77827308550609597404,
(c_float)0.00098873859167921091,
(c_float)-0.01247560016090967089,
(c_float)-0.01247560016090967089,
(c_float)0.01247560016090967089,
(c_float)0.01247560016090967089,
(c_float)-0.98873509495945977044,
(c_float)-0.00000050615752615586,
(c_float)-0.00000050615752615586,
(c_float)0.00000050615752615586,
(c_float)0.00000050615752615586,
(c_float)-0.17782793037490604582,
(c_float)-0.99999996000351865089,
(c_float)-0.00000100000000000000,
(c_float)-2.51485421031885136856,
(c_float)0.29667570599448817026,
(c_float)2.47612545547994100659,
(c_float)-0.10000000000000000555,
(c_float)100.00000000000000000000,
(c_float)0.00993420157827734109,
(c_float)-0.00000100000000000000,
(c_float)100.00000000000000000000,
(c_float)0.00117502113501049416,
(c_float)0.00980700368900603962,
(c_float)-0.10000000000000000555,
(c_float)-1.50979329263400496863,
(c_float)1.50979329263400496863,
(c_float)0.00075622683674709786,
(c_float)-0.01022795494093185764,
(c_float)-0.01022795494093185764,
(c_float)-0.01022795494093185764,
(c_float)-0.01022795494093185764,
(c_float)-0.98687936289805600065,
(c_float)-0.10318815837440543925,
(c_float)-0.10318815837440543925,
(c_float)-0.10318815837440543925,
(c_float)-0.10318815837440543925,
(c_float)-0.11660401422734864574,
(c_float)-0.97696443429363311051,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)-83.01113752136491541478,
(c_float)100.00000000000000000000,
(c_float)-100.00000000000000000000,
(c_float)83.01113752136491541478,
(c_float)-0.10000000000000000555,
(c_float)-9.99990000099998965766,
(c_float)9.99990000099998965766,
(c_float)9.99937345408226718746,
(c_float)0.86779167418225255926,
(c_float)-0.00000100000000000000,
(c_float)-1.72765134256895058762,
(c_float)1.70104551189338870643,
(c_float)0.29667570599448817026,
(c_float)-0.00000100000000000000,
(c_float)-1.77827308550609597404,
(c_float)1.77827308550609597404,
(c_float)-0.10000000000000000555,
(c_float)-9.99990000099998965766,
(c_float)0.86779167418225255926,
(c_float)9.99937345408226718746,
(c_float)9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)-1.72765134256895058762,
(c_float)0.29667570599448817026,
(c_float)1.70104551189338870643,
(c_float)-0.00000100000000000000,
(c_float)-1.77827308550609597404,
(c_float)1.77827308550609597404,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)-0.00000100000000000000,
(c_float)100.00000000000000000000,
(c_float)-0.10000000000000000555,
(c_float)100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)100.00000000000000000000,
(c_float)0.00994408012914218924,
(c_float)-0.00000100000000000000,
(c_float)100.00000000000000000000,
(c_float)0.00117502113501049416,
(c_float)0.00673720290439491082,
(c_float)0.00086693341001964049,
(c_float)0.00998948386405791068,
(c_float)0.00999000989020969003,
(c_float)0.00830111358611422025,
(c_float)0.00002845246999157337,
(c_float)0.00050130911700634322,
(c_float)0.00099999998000000053,
(c_float)0.00139831451053373801,
(c_float)0.02463715145927342781,
(c_float)0.04914562718040228761,
(c_float)49.14562816331482508758,
(c_float)59.20365583553895305613,
(c_float)-59.20365583553895305613,
(c_float)0.00092442555381357333,
(c_float)0.00048298648658195976,
(c_float)-0.98879135405116513891,
(c_float)-0.00064542745367997549,
(c_float)-0.00028503318323366871,
(c_float)0.00028503318323366871,
(c_float)0.00826993777363479662,
(c_float)-0.16688363644464254687,
(c_float)-0.96463900915354849896,
(c_float)-0.01688144702542848652,
(c_float)-0.00372435763395965623,
(c_float)0.00372435763395965623,
(c_float)-0.08635453775333701287,
(c_float)-0.99699031993390851980,
(c_float)-0.99800736427907354997,
(c_float)-0.00405461104420909262,
(c_float)0.00111358956936138996,
(c_float)-0.00111358956936138996,
(c_float)0.00593532026039035598,
(c_float)0.00235072188738055842,
(c_float)-0.00602973643744201399,
(c_float)0.00048932368883550302,
(c_float)-0.00048932368883550302,
(c_float)-0.00016367467062421970,
(c_float)0.00016367467062421970,
(c_float)-0.98244236284412711502,
(c_float)0.02346073372849344710,
(c_float)-0.08758488807214558081,
(c_float)0.01597653990140427901,
(c_float)-0.01597653990140427901,
(c_float)0.00537905201412179190,
(c_float)-0.00537905201412179190,
(c_float)-0.92794289649746963811,
(c_float)-0.15345936409919880283,
(c_float)-0.01375318455439787819,
(c_float)0.00596183551148400919,
(c_float)-0.00596183551148400919,
(c_float)0.00109456811656626790,
(c_float)-0.00109456811656626790,
(c_float)-0.99708618436171114752,
(c_float)-0.99430006439133467033,
(c_float)-0.08565829102688239660,
(c_float)-0.75661702124776197831,
(c_float)0.75661702124776197831,
(c_float)0.44792447989810674214,
(c_float)-0.44792447989810674214,
(c_float)-0.62805680139003738915,
(c_float)0.00125350225823292710,
(c_float)0.00218285561203101705,
(c_float)0.00035968740030870982,
(c_float)-0.99699269863923556390,
(c_float)0.99680437630538554217,
(c_float)-0.99680437630538554217,
(c_float)0.00060149656273087273,
(c_float)0.00678495522951019087,
(c_float)0.00041946470822021379,
(c_float)0.00701126849498776711,
(c_float)0.82764349026425509948,
(c_float)0.00119132038741940954,
(c_float)0.00276345572375055835,
(c_float)0.00055704542281831805,
(c_float)-0.49915273950908906064,
(c_float)0.49915273950908906064,
(c_float)0.20001206748963501481,
(c_float)2.25616072869586581362,
(c_float)0.13948209969671915287,
(c_float)2.33141533019016167216,
(c_float)-0.41444492552638567684,
(c_float)-0.00059655720736045872,
(c_float)-0.00138380862665833009,
(c_float)-0.00027894214295220830,
(c_float)-0.20111436634034929316,
(c_float)0.12849669543570160779,
(c_float)0.12849669543570160779,
(c_float)0.32134783337644184931,
(c_float)1.21040213221948289579,
(c_float)0.21970667455514222843,
(c_float)1.22201484948053140656,
(c_float)0.21194192404646564065,
(c_float)-0.55916407299057024893,
(c_float)-1.08753230105315945586,
(c_float)-0.19183445579070673714,
(c_float)0.16084491950001075788,
(c_float)0.16084491950001075788,
(c_float)0.40224510222365439382,
(c_float)1.51511315415026581732,
(c_float)0.27501642950903182339,
(c_float)1.52964929896469747916,
(c_float)-0.17645440766163433821,
(c_float)0.46553774449828266846,
(c_float)0.90543609462164620361,
(c_float)0.15971373015477546842,
(c_float)-0.00000100000000000000,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)-83.01113752136491541478,
(c_float)100.00000000000000000000,
(c_float)-100.00000000000000000000,
(c_float)83.01113752136491541478,
(c_float)-0.10000000000000000555,
(c_float)-9.99990000099998965766,
(c_float)9.99990000099998965766,
(c_float)-9.99937345408226718746,
(c_float)-0.86779167418225255926,
(c_float)-0.00000100000000000000,
(c_float)-1.72765134256895058762,
(c_float)1.70104551189338870643,
(c_float)0.29667570599448817026,
(c_float)-0.00000100000000000000,
(c_float)-1.77827308550609597404,
(c_float)1.77827308550609597404,
(c_float)-0.10000000000000000555,
(c_float)-9.99990000099998965766,
(c_float)-0.86779167418225255926,
(c_float)-9.99937345408226718746,
(c_float)9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)-1.72765134256895058762,
(c_float)0.29667570599448817026,
(c_float)1.70104551189338870643,
(c_float)-0.00000100000000000000,
(c_float)-1.77827308550609597404,
(c_float)1.77827308550609597404,
(c_float)-0.09996364032263393984,
(c_float)-0.09996364032263393984,
(c_float)-0.00000100000000000000,
(c_float)100.00000000000000000000,
(c_float)-0.10000000000000000555,
(c_float)100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)100.00000000000000000000,
(c_float)0.00994408012914218924,
(c_float)-0.00000100000000000000,
(c_float)100.00000000000000000000,
(c_float)0.00117502113501049416,
(c_float)0.00673720290439491082,
(c_float)-0.00086693341001964049,
(c_float)-0.00998948386405791068,
(c_float)0.00999000989020969003,
(c_float)0.00830111358611422025,
(c_float)-0.00002845246999157337,
(c_float)-0.00050130911700634322,
(c_float)0.00099999998000000053,
(c_float)-0.00139831451053373801,
(c_float)-0.02463715145927342781,
(c_float)0.04914562718040228761,
(c_float)49.14562816331482508758,
(c_float)59.20365583553895305613,
(c_float)-59.20365583553895305613,
(c_float)0.00092442555381357333,
(c_float)-0.00048298648658195976,
(c_float)-0.98879135405116513891,
(c_float)0.00064542745367997549,
(c_float)0.00028503318323366871,
(c_float)-0.00028503318323366871,
(c_float)-0.00826993777363479662,
(c_float)-0.16688363644464254687,
(c_float)-0.96463900915354849896,
(c_float)0.01688144702542848652,
(c_float)0.00372435763395965623,
(c_float)-0.00372435763395965623,
(c_float)0.08635453775333701287,
(c_float)0.99699031993390851980,
(c_float)-0.99800736427907354997,
(c_float)-0.00405461104420909262,
(c_float)0.00111358956936138996,
(c_float)-0.00111358956936138996,
(c_float)0.00593532026039035598,
(c_float)-0.00235072188738055842,
(c_float)0.00602973643744201399,
(c_float)-0.00048932368883550302,
(c_float)0.00048932368883550302,
(c_float)0.00016367467062421970,
(c_float)-0.00016367467062421970,
(c_float)-0.98244236284412711502,
(c_float)-0.02346073372849344710,
(c_float)0.08758488807214558081,
(c_float)-0.01597653990140427901,
(c_float)0.01597653990140427901,
(c_float)-0.00537905201412179190,
(c_float)0.00537905201412179190,
(c_float)-0.92794289649746963811,
(c_float)-0.15345936409919880283,
(c_float)-0.01375318455439787819,
(c_float)0.00596183551148400919,
(c_float)-0.00596183551148400919,
(c_float)0.00109456811656626790,
(c_float)-0.00109456811656626790,
(c_float)-0.99708618436171114752,
(c_float)0.99430006439133467033,
(c_float)0.08565829102688239660,
(c_float)-0.75661702124776197831,
(c_float)0.75661702124776197831,
(c_float)0.44792447989810674214,
(c_float)-0.44792447989810674214,
(c_float)-0.62805680139003738915,
(c_float)0.00125350225823292710,
(c_float)-0.00218285561203101705,
(c_float)-0.00035968740030870982,
(c_float)-0.99699269863923556390,
(c_float)0.99680437630538554217,
(c_float)-0.99680437630538554217,
(c_float)-0.00060149656273087273,
(c_float)-0.00678495522951019087,
(c_float)0.00041946470822021379,
(c_float)0.00701126849498776711,
(c_float)0.82764349026425509948,
(c_float)0.00119132038741940954,
(c_float)-0.00276345572375055835,
(c_float)-0.00055704542281831805,
(c_float)-0.49915273950908906064,
(c_float)0.49915273950908906064,
(c_float)-0.20001206748963501481,
(c_float)-2.25616072869586581362,
(c_float)0.13948209969671915287,
(c_float)2.33141533019016167216,
(c_float)-0.41444492552638567684,
(c_float)-0.00059655720736045872,
(c_float)0.00138380862665833009,
(c_float)0.00027894214295220830,
(c_float)-0.25290452980381733505,
(c_float)-0.44303551834884491445,
(c_float)-1.66875603430695251816,
(c_float)0.15545517764681748840,
(c_float)0.86464617380282959846,
(c_float)0.22108052666311109324,
(c_float)0.00000000000000000000,
(c_float)-0.58327434887655060791,
(c_float)1.13442498439933858734,
(c_float)0.20010605598278277673,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.59301058033787812374,
(c_float)-2.23365834873643631653,
(c_float)0.20807940062331153896,
(c_float)1.15734361710920352628,
(c_float)-0.17645440766163433821,
(c_float)0.00000000000000000000,
(c_float)0.46553774449828266846,
(c_float)-0.90543609462164631463,
(c_float)-0.15971373015477549617,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.40672648961595264661,
(c_float)-0.00000000000000098659,
(c_float)-0.00000000000000547599,
(c_float)0.00000000000000006461,
(c_float)-0.00000000000000006286,
(c_float)0.00000000000000039463,
(c_float)-0.00000000000000076831,
(c_float)-0.00000000000000013446,
(c_float)-0.89500304718322376551,
(c_float)-0.00000000000000039813,
(c_float)-0.00000000000000076831,
(c_float)-0.00000000000000013620,
(c_float)-0.00000000000000000001,
(c_float)-0.00000000000000000007,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000002,
(c_float)-0.00000000000000000003,
(c_float)-0.00000000000000000001,
(c_float)-0.17782498415003750170,
(c_float)-0.99999150682062298845,
(c_float)-0.00000000000000000002,
(c_float)-0.00000000000000000003,
(c_float)-0.00000000000000000001,
(c_float)0.22806666303714528965,
(c_float)-0.00000000000000005738,
(c_float)-0.00000000000000005793,
(c_float)-0.00000000000000034539,
(c_float)0.00000000000000067329,
(c_float)0.00000000000000011805,
(c_float)-0.95128709752812568023,
(c_float)-0.00000000000000184486,
(c_float)-0.00000000000000415821,
(c_float)-0.00000000000000034539,
(c_float)-0.00000000000000066892,
(c_float)-0.00000000000000011805,
(c_float)-0.00000000000000023385,
(c_float)-0.00000000000000023267,
(c_float)-0.00000000000000148971,
(c_float)0.00000000000000289377,
(c_float)0.00000000000000051085,
(c_float)0.00783445061715647671,
(c_float)-0.40704937417065711891,
(c_float)-0.00000000000000255599,
(c_float)-0.00000000000000574139,
(c_float)-0.00000000000000148971,
(c_float)-0.00000000000000291685,
(c_float)-0.00000000000000051199,
(c_float)0.00000000000000000000,
(c_float)2.26892849103989169279,
(c_float)-2.87887143426446101557,
(c_float)-0.38395948711563476774,
(c_float)-0.00000000000001181007,
(c_float)-0.00000000000020759243,
(c_float)0.00000000000001291523,
(c_float)0.00000000000018869259,
(c_float)-0.62249669959634645888,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000001179587,
(c_float)-0.00000000000020763924,
(c_float)-0.00000000000001296122,
(c_float)-0.00000000000018866985,
(c_float)0.00000000000000000000,
(c_float)2.26892849103989169279,
(c_float)2.87887143426446057148,
(c_float)0.38395948711563476774,
(c_float)-0.62249669959634645888,
(c_float)0.05880153944014404610,
(c_float)-0.04922160534495790413,
(c_float)-0.00000000000000102308,
(c_float)-0.00000000000001593573,
(c_float)0.00000000000000195485,
(c_float)0.00000000000001699368,
(c_float)-0.09088645767194737657,
(c_float)0.01128855682625602361,
(c_float)-0.01128855682625602361,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.77278646870355582887,
(c_float)0.13471472409041887475,
(c_float)1.15778307464994600906,
(c_float)0.06315236178213096130,
(c_float)0.00000000000000149811,
(c_float)0.00000000000002222913,
(c_float)-0.00000000000000330667,
(c_float)-0.00000000000002522731,
(c_float)0.06686314256784589571,
(c_float)0.00011668179832308271,
(c_float)-0.00011668179832308271,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.33726865080569579103,
(c_float)-0.01460467312222935195,
(c_float)0.22671128285900832289,
(c_float)0.00000000000000014132,
(c_float)0.00000000000000175668,
(c_float)-0.00000000000000045677,
(c_float)-0.00000000000000250623,
(c_float)-0.00335108045023735672,
(c_float)0.00163848908746246622,
(c_float)-0.00163848908746246622,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.03708441505310124814,
(c_float)-0.90007586217603907652,
(c_float)0.14591927937934379145,
(c_float)0.43057919981596848302,
(c_float)-0.00000000000000299717,
(c_float)-0.00000000000000675651,
(c_float)-0.00000000000000084413,
(c_float)-0.00014121430315747167,
(c_float)-0.00014121430315752409,
(c_float)-0.00000000000000237567,
(c_float)-0.00000000000000335132,
(c_float)-0.00000000000000048935,
(c_float)-0.00000000000000039494,
(c_float)-0.00014121430315749788,
(c_float)-0.00014121430315749788,
(c_float)-0.83196652744923971223,
(c_float)0.08125014414672539209,
(c_float)-0.00000000000000065066,
(c_float)0.00000000000000050790,
(c_float)0.00000000000000193691,
(c_float)-0.00000000000000103925,
(c_float)-0.00000000000000233257,
(c_float)-0.00000000000000224481,
(c_float)-0.02616160223482365266,
(c_float)-0.02616160223482379490,
(c_float)-0.00000000000000642023,
(c_float)-0.00000000000000863059,
(c_float)-0.00000000000000120887,
(c_float)-0.00000000000000106075,
(c_float)-0.02616160223482372552,
(c_float)-0.02616160223482372552,
(c_float)0.01933622779824007223,
(c_float)-0.25254773888784981528,
(c_float)-0.00000000000000191489,
(c_float)0.00000000000000128778,
(c_float)0.00000000000000534516,
(c_float)-3.48755420433881724307,
(c_float)0.00000000000000062135,
(c_float)-0.00000000626459447103,
(c_float)-0.00000000626459443406,
(c_float)-0.00000000000000169879,
(c_float)-0.00000000000000278988,
(c_float)-0.00000000000000045013,
(c_float)-0.00000000000000027459,
(c_float)0.00000000626459433850,
(c_float)0.00000000626459433850,
(c_float)0.74539736810344070506,
(c_float)4.19262050789119111727,
(c_float)-0.00000000000000088747,
(c_float)-0.00000000000000045817,
(c_float)0.00000000000000033011,
(c_float)-0.00000000000000043460,
(c_float)-0.00000000000000128091,
(c_float)0.00000000000000019506,
(c_float)0.00139767200112634053,
(c_float)0.00139767200112635246,
(c_float)-0.00000000000000054521,
(c_float)-0.00000000000000081528,
(c_float)-0.00000000000000012378,
(c_float)-0.00000000000000008906,
(c_float)-0.00139767200112636374,
(c_float)-0.00139767200112636374,
(c_float)-0.26413208841337165733,
(c_float)-0.86261069484576669009,
(c_float)-0.00000000000000013458,
(c_float)-0.00000000000000006933,
(c_float)0.00000000000000013407,
(c_float)-0.00000000000000012470,
(c_float)-0.00000000000000043264,
(c_float)-0.61108155807134501014,
(c_float)0.61108155807135300375,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000335175,
(c_float)0.00000000000000335175,
(c_float)-0.00000000000003576787,
(c_float)-0.00000000000001577809,
(c_float)-0.00000000000000086655,
(c_float)-0.00000000000006460574,
(c_float)2.47544977144614275488,
(c_float)-0.37655763132260894333,
(c_float)-3.36844071638322484219,
(c_float)-0.48225625712354192043,
(c_float)-0.05907373890659649990,
(c_float)0.00000000000000492488,
(c_float)0.00000000000000655399,
(c_float)0.00000000000000091028,
(c_float)0.00000000000000081451,
(c_float)0.10899520099326244493,
(c_float)0.10899520099326244493,
(c_float)-0.35797197752512760482,
(c_float)-1.50984753719669639871,
(c_float)0.08543539389534536566,
(c_float)1.04651380658860304607,
(c_float)0.03735078735205399347,
(c_float)-0.07459080066024684208,
(c_float)-0.39647582271620918082,
(c_float)0.14676030558502586865,
(c_float)0.00000000000000523408,
(c_float)0.00000000000000696547,
(c_float)0.00000000000000096743,
(c_float)0.00000000000000086565,
(c_float)0.11583819636047060220,
(c_float)0.11583819636047060220,
(c_float)-0.38044636686952687521,
(c_float)-1.60463959783864540398,
(c_float)0.09079924477404720251,
(c_float)1.11221659960110885734,
(c_float)-0.03526740960513558010,
(c_float)0.07043022399673037892,
(c_float)0.37436092327767767074,
(c_float)-0.13857420894652905385,
(c_float)-0.05880153944014407386,
(c_float)0.04922160534495790413,
(c_float)-0.09088645767194737657,
(c_float)0.01128855682625567146,
(c_float)-0.01128855682625637576,
(c_float)-0.77278646870355582887,
(c_float)-0.13471472409041887475,
(c_float)-1.15778307464994600906,
(c_float)-0.00000000000000511576,
(c_float)-0.00000000000001352934,
(c_float)-0.00000000000000048307,
(c_float)-0.00000000000000300296,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.06315236178213097518,
(c_float)-0.06686314256784589571,
(c_float)-0.00011668179832358040,
(c_float)0.00011668179832258427,
(c_float)0.33726865080569579103,
(c_float)-0.01460467312222935715,
(c_float)0.22671128285900826738,
(c_float)-0.00000000000000764400,
(c_float)-0.00000000000001950660,
(c_float)-0.00000000000000073083,
(c_float)-0.00000000000000419418,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00335108045023735888,
(c_float)-0.00163848908746250721,
(c_float)0.00163848908746242502,
(c_float)0.03708441505310124120,
(c_float)-0.90007586217603907652,
(c_float)0.14591927937934379145,
(c_float)-0.00000000000000077284,
(c_float)-0.00000000000000174255,
(c_float)-0.00000000000000007810,
(c_float)-0.00000000000000033068,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.61108155807134578730,
(c_float)0.61108155807135233761,
(c_float)2.47544977144614275488,
(c_float)0.37655763132260899884,
(c_float)3.36844071638322573037,
(c_float)0.00000000000003833257,
(c_float)0.00000000000002656978,
(c_float)-0.00000000000000147961,
(c_float)-0.00000000000007205281,
(c_float)-0.48225625712354192043,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.08650986777038405240,
(c_float)0.03831838854667293792,
(c_float)0.07652313336172651692,
(c_float)0.40674683724882271241,
(c_float)0.45232763732028574699,
(c_float)1.90781908107904385474,
(c_float)0.06734253786561139599,
(c_float)0.82489109529249760033,
(c_float)0.15056224543888016565,
(c_float)-0.00000000000000208280,
(c_float)-0.00000000000000013676,
(c_float)0.00000000000000175514,
(c_float)0.00000000000000041314,
(c_float)-0.03526740960513517070,
(c_float)-0.07043022399673044831,
(c_float)-0.37436092327767755972,
(c_float)0.49516422932368631127,
(c_float)2.08849446071468936026,
(c_float)0.07372004960934590290,
(c_float)0.90301040612143368413,
(c_float)-0.13857420894652916488,
(c_float)-0.00000000000000228005,
(c_float)-0.00000000000000014972,
(c_float)0.00000000000000192135,
(c_float)0.00000000000000045227,
(c_float)0.08075849035787306673,
(c_float)0.11802392584045570678,
(c_float)0.00000000000000200375,
(c_float)-0.00000000000000208185,
(c_float)-0.00000000000000028505,
(c_float)-0.00000000000000781102,
(c_float)-0.08693522264928277288,
(c_float)0.01044526606226019519,
(c_float)-0.01044526606226019519,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.71505688452119953169,
(c_float)-0.12465110972870824257,
(c_float)-1.07129303091892547073,
(c_float)0.38513067358094604797,
(c_float)-0.00000000000000120501,
(c_float)-0.00000000000000457881,
(c_float)-0.00000000000000023703,
(c_float)-0.00000000000000276679,
(c_float)0.00407350530613060594,
(c_float)-0.00215375980864734992,
(c_float)0.00215375980864734992,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.12103529069284618691,
(c_float)-0.79263402779225944350,
(c_float)0.22749872274101437530,
(c_float)0.00000000000000081753,
(c_float)-0.00000000000000554321,
(c_float)-0.00000000000000043009,
(c_float)-0.00000000000000905489,
(c_float)-0.03574183530620882177,
(c_float)-0.00114611663102284246,
(c_float)0.00114611663102284246,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.30771556090027035735,
(c_float)0.11430816126157838930,
(c_float)0.26381109015249809557,
(c_float)-2.28748490118236524893,
(c_float)-0.00000000000000049078,
(c_float)-0.00000000000000015218,
(c_float)-0.00000000000000003138,
(c_float)0.00000000467927212705,
(c_float)0.00000000467927215629,
(c_float)-0.00000000000000140195,
(c_float)-0.00000000000000029071,
(c_float)0.00000000000000078509,
(c_float)0.00000000000000018256,
(c_float)-0.00000000467927214167,
(c_float)-0.00000000467927214167,
(c_float)0.00000000000000081218,
(c_float)0.55676663465022824262,
(c_float)-0.00000000000000023534,
(c_float)3.13163328773676408190,
(c_float)0.00000000000000138865,
(c_float)-0.00000000000000004812,
(c_float)-0.00000000000000001443,
(c_float)0.00000000000000001970,
(c_float)-0.00121373962539389002,
(c_float)-0.00121373962539388829,
(c_float)-0.00000000000000005390,
(c_float)-0.00000000000000007111,
(c_float)-0.00000000000000012983,
(c_float)-0.00000000000000001869,
(c_float)0.00121373962539388916,
(c_float)0.00121373962539388916,
(c_float)0.00000000000000008055,
(c_float)-0.24988679241067224268,
(c_float)-0.00000000000000004110,
(c_float)-0.86447803461199845820,
(c_float)0.00000000000000010304,
(c_float)0.43499518230460898494,
(c_float)0.00000000000000017769,
(c_float)-0.00013022033917246185,
(c_float)-0.00013022033917247142,
(c_float)-0.00000000000000065556,
(c_float)0.00000000000000031655,
(c_float)0.00000000000000153040,
(c_float)0.00000000000000028233,
(c_float)-0.00013022033917245405,
(c_float)-0.00013022033917245405,
(c_float)-0.00000000000000002380,
(c_float)-0.00000000000000039207,
(c_float)-0.00000000000000013692,
(c_float)-0.76719539708199668837,
(c_float)0.00000000000000060312,
(c_float)-0.00000000000000029031,
(c_float)0.07492457273819787267,
(c_float)0.00000000000000063155,
(c_float)-0.02608674896411046149,
(c_float)-0.02608674896411053087,
(c_float)-0.00000000000000342916,
(c_float)0.00000000000000084742,
(c_float)0.00000000000000599871,
(c_float)0.00000000000000112092,
(c_float)-0.02608674896411049618,
(c_float)-0.02608674896411049618,
(c_float)-0.00000000000000073002,
(c_float)0.00000000000000001201,
(c_float)-0.00000000000000026641,
(c_float)0.01993389993237925858,
(c_float)-0.00000000000000000563,
(c_float)-0.00000000000000212764,
(c_float)-0.25188998595264505687,
(c_float)-0.66282083337676478418,
(c_float)0.66282083337676245272,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000108720,
(c_float)-0.00000000000000108720,
(c_float)-0.52519156369013109131,
(c_float)2.79940255277734406647,
(c_float)-0.00000000000001701588,
(c_float)0.19096944830122358772,
(c_float)-0.00000000000000742013,
(c_float)-0.00000000000013330618,
(c_float)3.58654369834011266249,
(c_float)-0.00000000000011862957,
(c_float)-0.04451976291474893377,
(c_float)0.00000000000000295514,
(c_float)-0.00000000000000066109,
(c_float)-0.00000000000000498583,
(c_float)-0.00000000000000093622,
(c_float)0.11081818981392166368,
(c_float)0.11081818981392166368,
(c_float)0.13610118211187829940,
(c_float)0.13837460803017156974,
(c_float)0.35197823098377212236,
(c_float)0.01816422011880428761,
(c_float)0.08623579513219098436,
(c_float)1.53542312035474548537,
(c_float)0.47320994682577643964,
(c_float)1.06262440842501182381,
(c_float)0.00000000000000309283,
(c_float)-0.00000000000000069190,
(c_float)-0.00000000000000521814,
(c_float)-0.00000000000000097984,
(c_float)0.11598166609073952416,
(c_float)0.11598166609073952416,
(c_float)-0.13030024604999931870,
(c_float)-0.13247677348299327638,
(c_float)0.36837834768567911947,
(c_float)-0.01739002052782222946,
(c_float)0.09025387630754705404,
(c_float)1.60696481283460856915,
(c_float)-0.45304068302668420998,
(c_float)1.11213645995099330577,
(c_float)-0.08075849035787306673,
(c_float)-0.11802392584045567903,
(c_float)-0.08693522264928277288,
(c_float)0.01044526606226004427,
(c_float)-0.01044526606226034611,
(c_float)-0.71505688452119953169,
(c_float)0.12465110972870824257,
(c_float)1.07129303091892547073,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000063876,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000002727,
(c_float)0.00000000000000476158,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000146116,
(c_float)0.38513067358094604797,
(c_float)-0.00407350530613060941,
(c_float)0.00215375980864741410,
(c_float)-0.00215375980864728573,
(c_float)-0.12103529069284618691,
(c_float)-0.79263402779225944350,
(c_float)0.22749872274101437530,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000120205,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000011766,
(c_float)-0.00000000000000275098,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000051710,
(c_float)0.03574183530620882870,
(c_float)0.00114611663102302569,
(c_float)-0.00114611663102265966,
(c_float)-0.30771556090027046837,
(c_float)0.11430816126157843093,
(c_float)0.26381109015249820660,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000153932,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000011194,
(c_float)-0.00000000000000645988,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000169600,
(c_float)-0.66282083337676445112,
(c_float)0.66282083337676278578,
(c_float)2.79940255277734406647,
(c_float)-0.19096944830122350445,
(c_float)-3.58654369834011266249,
(c_float)-0.52519156369013109131,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000001787454,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000722556,
(c_float)0.00000000000013699905,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000011623126,
(c_float)-0.07207834795726358879,
(c_float)0.14202548045836008161,
(c_float)-0.01864346447841125007,
(c_float)-0.48569510701668544561,
(c_float)0.13969207252370788308,
(c_float)-0.00000000000000013321,
(c_float)0.00000000000000178392,
(c_float)-0.44506500702909984613,
(c_float)0.00000000000000022227,
(c_float)0.06797972295024044820,
(c_float)-1.94149251771435960023,
(c_float)0.00000000000000325507,
(c_float)0.83766738364461235111,
(c_float)-0.13247677348299374822,
(c_float)0.01739002052782225721,
(c_float)0.45304068302668493162,
(c_float)-0.13030024604999920768,
(c_float)-0.00000000000000014356,
(c_float)0.00000000000000192249,
(c_float)-0.47963640685540998021,
(c_float)0.00000000000000023953,
(c_float)0.07326019691489696339,
(c_float)-2.09230220400662281222,
(c_float)0.00000000000000350791,
(c_float)0.90273503350274941770,
(c_float)-0.12966786579209396701,
(c_float)-0.82091967349375971619,
(c_float)-0.29361204096868581681,
(c_float)0.00000000000000000000,
(c_float)0.11896017894824251204,
(c_float)-0.11896017894824251204,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000316184,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000151250,
(c_float)0.00000000000003084387,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000002639973,
(c_float)-0.00000000000000000221,
(c_float)0.00000000000000000221,
(c_float)-0.00000000000000001893,
(c_float)0.00000000000000005422,
(c_float)-0.00000000000000000000,
(c_float)0.30595922057225338753,
(c_float)-0.05280794933581747785,
(c_float)0.00000000000000000000,
(c_float)0.03075522660172481590,
(c_float)-0.03075522660172481590,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000071722,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000002061,
(c_float)0.00000000000000073975,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000188648,
(c_float)-0.00000000000000000062,
(c_float)0.00000000000000000062,
(c_float)-0.00000000000000000640,
(c_float)0.00000000000000001437,
(c_float)-0.00000000000000000000,
(c_float)-0.05896595677589915041,
(c_float)-0.00000000000000000000,
(c_float)0.05955470295047372159,
(c_float)-0.05955470295047372159,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000107408,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000043592,
(c_float)-0.00000000000000982423,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000780317,
(c_float)-0.00000000000000000119,
(c_float)0.00000000000000000119,
(c_float)-0.00000000000000000920,
(c_float)0.00000000000000002895,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.47411644271855007959,
(c_float)0.47411644271855007959,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000015056,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000023929,
(c_float)-0.00000000000000463293,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000081487,
(c_float)0.00000000000000000943,
(c_float)-0.00000000000000000900,
(c_float)0.00000000000000007071,
(c_float)-0.00000000000000021485,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.52674850297012665301,
(c_float)0.00000000000000000000,
(c_float)0.52674850297012665301,
(c_float)-0.06244610390836265179,
(c_float)11.68926667397812302340,
(c_float)-0.00000000000003037618,
(c_float)0.05435047498554378825,
(c_float)-0.00000000000001638142,
(c_float)-0.00000000000028244903,
(c_float)3.54662977818937807939,
(c_float)-0.00000000000027074556,
(c_float)-0.00000000000000001233,
(c_float)-0.00000000000000001019,
(c_float)-0.00000000000000005914,
(c_float)-0.00000000000000022191,
(c_float)0.06244610390836265179,
(c_float)0.10112552976641918379,
(c_float)0.02531016311049377049,
(c_float)0.10112552976641918379,
(c_float)-0.00000000000000000020,
(c_float)0.00000000000000000000,
(c_float)0.00000010957620402658,
(c_float)0.00000000000000000000,
(c_float)0.00092073258170856387,
(c_float)-0.92573526853826859639,
(c_float)0.00000000000000000000,
(c_float)0.95748760304115720920,
(c_float)-0.16559577613073489522,
(c_float)0.11561994299685257603,
(c_float)-0.93121409814508715463,
(c_float)0.95730667828144366815,
(c_float)-0.00000000000000000020,
(c_float)0.10176088541959203704,
(c_float)0.03742825560064920454,
(c_float)0.00727262060598269418,
(c_float)0.27179596009476647556,
(c_float)-0.00000012456806488550,
(c_float)0.05098146745462455581,
(c_float)0.00085444908124952205,
(c_float)1.05239135370953573734,
(c_float)0.74756911428943795261,
(c_float)0.88855811011789520570,
(c_float)0.18825205102748374109,
(c_float)0.10729646808472294295,
(c_float)1.05861977894365288932,
(c_float)0.88839021012415808620,
(c_float)-0.00727262060598269418,
(c_float)0.09590020072214069935,
(c_float)-0.00072456497912779081,
(c_float)-0.02707879935206274763,
(c_float)0.00000012036537969299,
(c_float)-0.00507924005712282779,
(c_float)0.00082198020147261489,
(c_float)-1.01688574591039770922,
(c_float)-0.07447967232693517015,
(c_float)0.85479309463873720532,
(c_float)-0.18190079826622457260,
(c_float)0.10321922556733299781,
(c_float)-1.02290403636638949081,
(c_float)0.85463157480830531654,
(c_float)0.00072456497912779113,
(c_float)-0.00755540628352938202,
(c_float)-0.28236436574851714321,
(c_float)-0.00000013313284467743,
(c_float)-0.05296381049127639462,
(c_float)0.00075005023352645614,
(c_float)1.12474944328340686894,
(c_float)-0.77663729341642429027,
(c_float)0.77999173106774299935,
(c_float)0.20119548573234508071,
(c_float)0.09418670194541160778,
(c_float)1.13140610935145002891,
(c_float)0.77984434553908110743,
(c_float)0.00755540628352938202,
(c_float)0.09886424866699547453,
(c_float)-0.00000000000000033942,
(c_float)0.00323009746750420020,
(c_float)-0.00000000000000018304,
(c_float)-0.00000000000000315662,
(c_float)0.07026624064571644623,
(c_float)-0.00000000000000302758,
(c_float)-0.00000014477366906507,
(c_float)0.00081563285117503968,
(c_float)1.22309487322157939282,
(c_float)0.84819236241559503586,
(c_float)0.08743763379969823890,
(c_float)-0.00000000000000275994,
(c_float)0.03355709251288963718,
(c_float)-0.00000000000000144779,
(c_float)-0.00000000000002601647,
(c_float)0.43087765846080405829,
(c_float)-0.00000000000002423800,
(c_float)-0.00000000039374892266,
(c_float)0.00000221832159476065,
(c_float)0.00332651850130842542,
(c_float)0.00230687549103381854,
(c_float)0.00295756403452638499,
(c_float)-0.00000000000000017612,
(c_float)-0.00000000000000008421,
(c_float)-2.19968956450203156905,
(c_float)-0.00000000000000100396,
(c_float)0.00000000000000001669,
(c_float)0.53416388227163458247,
(c_float)-0.00000000000000000000,
(c_float)3.00450007368343685599,
(c_float)-0.00000000000000000210,
(c_float)-0.00000000000000000269,
(c_float)-0.00000000000000001546,
(c_float)-0.00000000000000234455,
(c_float)0.38144227022734178423,
(c_float)0.00000000000000107874,
(c_float)0.00000000000219480659,
(c_float)-0.00000001236273548111,
(c_float)-0.00001853873144272779,
(c_float)-0.00001285624751191529,
(c_float)-0.00001648254333967366,
(c_float)-0.00000000000000012215,
(c_float)-0.00000000000000128688,
(c_float)0.41066482887672306479,
(c_float)0.00000000000000014797,
(c_float)-0.73095421057223719519,
(c_float)0.00000000000000082631,
(c_float)0.07177685342156209236,
(c_float)-0.00000000000000000530,
(c_float)-0.00000000000000034697,
(c_float)0.00000000000000000151,
(c_float)-0.24739523055443246591,
(c_float)-0.00000000000000000083,
(c_float)-0.85698064735126422420,
(c_float)-0.00000000000000000064,
(c_float)-0.00000000000000000093,
(c_float)-0.00000000000000857358,
(c_float)-0.00000000018470029395,
(c_float)0.00000104055790340640,
(c_float)0.00156038471910697313,
(c_float)0.00108209627082930262,
(c_float)0.00138731761854454143,
(c_float)-0.00000000000000000891,
(c_float)0.02413909037253485559,
(c_float)-0.00000000000000007481,
(c_float)-0.15698745303814107266,
(c_float)-0.00000000000000000933,
(c_float)0.00000000000367389297,
(c_float)-2.18885792890650421327,
(c_float)0.00000000382052437339,
(c_float)0.00000000489816010589,
(c_float)-0.00009954133012106361,
(c_float)0.39425198027553221003,
(c_float)-0.00008850091592058052,
(c_float)-0.00101938895269077694,
(c_float)-0.00130741852110501772,
(c_float)-0.01787607127053862655,
};
csc linsys_solver_L = {1770493808, 476, 476, linsys_solver_L_p, linsys_solver_L_i, linsys_solver_L_x, -1};
c_float linsys_solver_Dinv[476] = {
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)500000.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)500000.00000000000000000000,
(c_float)-0.00000099999999000000,
(c_float)-0.00012643406613659880,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.00000100000000000000,
(c_float)500000.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-0.00000290239991576074,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.04997551199412538642,
(c_float)-0.13606612920762170549,
(c_float)-0.27898725394750989448,
(c_float)0.01451189377703507137,
(c_float)-0.08614929125236493734,
(c_float)-0.20577823137460837799,
(c_float)-0.36934756621989345282,
(c_float)8.66954992049508277319,
(c_float)8.67607844505480052533,
(c_float)0.01451131493894157291,
(c_float)-0.09434053925923413519,
(c_float)-0.27882561513673198572,
(c_float)-0.41238944251957848630,
(c_float)-0.10000000000000000555,
(c_float)1.50979329263400496863,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)1.50979329263400496863,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.20149723187988005657,
(c_float)-0.32814942747819936253,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)500000.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)500000.00000000000000000000,
(c_float)-0.00000099999999000000,
(c_float)-0.00012643406613659880,
(c_float)-0.10000000000000000555,
(c_float)1.50979329263400496863,
(c_float)-0.10000000000000000555,
(c_float)1.50979329263400496863,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.20149723187988005657,
(c_float)-0.32814942747819936253,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.00000100000000000000,
(c_float)500000.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-0.00000290239991576074,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.04997551199412538642,
(c_float)-0.13606612920762170549,
(c_float)-0.27898725394750989448,
(c_float)0.01451189377703507137,
(c_float)-0.08614929125236493734,
(c_float)-0.20577823137460837799,
(c_float)-0.36934756621989345282,
(c_float)8.66954992049508277319,
(c_float)8.67607844505480052533,
(c_float)0.01451131493894157291,
(c_float)-0.09434053925923413519,
(c_float)-0.27882561513673198572,
(c_float)-0.41238944251957848630,
(c_float)7.91490443340838378816,
(c_float)8.00501978366833100154,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)0.00725599989470093049,
(c_float)0.01995522730488230589,
(c_float)4.46203150954130922656,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.04997551199412538642,
(c_float)-0.13606612920762170549,
(c_float)-0.27898725394750989448,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)0.00725599989470093049,
(c_float)0.01995522730488230589,
(c_float)4.46203150954130922656,
(c_float)-0.10000000000000000555,
(c_float)1.50979329263400496863,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)1.50979329263400496863,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.20149723187988005657,
(c_float)-0.32814942747819936253,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)500000.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)500000.00000000000000000000,
(c_float)-0.00000099999999000000,
(c_float)-0.00012643406613659880,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.04997551199412538642,
(c_float)-0.13606612920762170549,
(c_float)-0.27898725394750989448,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)0.00725599989470093049,
(c_float)0.01995522730488230589,
(c_float)4.46203150954130922656,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.04997551199412538642,
(c_float)-0.13606612920762170549,
(c_float)-0.27898725394750989448,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)0.00725599989470093049,
(c_float)0.01995522730488230589,
(c_float)4.46203150954130922656,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)500000.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)500000.00000000000000000000,
(c_float)-0.00000099999999000000,
(c_float)-0.00012643406613659880,
(c_float)-0.10000000000000000555,
(c_float)1.50979329263400496863,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)1.50979329263400496863,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.20149723187988005657,
(c_float)-0.32814942747819936253,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.04997551199412538642,
(c_float)-0.13606612920762170549,
(c_float)-0.27898725394750989448,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)0.00998879435586480166,
(c_float)4.45953571094490630600,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)0.00998879435586480166,
(c_float)4.45953571094490630600,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)1.50979329263400496863,
(c_float)-0.00000100000000000000,
(c_float)500000.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.00000100000000000000,
(c_float)500000.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-100.00000000000000000000,
(c_float)0.00994408012914218924,
(c_float)-0.00000100000000000000,
(c_float)-100.00000000000000000000,
(c_float)0.00999999980000000448,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.55600858103189820802,
(c_float)-0.00000199999992000704,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)0.00993420157827734109,
(c_float)-0.00000100000000000000,
(c_float)-100.00000000000000000000,
(c_float)0.00996039375279914735,
(c_float)-0.10000000000000000555,
(c_float)1.50979329263400496863,
(c_float)-0.65365197190426871376,
(c_float)-0.39455368956831415872,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-0.00000100000000000000,
(c_float)-100.00000000000000000000,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-100.00000000000000000000,
(c_float)0.00994408012914218924,
(c_float)-0.00000100000000000000,
(c_float)-100.00000000000000000000,
(c_float)0.00996039375279914735,
(c_float)0.00999000989020969003,
(c_float)0.00999999980000000448,
(c_float)-59.20365583553895305613,
(c_float)-0.55604021795659996918,
(c_float)-0.82547818094041458536,
(c_float)-0.09980173443527164123,
(c_float)-0.55246990512962990127,
(c_float)-0.79407592575947905100,
(c_float)-0.09970961552235547976,
(c_float)0.00911436030177222370,
(c_float)0.02680730754641072466,
(c_float)4.46374903170435111122,
(c_float)4.78270921470687326860,
(c_float)4.98430952394718307374,
(c_float)-0.00000100000000000000,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.10000000000000000555,
(c_float)9.99990000099998965766,
(c_float)-0.00000100000000000000,
(c_float)2.51485421031885136856,
(c_float)-0.00000100000000000000,
(c_float)1.77827308550609597404,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-0.00000100000000000000,
(c_float)-100.00000000000000000000,
(c_float)-0.10000000000000000555,
(c_float)-100.00000000000000000000,
(c_float)-0.00000100000000000000,
(c_float)-100.00000000000000000000,
(c_float)0.00994408012914218924,
(c_float)-0.00000100000000000000,
(c_float)-100.00000000000000000000,
(c_float)0.00996039375279914735,
(c_float)0.00999000989020969003,
(c_float)0.00999999980000000448,
(c_float)-59.20365583553895305613,
(c_float)-0.55604021795659996918,
(c_float)-0.82547818094041458536,
(c_float)-0.09980173443527164123,
(c_float)-0.55246990512962990127,
(c_float)-0.79407592575947905100,
(c_float)-0.09970961552235547976,
(c_float)0.00911436030177222370,
(c_float)0.02680730754641072466,
(c_float)4.46374903170435111122,
(c_float)4.98893211817826198740,
(c_float)5.32983190401188888785,
(c_float)-0.50329898960850893985,
(c_float)-0.00000199998301364125,
(c_float)-0.63007770810035712561,
(c_float)-0.16438964078730808338,
(c_float)16.11439840471128448485,
(c_float)16.11439840471128448485,
(c_float)-0.48413412597973032314,
(c_float)-0.48128005149245745020,
(c_float)-1.04929172886507959817,
(c_float)-1.12525629833265616853,
(c_float)-0.20054752774931000614,
(c_float)-0.41945269533931323069,
(c_float)-0.01107707029276068338,
(c_float)12.49103205839583274894,
(c_float)6.40401993822099857567,
(c_float)6.42644635064797320467,
(c_float)-0.48413412597973032314,
(c_float)-0.48128005149245745020,
(c_float)-1.04929172886507959817,
(c_float)12.49103205839583274894,
(c_float)6.56992105522225333658,
(c_float)6.61946081680563747085,
(c_float)-0.44796778131250808430,
(c_float)-0.93148632582283408254,
(c_float)-0.32981861358587943878,
(c_float)-0.31330572869238831890,
(c_float)-0.00961933645874656594,
(c_float)-1.03765166522400908988,
(c_float)-0.19997457304262725164,
(c_float)13.60310947126168557020,
(c_float)6.50599695830891011639,
(c_float)6.51891751338107106051,
(c_float)-0.44796778131250808430,
(c_float)-0.93148632582283408254,
(c_float)-0.32981861358587954980,
(c_float)13.60310947126168557020,
(c_float)6.67765103018742500751,
(c_float)6.71252452996718229628,
(c_float)-0.99220304032532058436,
(c_float)-1.42789967037929743654,
(c_float)-0.41628038037188913600,
(c_float)8.16487256226244184631,
(c_float)39.48009873350905252209,
(c_float)7.33679318229697763343,
(c_float)7.57460354049413364663,
(c_float)7.41592683520174489331,
(c_float)7.65725987393702389738,
(c_float)8.09639548559846566889,
(c_float)-0.22273180742132700494,
(c_float)-0.30058662838758920532,
(c_float)-1.40251380800104374913,
(c_float)-0.98869774544423982832,
(c_float)-0.00950349600882797151,
(c_float)-0.37287593135728941407,
(c_float)-0.18449144877849646407,
(c_float)-0.30006486074286564669,
(c_float)-0.96233171762820957085,
(c_float)-0.00948039243435848562,
(c_float)-0.18671360981737303297,
(c_float)8.03189145285534422669,
};
c_int linsys_solver_P[476] = {
390,
86,
396,
92,
402,
98,
408,
104,
276,
270,
391,
87,
397,
93,
388,
84,
394,
465,
467,
274,
403,
99,
409,
105,
400,
96,
406,
469,
471,
286,
427,
123,
415,
111,
433,
129,
421,
117,
430,
126,
418,
473,
475,
298,
424,
120,
412,
108,
292,
301,
295,
114,
280,
289,
283,
165,
167,
102,
268,
277,
271,
393,
89,
399,
95,
405,
101,
411,
107,
279,
273,
426,
122,
414,
110,
432,
128,
420,
116,
300,
294,
429,
125,
417,
113,
435,
131,
423,
119,
303,
297,
392,
88,
398,
94,
389,
85,
395,
464,
466,
275,
404,
100,
410,
106,
401,
97,
407,
468,
470,
287,
428,
124,
416,
112,
434,
130,
422,
118,
431,
127,
419,
472,
474,
299,
425,
121,
413,
109,
293,
302,
296,
115,
281,
290,
284,
164,
166,
103,
269,
278,
272,
168,
170,
371,
456,
458,
251,
383,
460,
462,
263,
79,
156,
158,
368,
64,
374,
70,
365,
61,
380,
76,
386,
82,
377,
73,
245,
254,
248,
370,
457,
459,
250,
382,
461,
463,
262,
78,
157,
159,
369,
65,
375,
71,
381,
77,
387,
83,
255,
249,
366,
62,
372,
68,
378,
74,
384,
80,
252,
246,
367,
63,
373,
69,
364,
60,
379,
75,
385,
81,
376,
72,
244,
253,
247,
346,
449,
451,
226,
358,
453,
455,
238,
54,
149,
151,
343,
39,
349,
45,
340,
36,
355,
51,
361,
57,
352,
48,
220,
229,
223,
347,
448,
450,
227,
359,
452,
454,
239,
55,
148,
150,
342,
38,
348,
44,
354,
50,
360,
56,
228,
222,
345,
41,
351,
47,
357,
53,
363,
59,
231,
225,
344,
40,
350,
46,
341,
37,
356,
52,
362,
58,
353,
49,
221,
230,
224,
334,
445,
447,
214,
141,
143,
335,
444,
446,
215,
140,
142,
339,
35,
333,
29,
336,
32,
330,
26,
324,
20,
306,
174,
2,
312,
180,
8,
318,
14,
186,
192,
327,
23,
309,
177,
5,
315,
183,
11,
321,
17,
189,
195,
322,
441,
443,
202,
328,
24,
337,
33,
331,
27,
316,
12,
325,
21,
319,
15,
437,
439,
310,
178,
304,
172,
307,
175,
3,
313,
181,
9,
0,
6,
190,
187,
193,
184,
199,
205,
196,
18,
137,
139,
133,
135,
323,
440,
442,
203,
329,
25,
338,
34,
332,
28,
317,
13,
326,
22,
320,
16,
436,
438,
311,
179,
305,
173,
308,
176,
4,
314,
182,
10,
1,
7,
191,
188,
194,
185,
200,
206,
197,
19,
136,
138,
132,
134,
198,
204,
201,
207,
31,
30,
209,
218,
212,
213,
219,
210,
216,
43,
144,
146,
208,
217,
211,
42,
145,
147,
232,
235,
241,
234,
240,
237,
243,
66,
153,
155,
233,
236,
242,
67,
152,
154,
257,
260,
266,
91,
90,
160,
161,
162,
163,
169,
256,
258,
259,
261,
264,
265,
267,
282,
285,
288,
291,
171,
};
c_float linsys_solver_bp[476];
c_float linsys_solver_sol[476];
c_float linsys_solver_rho_inv_vec[304] = {
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
};
c_int linsys_solver_Pdiag_idx[117] = {
0,
1,
2,
3,
4,
5,
6,
7,
8,
9,
10,
11,
12,
13,
14,
15,
16,
17,
18,
19,
20,
21,
22,
23,
24,
25,
26,
27,
28,
29,
30,
31,
32,
33,
34,
35,
36,
37,
38,
39,
40,
41,
42,
43,
44,
45,
46,
47,
48,
49,
50,
51,
52,
53,
54,
55,
56,
57,
58,
59,
60,
61,
62,
63,
64,
65,
66,
67,
68,
69,
70,
71,
72,
73,
74,
75,
76,
77,
78,
79,
80,
81,
82,
83,
84,
85,
86,
87,
88,
89,
90,
91,
92,
93,
94,
95,
96,
97,
98,
99,
100,
101,
102,
103,
104,
105,
106,
107,
108,
109,
110,
111,
112,
113,
114,
115,
116,
};
c_int linsys_solver_KKT_i[1360] = {
0,
1,
0,
2,
3,
2,
4,
5,
4,
6,
7,
6,
3,
7,
8,
1,
3,
5,
9,
10,
11,
10,
12,
13,
12,
14,
15,
14,
16,
17,
18,
19,
20,
21,
20,
22,
23,
22,
24,
25,
24,
26,
27,
28,
29,
30,
31,
30,
32,
33,
32,
34,
35,
34,
36,
37,
36,
38,
39,
38,
40,
41,
42,
39,
43,
44,
45,
44,
46,
47,
46,
47,
45,
48,
47,
37,
35,
49,
47,
33,
37,
31,
50,
51,
29,
48,
50,
43,
49,
40,
25,
47,
52,
25,
23,
37,
53,
25,
21,
23,
33,
54,
55,
52,
54,
29,
53,
27,
56,
52,
54,
29,
53,
28,
57,
19,
52,
54,
29,
53,
26,
15,
25,
58,
15,
13,
23,
59,
15,
11,
13,
21,
60,
61,
62,
61,
63,
64,
63,
65,
66,
65,
67,
68,
67,
64,
68,
69,
62,
64,
66,
70,
71,
72,
71,
73,
74,
73,
75,
76,
75,
77,
78,
77,
78,
76,
79,
74,
78,
72,
80,
81,
82,
81,
83,
84,
83,
85,
86,
85,
87,
88,
87,
88,
86,
89,
84,
88,
82,
90,
91,
92,
91,
93,
94,
93,
95,
96,
95,
97,
98,
99,
100,
101,
102,
101,
103,
104,
103,
105,
106,
105,
107,
108,
109,
110,
111,
112,
111,
113,
114,
113,
115,
116,
115,
117,
118,
117,
119,
120,
119,
121,
122,
123,
120,
124,
125,
126,
125,
127,
128,
127,
128,
126,
129,
128,
118,
116,
130,
128,
114,
118,
112,
131,
132,
110,
129,
131,
124,
130,
121,
106,
128,
133,
106,
104,
118,
134,
106,
102,
104,
114,
135,
136,
133,
135,
110,
134,
108,
137,
133,
135,
110,
134,
109,
138,
100,
133,
135,
110,
134,
107,
96,
106,
139,
96,
94,
104,
140,
96,
92,
94,
102,
141,
142,
129,
80,
131,
90,
124,
79,
130,
89,
122,
143,
129,
80,
131,
90,
124,
79,
130,
89,
123,
144,
145,
146,
147,
148,
149,
150,
151,
152,
147,
151,
148,
153,
151,
149,
154,
151,
150,
155,
156,
155,
157,
158,
157,
159,
160,
159,
161,
162,
161,
163,
164,
163,
165,
166,
165,
160,
166,
167,
160,
158,
164,
168,
160,
156,
158,
162,
169,
170,
171,
172,
173,
174,
175,
176,
177,
178,
173,
177,
174,
179,
177,
175,
180,
177,
176,
181,
182,
181,
183,
184,
183,
185,
186,
185,
187,
188,
187,
184,
188,
189,
182,
184,
186,
190,
191,
192,
191,
193,
194,
193,
195,
196,
195,
197,
198,
197,
194,
198,
199,
192,
194,
196,
200,
201,
202,
201,
203,
204,
203,
205,
206,
205,
207,
208,
207,
209,
210,
209,
211,
212,
211,
206,
212,
213,
206,
204,
210,
214,
206,
202,
204,
208,
215,
216,
217,
218,
219,
220,
221,
222,
223,
224,
219,
223,
220,
225,
223,
221,
226,
223,
222,
227,
228,
227,
229,
230,
229,
231,
232,
231,
233,
234,
233,
235,
236,
235,
237,
238,
237,
232,
238,
239,
232,
230,
236,
240,
232,
228,
230,
234,
241,
242,
243,
244,
245,
246,
247,
248,
249,
250,
245,
249,
246,
251,
249,
247,
252,
249,
248,
253,
254,
253,
255,
256,
255,
257,
258,
257,
259,
260,
259,
256,
260,
261,
254,
256,
258,
262,
263,
264,
263,
265,
266,
265,
267,
268,
267,
269,
270,
269,
266,
270,
271,
264,
266,
268,
272,
273,
274,
273,
275,
276,
275,
277,
278,
277,
279,
280,
279,
281,
282,
281,
283,
284,
283,
278,
284,
285,
278,
276,
282,
286,
278,
274,
276,
280,
287,
288,
289,
290,
291,
292,
291,
289,
293,
291,
290,
294,
295,
296,
297,
298,
297,
295,
299,
297,
296,
300,
301,
300,
302,
303,
302,
304,
305,
304,
306,
307,
306,
308,
309,
308,
310,
311,
312,
311,
310,
313,
314,
315,
314,
313,
316,
317,
316,
312,
315,
317,
318,
315,
309,
319,
320,
321,
320,
322,
323,
324,
323,
322,
325,
326,
327,
326,
325,
328,
329,
328,
324,
327,
329,
330,
327,
321,
331,
332,
333,
334,
335,
336,
337,
336,
338,
339,
338,
340,
341,
340,
342,
343,
342,
344,
345,
344,
346,
347,
346,
348,
349,
350,
351,
352,
353,
354,
355,
356,
355,
354,
357,
358,
359,
358,
357,
360,
353,
352,
361,
351,
350,
361,
362,
360,
356,
361,
359,
347,
363,
360,
361,
359,
345,
364,
360,
361,
343,
365,
343,
347,
345,
341,
366,
343,
345,
339,
367,
343,
337,
368,
369,
362,
368,
366,
335,
367,
332,
370,
368,
366,
335,
367,
333,
371,
368,
366,
335,
367,
334,
372,
365,
318,
363,
330,
362,
319,
364,
331,
348,
373,
365,
318,
363,
330,
362,
319,
364,
331,
349,
374,
375,
376,
377,
378,
379,
378,
380,
381,
380,
382,
383,
382,
384,
385,
384,
386,
387,
386,
388,
389,
388,
390,
391,
392,
393,
394,
395,
396,
397,
398,
397,
396,
399,
400,
401,
400,
399,
402,
395,
394,
403,
393,
392,
403,
404,
402,
398,
403,
401,
389,
405,
402,
403,
401,
387,
406,
402,
403,
385,
407,
385,
389,
387,
383,
408,
385,
387,
381,
409,
385,
379,
410,
411,
404,
410,
408,
377,
409,
374,
412,
410,
408,
377,
409,
375,
413,
410,
408,
377,
409,
376,
414,
407,
318,
405,
330,
404,
319,
406,
331,
390,
415,
407,
318,
405,
330,
404,
319,
406,
331,
391,
317,
309,
307,
412,
370,
413,
371,
416,
309,
305,
412,
370,
413,
371,
417,
329,
321,
303,
412,
370,
413,
371,
418,
321,
301,
412,
370,
413,
371,
419,
420,
377,
297,
294,
421,
335,
291,
288,
379,
420,
278,
298,
299,
422,
379,
420,
381,
276,
298,
299,
423,
379,
383,
420,
381,
274,
298,
299,
424,
303,
301,
264,
298,
292,
299,
293,
425,
301,
266,
298,
292,
299,
293,
426,
307,
305,
254,
298,
292,
299,
293,
427,
305,
256,
298,
292,
299,
293,
428,
429,
297,
285,
287,
245,
286,
242,
430,
285,
262,
287,
272,
245,
261,
286,
271,
243,
431,
285,
262,
287,
272,
245,
261,
286,
271,
244,
337,
421,
232,
292,
293,
432,
337,
421,
339,
230,
292,
293,
433,
337,
341,
421,
339,
228,
292,
293,
434,
435,
291,
239,
241,
219,
240,
216,
436,
239,
262,
241,
272,
219,
261,
240,
271,
217,
437,
239,
262,
241,
272,
219,
261,
240,
271,
218,
238,
224,
206,
225,
226,
438,
238,
234,
224,
236,
202,
225,
226,
439,
238,
224,
236,
204,
225,
226,
440,
258,
260,
192,
251,
225,
252,
226,
441,
260,
194,
251,
225,
252,
226,
442,
268,
270,
182,
251,
225,
252,
226,
443,
270,
184,
251,
225,
252,
226,
444,
445,
223,
213,
215,
173,
214,
170,
446,
213,
200,
215,
190,
173,
199,
214,
189,
171,
447,
213,
200,
215,
190,
173,
199,
214,
189,
172,
284,
250,
160,
251,
252,
448,
284,
280,
250,
282,
156,
251,
252,
449,
284,
250,
282,
158,
251,
252,
450,
451,
249,
167,
169,
147,
168,
144,
452,
167,
200,
169,
190,
147,
199,
168,
189,
145,
453,
167,
200,
169,
190,
147,
199,
168,
189,
146,
166,
152,
96,
153,
154,
454,
166,
162,
152,
164,
92,
153,
154,
455,
166,
152,
164,
94,
153,
154,
456,
457,
151,
139,
141,
100,
140,
97,
458,
177,
58,
60,
19,
59,
16,
459,
139,
9,
141,
70,
100,
8,
140,
69,
98,
460,
58,
9,
60,
70,
19,
8,
59,
69,
17,
461,
139,
9,
141,
70,
100,
8,
140,
69,
99,
462,
58,
9,
60,
70,
19,
8,
59,
69,
18,
463,
48,
80,
50,
90,
43,
79,
49,
89,
41,
212,
178,
15,
179,
180,
464,
196,
198,
1,
153,
179,
154,
180,
465,
212,
208,
178,
210,
11,
179,
180,
466,
186,
188,
62,
153,
179,
154,
180,
467,
198,
3,
153,
179,
154,
180,
468,
212,
178,
210,
13,
179,
180,
469,
188,
64,
153,
179,
154,
180,
470,
5,
7,
74,
136,
55,
137,
56,
471,
66,
68,
84,
136,
55,
137,
56,
472,
7,
78,
136,
55,
137,
56,
473,
68,
88,
136,
55,
137,
56,
474,
475,
48,
80,
50,
90,
43,
79,
49,
89,
42,
};
c_int linsys_solver_KKT_p[477] = {
0,
1,
3,
4,
6,
7,
9,
10,
12,
15,
19,
20,
22,
23,
25,
26,
28,
29,
30,
31,
32,
33,
35,
36,
38,
39,
41,
42,
43,
44,
45,
46,
48,
49,
51,
52,
54,
55,
57,
58,
60,
61,
62,
63,
65,
66,
68,
69,
71,
74,
78,
83,
90,
93,
97,
102,
108,
114,
121,
124,
128,
133,
134,
136,
137,
139,
140,
142,
143,
145,
148,
152,
153,
155,
156,
158,
159,
161,
162,
164,
167,
171,
172,
174,
175,
177,
178,
180,
181,
183,
186,
190,
191,
193,
194,
196,
197,
199,
200,
201,
202,
203,
204,
206,
207,
209,
210,
212,
213,
214,
215,
216,
217,
219,
220,
222,
223,
225,
226,
228,
229,
231,
232,
233,
234,
236,
237,
239,
240,
242,
245,
249,
254,
261,
264,
268,
273,
279,
285,
292,
295,
299,
304,
314,
324,
325,
326,
327,
328,
329,
330,
331,
332,
336,
339,
342,
343,
345,
346,
348,
349,
351,
352,
354,
355,
357,
358,
360,
363,
367,
372,
373,
374,
375,
376,
377,
378,
379,
380,
384,
387,
390,
391,
393,
394,
396,
397,
399,
400,
402,
405,
409,
410,
412,
413,
415,
416,
418,
419,
421,
424,
428,
429,
431,
432,
434,
435,
437,
438,
440,
441,
443,
444,
446,
449,
453,
458,
459,
460,
461,
462,
463,
464,
465,
466,
470,
473,
476,
477,
479,
480,
482,
483,
485,
486,
488,
489,
491,
492,
494,
497,
501,
506,
507,
508,
509,
510,
511,
512,
513,
514,
518,
521,
524,
525,
527,
528,
530,
531,
533,
534,
536,
539,
543,
544,
546,
547,
549,
550,
552,
553,
555,
558,
562,
563,
565,
566,
568,
569,
571,
572,
574,
575,
577,
578,
580,
583,
587,
592,
593,
594,
595,
596,
599,
602,
603,
604,
605,
606,
609,
612,
613,
615,
616,
618,
619,
621,
622,
624,
625,
627,
628,
629,
632,
633,
634,
637,
638,
640,
644,
647,
648,
650,
651,
652,
655,
656,
657,
660,
661,
663,
667,
670,
671,
672,
673,
674,
675,
677,
678,
680,
681,
683,
684,
686,
687,
689,
690,
692,
693,
694,
695,
696,
697,
698,
699,
700,
703,
704,
705,
708,
711,
714,
716,
722,
727,
731,
736,
740,
743,
750,
756,
762,
772,
782,
783,
784,
785,
786,
787,
789,
790,
792,
793,
795,
796,
798,
799,
801,
802,
804,
805,
806,
807,
808,
809,
810,
811,
812,
815,
816,
817,
820,
823,
826,
828,
834,
839,
843,
848,
852,
855,
862,
868,
874,
884,
894,
902,
909,
917,
924,
928,
932,
938,
945,
953,
961,
968,
976,
983,
990,
1000,
1010,
1016,
1023,
1031,
1038,
1048,
1058,
1064,
1072,
1079,
1087,
1094,
1102,
1109,
1116,
1126,
1136,
1142,
1150,
1157,
1164,
1174,
1184,
1190,
1198,
1205,
1212,
1219,
1229,
1239,
1249,
1259,
1269,
1275,
1283,
1291,
1299,
1306,
1313,
1320,
1328,
1336,
1343,
1350,
1360,
};
c_float linsys_solver_KKT_x[1360] = {
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)-0.01000000000000000021,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.99994734478168090241,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.01000000000000000021,
(c_float)0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.99994734478168090241,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.01000000000000000021,
(c_float)0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.01225485481008077929,
(c_float)-0.06026608584051092871,
(c_float)-0.00088570197994891598,
(c_float)-1.00000000000000000000,
(c_float)-0.02012491977216335909,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)0.06026608584051092871,
(c_float)0.00088570197994891598,
(c_float)1.00000000000000000000,
(c_float)0.02012491977216335909,
(c_float)0.99963640322633928736,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.99994734478168090241,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.01000000000000000021,
(c_float)0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-10.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)0.98459999999999991971,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-10.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)0.98459999999999991971,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.83011137521364919412,
(c_float)-0.01000000000000000021,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-0.99994734478168090241,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.01000000000000000021,
(c_float)-0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)-0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)-0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-0.99994734478168090241,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.01000000000000000021,
(c_float)-0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.01225485481008077929,
(c_float)-0.06026608584051092871,
(c_float)0.00088570197994891598,
(c_float)-1.00000000000000000000,
(c_float)0.02012491977216335909,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)0.06026608584051092871,
(c_float)-0.00088570197994891598,
(c_float)1.00000000000000000000,
(c_float)-0.02012491977216335909,
(c_float)0.99963640322633928736,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)-0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)-0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-0.99994734478168090241,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.01000000000000000021,
(c_float)-0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.01225485481008077929,
(c_float)-0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)0.01225485481008077929,
(c_float)-1.00000000000000000000,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)1.00000000000000000000,
(c_float)0.99963640322633928736,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-0.99994734478168090241,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.01000000000000000021,
(c_float)-0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)0.01225485481008077929,
(c_float)-1.00000000000000000000,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)1.00000000000000000000,
(c_float)0.99963640322633928736,
(c_float)-10.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)0.98459999999999991971,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.99994734478168090241,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.01000000000000000021,
(c_float)0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)0.01225485481008077929,
(c_float)-1.00000000000000000000,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)1.00000000000000000000,
(c_float)0.99963640322633928736,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.99994734478168090241,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.01000000000000000021,
(c_float)0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)0.01225485481008077929,
(c_float)-1.00000000000000000000,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)1.00000000000000000000,
(c_float)0.99963640322633928736,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-10.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)0.98459999999999991971,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-0.99994734478168090241,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.01000000000000000021,
(c_float)-0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.01225485481008077929,
(c_float)-1.00000000000000000000,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)1.00000000000000000000,
(c_float)0.99963640322633928736,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.01225485481008077929,
(c_float)-1.00000000000000000000,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)1.00000000000000000000,
(c_float)0.99963640322633928736,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.56234232519034910158,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.00000100000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.56234232519034910158,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.39763636438352534253,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.98459999999999991971,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.56234232519034910158,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.39763636438352534253,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.83011137521364919412,
(c_float)-0.01000000000000000021,
(c_float)0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)0.00284524705606227675,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.99994734478168090241,
(c_float)0.05013091270325255311,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.10000000000000000555,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.99994734478168090241,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)0.01225485481008077929,
(c_float)-0.06026608584051092871,
(c_float)-0.00088570197994891598,
(c_float)-1.00000000000000000000,
(c_float)-0.02012491977216335909,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)0.06026608584051092871,
(c_float)0.00088570197994891598,
(c_float)1.00000000000000000000,
(c_float)0.02012491977216335909,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)-0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)-1000000.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.39763636438352534253,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)0.56234232519034910158,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-10.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-1000000.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.56234232519034910158,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-1000000.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.39763636438352534253,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)-1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.83011137521364919412,
(c_float)-0.01000000000000000021,
(c_float)-0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)-0.00284524705606227675,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-0.99994734478168090241,
(c_float)-0.05013091270325255311,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.10000000000000000555,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)-0.99994734478168090241,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.01000000000000000021,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)-0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)-0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)0.01225485481008077929,
(c_float)-0.06026608584051092871,
(c_float)0.00088570197994891598,
(c_float)-1.00000000000000000000,
(c_float)0.02012491977216335909,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)0.06026608584051092871,
(c_float)-0.00088570197994891598,
(c_float)1.00000000000000000000,
(c_float)-0.02012491977216335909,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)-0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)-1.00000000000000000000,
(c_float)-0.02243778349203921080,
(c_float)0.02243778349203921080,
(c_float)-0.02243778349203921080,
(c_float)0.02243778349203921080,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.25310095830264162098,
(c_float)-0.25310095830264162098,
(c_float)0.25310095830264162098,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)0.01564740164576418346,
(c_float)0.01564740164576418346,
(c_float)0.01564740164576418346,
(c_float)0.01564740164576418346,
(c_float)-0.01000000000000000021,
(c_float)0.98459999999999991971,
(c_float)-1.00000000000000000000,
(c_float)0.26154318119599884174,
(c_float)0.26154318119599884174,
(c_float)0.26154318119599884174,
(c_float)0.26154318119599884174,
(c_float)-0.01000000000000000021,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.83011137521364919412,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.10000000000000000555,
(c_float)-1.00000000000000000000,
(c_float)-0.06026608584051092871,
(c_float)0.06026608584051092871,
(c_float)-0.01000000000000000021,
(c_float)-0.99994734478168090241,
(c_float)-0.05013091270325255311,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)0.02012491977216335909,
(c_float)-0.02012491977216335909,
(c_float)-0.01000000000000000021,
(c_float)-0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)-0.00284524705606227675,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)0.00088570197994891598,
(c_float)-0.00088570197994891598,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)0.01564740164576418346,
(c_float)0.01564740164576418346,
(c_float)0.01564740164576418346,
(c_float)0.01564740164576418346,
(c_float)-0.01000000000000000021,
(c_float)0.98459999999999991971,
(c_float)-1.00000000000000000000,
(c_float)0.26154318119599884174,
(c_float)0.26154318119599884174,
(c_float)0.26154318119599884174,
(c_float)0.26154318119599884174,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)-1.00000000000000000000,
(c_float)-0.02243778349203921080,
(c_float)0.02243778349203921080,
(c_float)-0.02243778349203921080,
(c_float)0.02243778349203921080,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.25310095830264162098,
(c_float)-0.25310095830264162098,
(c_float)0.25310095830264162098,
(c_float)-0.01000000000000000021,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)-0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)-0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)0.01225485481008077929,
(c_float)-0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)1.00000000000000000000,
(c_float)0.10000000000000000555,
(c_float)-1.00000000000000000000,
(c_float)-0.06026608584051092871,
(c_float)0.06026608584051092871,
(c_float)-0.01000000000000000021,
(c_float)0.99994734478168090241,
(c_float)0.05013091270325255311,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.02012491977216335909,
(c_float)0.02012491977216335909,
(c_float)-0.01000000000000000021,
(c_float)0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)0.00284524705606227675,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.00088570197994891598,
(c_float)0.00088570197994891598,
(c_float)-0.01000000000000000021,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)0.01225485481008077929,
(c_float)-0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)1.00000000000000000000,
(c_float)0.10000000000000000555,
(c_float)-1.00000000000000000000,
(c_float)-0.06026608584051092871,
(c_float)0.06026608584051092871,
(c_float)-0.01000000000000000021,
(c_float)0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)0.00284524705606227675,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.00088570197994891598,
(c_float)0.00088570197994891598,
(c_float)-0.01000000000000000021,
(c_float)0.99994734478168090241,
(c_float)0.05013091270325255311,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.02012491977216335909,
(c_float)0.02012491977216335909,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)-1.00000000000000000000,
(c_float)-0.02243778349203921080,
(c_float)0.02243778349203921080,
(c_float)-0.02243778349203921080,
(c_float)0.02243778349203921080,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.25310095830264162098,
(c_float)-0.25310095830264162098,
(c_float)0.25310095830264162098,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)0.01564740164576418346,
(c_float)0.01564740164576418346,
(c_float)0.01564740164576418346,
(c_float)0.01564740164576418346,
(c_float)-0.01000000000000000021,
(c_float)0.98459999999999991971,
(c_float)-1.00000000000000000000,
(c_float)0.26154318119599884174,
(c_float)0.26154318119599884174,
(c_float)0.26154318119599884174,
(c_float)0.26154318119599884174,
(c_float)-0.01000000000000000021,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)0.01225485481008077929,
(c_float)-0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)1.00000000000000000000,
(c_float)0.10000000000000000555,
(c_float)-1.00000000000000000000,
(c_float)-0.06026608584051092871,
(c_float)0.06026608584051092871,
(c_float)-0.01000000000000000021,
(c_float)-0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)-0.00284524705606227675,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)0.00088570197994891598,
(c_float)-0.00088570197994891598,
(c_float)-0.01000000000000000021,
(c_float)-0.99994734478168090241,
(c_float)-0.05013091270325255311,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)0.02012491977216335909,
(c_float)-0.02012491977216335909,
(c_float)-0.01000000000000000021,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)-0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)-0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)0.01225485481008077929,
(c_float)-0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)1.00000000000000000000,
(c_float)0.10000000000000000555,
(c_float)-1.00000000000000000000,
(c_float)-0.06026608584051092871,
(c_float)0.06026608584051092871,
(c_float)-0.01000000000000000021,
(c_float)-0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)-0.00284524705606227675,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)0.00088570197994891598,
(c_float)-0.00088570197994891598,
(c_float)-0.01000000000000000021,
(c_float)-0.99994734478168090241,
(c_float)-0.05013091270325255311,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)0.02012491977216335909,
(c_float)-0.02012491977216335909,
(c_float)-0.01000000000000000021,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)-0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)-0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)0.00000100000000000000,
(c_float)-0.83011137521364919412,
(c_float)0.10000000000000000555,
(c_float)0.00284524705606227675,
(c_float)0.83011137521364919412,
(c_float)0.05013091270325255311,
(c_float)1.00000000000000000000,
(c_float)0.01225485481008077929,
(c_float)-0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)-0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)0.06026608584051092871,
(c_float)-0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)0.01225485481008077929,
(c_float)-0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)-0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)-1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)-0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
(c_float)1.00000000000000000000,
(c_float)0.10000000000000000555,
(c_float)-1.00000000000000000000,
(c_float)-0.06026608584051092871,
(c_float)0.06026608584051092871,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)-1.00000000000000000000,
(c_float)-0.02243778349203921080,
(c_float)0.02243778349203921080,
(c_float)-0.02243778349203921080,
(c_float)0.02243778349203921080,
(c_float)-0.01000000000000000021,
(c_float)0.08678003520989943653,
(c_float)1.00000000000000000000,
(c_float)0.00284524705606227675,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)-0.00088570197994891598,
(c_float)0.00088570197994891598,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)0.01564740164576418346,
(c_float)0.01564740164576418346,
(c_float)0.01564740164576418346,
(c_float)0.01564740164576418346,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.25310095830264162098,
(c_float)-0.25310095830264162098,
(c_float)0.25310095830264162098,
(c_float)-0.01000000000000000021,
(c_float)0.99994734478168090241,
(c_float)0.05013091270325255311,
(c_float)0.67639925404571177303,
(c_float)-0.68697872643277657634,
(c_float)-0.02012491977216335909,
(c_float)0.02012491977216335909,
(c_float)-0.01000000000000000021,
(c_float)0.98459999999999991971,
(c_float)-1.00000000000000000000,
(c_float)0.26154318119599884174,
(c_float)0.26154318119599884174,
(c_float)0.26154318119599884174,
(c_float)0.26154318119599884174,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.17782794100389229253,
(c_float)-1.00000000000000000000,
(c_float)-0.02243778349203921080,
(c_float)0.02243778349203921080,
(c_float)-0.02243778349203921080,
(c_float)0.02243778349203921080,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)0.11796934580826992645,
(c_float)-1.00000000000000000000,
(c_float)0.01564740164576418346,
(c_float)0.01564740164576418346,
(c_float)0.01564740164576418346,
(c_float)0.01564740164576418346,
(c_float)-0.01000000000000000021,
(c_float)1.00000000000000000000,
(c_float)-1.00000000000000000000,
(c_float)-0.25310095830264162098,
(c_float)0.25310095830264162098,
(c_float)-0.25310095830264162098,
(c_float)0.25310095830264162098,
(c_float)-0.01000000000000000021,
(c_float)0.98459999999999991971,
(c_float)-1.00000000000000000000,
(c_float)0.26154318119599884174,
(c_float)0.26154318119599884174,
(c_float)0.26154318119599884174,
(c_float)0.26154318119599884174,
(c_float)-0.01000000000000000021,
(c_float)0.01225485481008077929,
(c_float)0.06026608584051092871,
(c_float)0.02243778349203921080,
(c_float)0.00088570197994891598,
(c_float)0.01564740164576418346,
(c_float)1.00000000000000000000,
(c_float)0.25310095830264162098,
(c_float)0.02012491977216335909,
(c_float)0.26154318119599884174,
(c_float)0.99963640322633928736,
};
csc linsys_solver_KKT = {1360, 476, 476, linsys_solver_KKT_p, linsys_solver_KKT_i, linsys_solver_KKT_x, -1};
c_int linsys_solver_PtoKKT[117] = {
629,
700,
812,
652,
705,
817,
657,
638,
690,
802,
661,
687,
799,
648,
622,
681,
793,
616,
678,
790,
613,
525,
477,
563,
544,
480,
566,
547,
531,
486,
572,
550,
489,
575,
553,
410,
429,
343,
391,
432,
346,
394,
416,
438,
352,
397,
441,
355,
400,
1,
20,
191,
134,
23,
194,
137,
7,
33,
204,
140,
36,
207,
143,
156,
49,
220,
175,
55,
226,
181,
153,
46,
217,
172,
52,
223,
178,
874,
762,
884,
772,
862,
750,
868,
756,
606,
596,
609,
599,
990,
1038,
1000,
1048,
518,
470,
521,
473,
1164,
1116,
1174,
1126,
336,
384,
339,
387,
1219,
1229,
1239,
1249,
273,
102,
279,
108,
304,
1259,
314,
1350,
};
c_int linsys_solver_AtoKKT[884] = {
709,
727,
716,
722,
710,
821,
839,
828,
834,
822,
630,
640,
631,
701,
717,
702,
813,
829,
814,
653,
663,
654,
712,
728,
718,
714,
723,
713,
824,
840,
830,
826,
835,
825,
635,
641,
644,
636,
706,
719,
724,
707,
818,
831,
836,
819,
658,
664,
667,
659,
729,
740,
731,
736,
685,
841,
852,
843,
848,
797,
642,
894,
639,
720,
732,
691,
832,
844,
803,
665,
909,
662,
744,
745,
746,
747,
748,
749,
856,
857,
858,
859,
860,
861,
645,
895,
902,
626,
725,
733,
737,
688,
837,
845,
849,
800,
668,
910,
917,
649,
741,
1010,
1023,
1016,
676,
853,
932,
945,
938,
788,
896,
968,
623,
734,
1024,
682,
846,
946,
794,
911,
953,
617,
929,
1011,
1025,
930,
1017,
931,
925,
933,
947,
926,
939,
927,
903,
969,
976,
620,
738,
1026,
1018,
679,
850,
948,
940,
791,
918,
954,
961,
614,
1012,
494,
501,
497,
484,
934,
580,
587,
583,
570,
970,
539,
526,
1027,
502,
478,
949,
588,
564,
955,
558,
545,
1032,
1033,
1034,
1035,
1036,
1037,
984,
985,
986,
987,
988,
989,
977,
540,
536,
529,
1019,
503,
498,
481,
941,
589,
584,
567,
962,
559,
555,
548,
495,
1058,
1064,
1072,
493,
581,
1136,
1142,
1150,
579,
541,
1079,
532,
504,
1065,
487,
590,
1143,
573,
560,
1094,
551,
467,
1059,
1066,
468,
1073,
469,
515,
1137,
1144,
516,
1151,
517,
537,
1080,
1087,
535,
499,
1067,
1074,
490,
585,
1145,
1152,
576,
556,
1095,
1102,
554,
1060,
446,
453,
449,
436,
1138,
360,
367,
363,
350,
1081,
424,
411,
1068,
454,
430,
1146,
368,
344,
1096,
405,
392,
1110,
1111,
1112,
1113,
1114,
1115,
1158,
1159,
1160,
1161,
1162,
1163,
1088,
425,
421,
414,
1075,
455,
450,
433,
1153,
369,
364,
347,
1103,
406,
402,
395,
447,
1269,
1283,
1306,
445,
361,
1184,
1190,
1198,
359,
426,
1275,
417,
456,
1284,
439,
370,
1191,
353,
407,
1291,
398,
381,
1270,
1285,
382,
1307,
383,
333,
1185,
1192,
334,
1199,
335,
422,
1276,
1299,
420,
451,
1286,
1308,
442,
365,
1193,
1200,
356,
403,
1292,
1313,
401,
1271,
121,
128,
124,
27,
1186,
292,
299,
295,
198,
1277,
15,
2,
1287,
129,
21,
1194,
300,
192,
1293,
148,
135,
1213,
1214,
1215,
1216,
1217,
1218,
1206,
1207,
1208,
1209,
1210,
1211,
1300,
16,
12,
5,
1309,
130,
125,
24,
1201,
301,
296,
195,
1314,
149,
145,
138,
122,
90,
97,
93,
40,
293,
261,
268,
264,
211,
17,
1320,
8,
131,
98,
34,
302,
269,
205,
150,
1328,
141,
115,
116,
117,
118,
119,
120,
286,
287,
288,
289,
290,
291,
13,
1321,
1336,
11,
126,
99,
94,
37,
297,
270,
265,
208,
146,
1329,
1343,
144,
91,
71,
78,
74,
70,
262,
242,
249,
245,
241,
1322,
167,
157,
100,
79,
50,
271,
250,
221,
1330,
186,
176,
84,
85,
86,
87,
88,
89,
255,
256,
257,
258,
259,
260,
1337,
168,
164,
163,
95,
80,
75,
56,
266,
251,
246,
227,
1344,
187,
183,
182,
72,
67,
243,
238,
169,
154,
81,
47,
252,
218,
188,
173,
63,
59,
234,
230,
165,
160,
76,
53,
247,
224,
184,
179,
875,
876,
877,
878,
879,
880,
881,
882,
883,
763,
764,
765,
766,
767,
768,
769,
770,
771,
885,
886,
887,
888,
889,
890,
891,
892,
893,
773,
774,
775,
776,
777,
778,
779,
780,
781,
863,
897,
864,
912,
865,
904,
866,
919,
867,
751,
898,
752,
913,
753,
905,
754,
920,
755,
869,
899,
870,
914,
871,
906,
872,
921,
873,
757,
900,
758,
915,
759,
907,
760,
922,
761,
935,
971,
950,
956,
607,
978,
942,
963,
608,
1013,
972,
1028,
957,
597,
979,
1020,
964,
598,
936,
973,
951,
958,
610,
980,
943,
965,
611,
1014,
974,
1029,
959,
600,
981,
1021,
966,
601,
991,
992,
993,
994,
995,
996,
997,
998,
999,
1039,
1040,
1041,
1042,
1043,
1044,
1045,
1046,
1047,
1001,
1002,
1003,
1004,
1005,
1006,
1007,
1008,
1009,
1049,
1050,
1051,
1052,
1053,
1054,
1055,
1056,
1057,
1139,
1082,
1147,
1097,
519,
1089,
1154,
1104,
520,
1061,
1083,
1069,
1098,
471,
1090,
1076,
1105,
472,
1140,
1084,
1148,
1099,
522,
1091,
1155,
1106,
523,
1062,
1085,
1070,
1100,
474,
1092,
1077,
1107,
475,
1165,
1166,
1167,
1168,
1169,
1170,
1171,
1172,
1173,
1117,
1118,
1119,
1120,
1121,
1122,
1123,
1124,
1125,
1175,
1176,
1177,
1178,
1179,
1180,
1181,
1182,
1183,
1127,
1128,
1129,
1130,
1131,
1132,
1133,
1134,
1135,
1187,
1278,
1195,
1294,
337,
1301,
1202,
1315,
338,
1272,
1279,
1288,
1295,
385,
1302,
1310,
1316,
386,
1188,
1280,
1196,
1296,
340,
1303,
1203,
1317,
341,
1273,
1281,
1289,
1297,
388,
1304,
1311,
1318,
389,
1220,
1221,
1222,
1223,
1224,
1225,
1226,
1227,
1228,
1230,
1231,
1232,
1233,
1234,
1235,
1236,
1237,
1238,
1240,
1241,
1242,
1243,
1244,
1245,
1246,
1247,
1248,
1250,
1251,
1252,
1253,
1254,
1255,
1256,
1257,
1258,
274,
1323,
275,
1331,
276,
1338,
277,
1345,
278,
103,
1324,
104,
1332,
105,
1339,
106,
1346,
107,
280,
1325,
281,
1333,
282,
1340,
283,
1347,
284,
109,
1326,
110,
1334,
111,
1341,
112,
1348,
113,
305,
306,
307,
308,
309,
310,
311,
312,
313,
1260,
1261,
1262,
1263,
1264,
1265,
1266,
1267,
1268,
315,
316,
317,
318,
319,
320,
321,
322,
323,
1351,
1352,
1353,
1354,
1355,
1356,
1357,
1358,
1359,
};
c_int linsys_solver_rhotoKKT[304] = {
697,
809,
628,
699,
811,
651,
695,
807,
633,
704,
816,
656,
730,
842,
643,
721,
833,
666,
715,
827,
646,
726,
838,
669,
742,
854,
901,
735,
847,
916,
673,
785,
908,
739,
851,
923,
1015,
937,
975,
1030,
952,
960,
595,
605,
982,
1022,
944,
967,
496,
582,
542,
505,
591,
561,
461,
509,
538,
500,
586,
557,
1063,
1141,
1086,
1071,
1149,
1101,
465,
513,
1093,
1078,
1156,
1108,
448,
362,
427,
457,
371,
408,
375,
327,
423,
452,
366,
404,
1274,
1189,
1282,
1290,
1197,
1298,
379,
331,
1305,
1312,
1204,
1319,
123,
294,
18,
132,
303,
151,
31,
202,
14,
127,
298,
147,
92,
263,
1327,
101,
272,
1335,
44,
215,
1342,
96,
267,
1349,
73,
244,
170,
82,
253,
189,
64,
235,
166,
77,
248,
185,
696,
808,
627,
698,
810,
650,
694,
806,
632,
703,
815,
655,
683,
795,
637,
689,
801,
660,
670,
782,
624,
686,
798,
647,
674,
786,
621,
680,
792,
615,
592,
602,
618,
677,
789,
612,
482,
568,
524,
476,
562,
543,
458,
506,
527,
479,
565,
546,
491,
577,
530,
485,
571,
549,
462,
510,
533,
488,
574,
552,
434,
348,
409,
428,
342,
390,
372,
324,
412,
431,
345,
393,
443,
357,
415,
437,
351,
396,
376,
328,
418,
440,
354,
399,
25,
196,
0,
19,
190,
133,
28,
199,
3,
22,
193,
136,
38,
209,
6,
32,
203,
139,
41,
212,
9,
35,
206,
142,
68,
239,
155,
48,
219,
174,
60,
231,
161,
54,
225,
180,
65,
236,
152,
45,
216,
171,
57,
228,
158,
51,
222,
177,
804,
692,
805,
693,
783,
671,
784,
672,
603,
593,
604,
594,
507,
459,
508,
460,
511,
463,
512,
464,
325,
373,
326,
374,
329,
377,
330,
378,
200,
29,
201,
30,
213,
42,
214,
43,
232,
61,
233,
62,
};
QDLDL_float linsys_solver_D[476] = {
-1000000,
5.623433e-01,
-1000000,
2.000000e-06,
-1000000,
5.623433e-01,
-1000000,
2.000000e-06,
-1.000000e+06,
-7.909261e+03,
-1000000,
5.623433e-01,
-1000000,
3.976374e-01,
-10,
1.000010e-01,
-1000000,
-10,
-10,
-1.000000e-02,
-1000000,
5.623433e-01,
-1000000,
3.976374e-01,
-10,
1.000010e-01,
-1000000,
-10,
-10,
-1.000000e-02,
-1000000,
5.623433e-01,
-1000000,
5.623433e-01,
-1000000,
3.976374e-01,
-1000000,
3.976374e-01,
-1000000,
2.000000e-06,
-1000000,
-10,
-10,
-3.445425e+05,
-10,
1.000010e-01,
-10,
1.000010e-01,
-2.000980e+01,
-7.349368e+00,
-3.584393e+00,
6.890899e+01,
-1.160776e+01,
-4.859601e+00,
-2.707477e+00,
1.153462e-01,
1.152594e-01,
6.891174e+01,
-1.059990e+01,
-3.586471e+00,
-2.424892e+00,
-10,
6.623423e-01,
-1000000,
3.976374e-01,
-10,
6.623423e-01,
-1000000,
3.976374e-01,
-4.962847e+00,
-3.047392e+00,
-1000000,
5.623433e-01,
-1000000,
5.623433e-01,
-1000000,
2.000000e-06,
-1000000,
2.000000e-06,
-1.000000e+06,
-7.909261e+03,
-10,
6.623423e-01,
-10,
6.623423e-01,
-1000000,
3.976374e-01,
-1000000,
3.976374e-01,
-4.962847e+00,
-3.047392e+00,
-1000000,
5.623433e-01,
-1000000,
3.976374e-01,
-10,
1.000010e-01,
-1000000,
-10,
-10,
-1.000000e-02,
-1000000,
5.623433e-01,
-1000000,
3.976374e-01,
-10,
1.000010e-01,
-1000000,
-10,
-10,
-1.000000e-02,
-1000000,
5.623433e-01,
-1000000,
5.623433e-01,
-1000000,
3.976374e-01,
-1000000,
3.976374e-01,
-1000000,
2.000000e-06,
-1000000,
-10,
-10,
-3.445425e+05,
-10,
1.000010e-01,
-10,
1.000010e-01,
-2.000980e+01,
-7.349368e+00,
-3.584393e+00,
6.890899e+01,
-1.160776e+01,
-4.859601e+00,
-2.707477e+00,
1.153462e-01,
1.152594e-01,
6.891174e+01,
-1.059990e+01,
-3.586471e+00,
-2.424892e+00,
1.263439e-01,
1.249216e-01,
-1000000,
-10,
-10,
-1.000000e-02,
-1000000,
-10,
-10,
-1.000000e-02,
1.378170e+02,
5.011218e+01,
2.241132e-01,
-1000000,
5.623433e-01,
-1000000,
3.976374e-01,
-10,
1.000010e-01,
-1000000,
5.623433e-01,
-1000000,
3.976374e-01,
-10,
1.000010e-01,
-2.000980e+01,
-7.349368e+00,
-3.584393e+00,
-1000000,
-10,
-10,
-1.000000e-02,
-1000000,
-10,
-10,
-1.000000e-02,
1.378170e+02,
5.011218e+01,
2.241132e-01,
-10,
6.623423e-01,
-1000000,
3.976374e-01,
-10,
6.623423e-01,
-1000000,
3.976374e-01,
-4.962847e+00,
-3.047392e+00,
-1000000,
5.623433e-01,
-1000000,
2.000000e-06,
-1000000,
5.623433e-01,
-1000000,
2.000000e-06,
-1.000000e+06,
-7.909261e+03,
-1000000,
5.623433e-01,
-1000000,
3.976374e-01,
-10,
1.000010e-01,
-1000000,
5.623433e-01,
-1000000,
3.976374e-01,
-10,
1.000010e-01,
-2.000980e+01,
-7.349368e+00,
-3.584393e+00,
-1000000,
-10,
-10,
-1.000000e-02,
-1000000,
-10,
-10,
-1.000000e-02,
1.378170e+02,
5.011218e+01,
2.241132e-01,
-1000000,
5.623433e-01,
-1000000,
3.976374e-01,
-10,
1.000010e-01,
-1000000,
5.623433e-01,
-1000000,
3.976374e-01,
-10,
1.000010e-01,
-2.000980e+01,
-7.349368e+00,
-3.584393e+00,
-1000000,
-10,
-10,
-1.000000e-02,
-1000000,
-10,
-10,
-1.000000e-02,
1.378170e+02,
5.011218e+01,
2.241132e-01,
-1000000,
5.623433e-01,
-1000000,
2.000000e-06,
-1000000,
5.623433e-01,
-1000000,
2.000000e-06,
-1.000000e+06,
-7.909261e+03,
-10,
6.623423e-01,
-1000000,
3.976374e-01,
-10,
6.623423e-01,
-1000000,
3.976374e-01,
-4.962847e+00,
-3.047392e+00,
-1000000,
5.623433e-01,
-1000000,
3.976374e-01,
-10,
1.000010e-01,
-1000000,
5.623433e-01,
-1000000,
3.976374e-01,
-10,
1.000010e-01,
-2.000980e+01,
-7.349368e+00,
-3.584393e+00,
-1000000,
-10,
-10,
-1.000000e-02,
1.001122e+02,
2.242386e-01,
-1000000,
-10,
-10,
-1.000000e-02,
1.001122e+02,
2.242386e-01,
-1000000,
3.976374e-01,
-10,
6.623423e-01,
-1000000,
2.000000e-06,
-1000000,
5.623433e-01,
-1000000,
2.000000e-06,
-1000000,
-1.000000e-02,
1.005623e+02,
-1000000,
-1.000000e-02,
1.000000e+02,
-1000000,
5.623433e-01,
-1.798533e+00,
-5.000000e+05,
-1000000,
3.976374e-01,
-10,
-1.000000e-02,
1.006623e+02,
-1000000,
-1.000000e-02,
1.003976e+02,
-10,
6.623423e-01,
-1.529866e+00,
-2.534509e+00,
-1000000,
-10,
-10,
-1.000000e-02,
-10,
1.000010e-01,
-1000000,
3.976374e-01,
-1000000,
5.623433e-01,
-10,
1.000010e-01,
-1000000,
3.976374e-01,
-1000000,
5.623433e-01,
-10,
-10,
-1000000,
-1.000000e-02,
-10,
-1.000000e-02,
-1000000,
-1.000000e-02,
1.005623e+02,
-1000000,
-1.000000e-02,
1.003976e+02,
1.001000e+02,
1.000000e+02,
-1.689085e-02,
-1.798431e+00,
-1.211419e+00,
-1.001987e+01,
-1.810053e+00,
-1.259325e+00,
-1.002912e+01,
1.097170e+02,
3.730326e+01,
2.240269e-01,
2.090865e-01,
2.006296e-01,
-1000000,
-10,
-10,
-1.000000e-02,
-10,
1.000010e-01,
-1000000,
3.976374e-01,
-1000000,
5.623433e-01,
-10,
1.000010e-01,
-1000000,
3.976374e-01,
-1000000,
5.623433e-01,
-10,
-10,
-1000000,
-1.000000e-02,
-10,
-1.000000e-02,
-1000000,
-1.000000e-02,
1.005623e+02,
-1000000,
-1.000000e-02,
1.003976e+02,
1.001000e+02,
1.000000e+02,
-1.689085e-02,
-1.798431e+00,
-1.211419e+00,
-1.001987e+01,
-1.810053e+00,
-1.259325e+00,
-1.002912e+01,
1.097170e+02,
3.730326e+01,
2.240269e-01,
2.004437e-01,
1.876232e-01,
-1.986891e+00,
-5.000042e+05,
-1.587106e+00,
-6.083108e+00,
6.205630e-02,
6.205630e-02,
-2.065543e+00,
-2.077792e+00,
-9.530238e-01,
-8.886864e-01,
-4.986349e+00,
-2.384059e+00,
-9.027658e+01,
8.005744e-02,
1.561519e-01,
1.556070e-01,
-2.065543e+00,
-2.077792e+00,
-9.530238e-01,
8.005744e-02,
1.522088e-01,
1.510697e-01,
-2.232303e+00,
-1.073553e+00,
-3.031970e+00,
-3.191771e+00,
-1.039573e+02,
-9.637145e-01,
-5.000636e+00,
7.351260e-02,
1.537043e-01,
1.533997e-01,
-2.232303e+00,
-1.073553e+00,
-3.031970e+00,
7.351260e-02,
1.497533e-01,
1.489752e-01,
-1.007858e+00,
-7.003293e-01,
-2.402227e+00,
1.224759e-01,
2.532922e-02,
1.362993e-01,
1.320201e-01,
1.348449e-01,
1.305950e-01,
1.235118e-01,
-4.489705e+00,
-3.326828e+00,
-7.130055e-01,
-1.011431e+00,
-1.052244e+02,
-2.681857e+00,
-5.420305e+00,
-3.332613e+00,
-1.039143e+00,
-1.054809e+02,
-5.355796e+00,
1.245037e-01,
};
QDLDL_int linsys_solver_etree[476] = {
1,
9,
3,
8,
5,
9,
7,
8,
9,
459,
11,
60,
13,
59,
15,
58,
458,
460,
462,
57,
21,
54,
23,
53,
25,
52,
57,
55,
56,
51,
31,
50,
33,
50,
35,
49,
37,
49,
39,
43,
51,
463,
475,
51,
45,
48,
47,
48,
49,
50,
51,
52,
53,
54,
55,
56,
57,
58,
59,
60,
458,
62,
70,
64,
69,
66,
70,
68,
69,
70,
459,
72,
80,
74,
80,
76,
79,
78,
79,
80,
142,
82,
90,
84,
90,
86,
89,
88,
89,
90,
142,
92,
141,
94,
140,
96,
139,
457,
459,
461,
138,
102,
135,
104,
134,
106,
133,
138,
136,
137,
132,
112,
131,
114,
131,
116,
130,
118,
130,
120,
124,
132,
142,
143,
132,
126,
129,
128,
129,
130,
131,
132,
133,
134,
135,
136,
137,
138,
139,
140,
141,
142,
143,
454,
451,
452,
453,
152,
152,
153,
154,
152,
153,
154,
451,
156,
169,
158,
168,
160,
167,
162,
169,
164,
168,
166,
167,
168,
169,
448,
445,
446,
447,
178,
178,
179,
180,
178,
179,
180,
445,
182,
190,
184,
189,
186,
190,
188,
189,
190,
443,
192,
200,
194,
199,
196,
200,
198,
199,
200,
441,
202,
215,
204,
214,
206,
213,
208,
215,
210,
214,
212,
213,
214,
215,
438,
435,
436,
437,
224,
224,
225,
226,
224,
225,
226,
435,
228,
241,
230,
240,
232,
239,
234,
241,
236,
240,
238,
239,
240,
241,
432,
429,
430,
431,
250,
250,
251,
252,
250,
251,
252,
429,
254,
262,
256,
261,
258,
262,
260,
261,
262,
427,
264,
272,
266,
271,
268,
272,
270,
271,
272,
425,
274,
287,
276,
286,
278,
285,
280,
287,
282,
286,
284,
285,
286,
287,
422,
421,
292,
293,
292,
293,
421,
420,
298,
299,
298,
299,
420,
301,
419,
303,
418,
305,
417,
307,
416,
309,
319,
312,
312,
318,
315,
315,
318,
317,
318,
319,
372,
321,
331,
324,
324,
330,
327,
327,
330,
329,
330,
331,
372,
369,
370,
371,
369,
337,
368,
339,
367,
341,
366,
343,
365,
345,
364,
347,
363,
372,
373,
361,
361,
360,
360,
356,
356,
363,
359,
359,
363,
363,
362,
363,
364,
365,
366,
367,
368,
369,
370,
371,
372,
373,
414,
411,
412,
413,
411,
379,
410,
381,
409,
383,
408,
385,
407,
387,
406,
389,
405,
414,
415,
403,
403,
402,
402,
398,
398,
405,
401,
401,
405,
405,
404,
405,
406,
407,
408,
409,
410,
411,
412,
413,
414,
415,
416,
417,
418,
419,
420,
421,
422,
423,
424,
425,
426,
427,
428,
429,
430,
431,
432,
433,
434,
435,
436,
437,
438,
439,
440,
441,
442,
443,
444,
445,
446,
447,
448,
449,
450,
451,
452,
453,
454,
455,
456,
457,
458,
459,
460,
461,
462,
463,
464,
465,
466,
467,
468,
469,
470,
471,
472,
473,
474,
475,
-1,
};
QDLDL_int linsys_solver_Lnz[476] = {
1,
2,
1,
3,
1,
2,
1,
3,
8,
8,
1,
2,
1,
3,
1,
4,
1,
1,
1,
4,
1,
2,
1,
3,
1,
4,
1,
1,
1,
4,
1,
1,
1,
2,
1,
1,
1,
3,
1,
1,
1,
1,
1,
3,
1,
1,
1,
4,
6,
6,
6,
8,
8,
8,
8,
11,
10,
12,
12,
12,
12,
1,
2,
1,
3,
1,
2,
1,
3,
8,
8,
1,
1,
1,
2,
1,
1,
1,
3,
6,
6,
1,
1,
1,
2,
1,
1,
1,
3,
6,
6,
1,
2,
1,
3,
1,
4,
1,
1,
1,
4,
1,
2,
1,
3,
1,
4,
1,
1,
1,
4,
1,
1,
1,
2,
1,
1,
1,
3,
1,
1,
1,
1,
1,
3,
1,
1,
1,
4,
6,
6,
6,
8,
8,
8,
8,
11,
10,
12,
12,
12,
12,
13,
12,
1,
1,
1,
4,
1,
1,
1,
4,
9,
12,
11,
1,
2,
1,
3,
1,
4,
1,
2,
1,
3,
1,
4,
9,
9,
9,
1,
1,
1,
4,
1,
1,
1,
4,
9,
12,
11,
1,
2,
1,
3,
1,
2,
1,
3,
8,
8,
1,
2,
1,
3,
1,
2,
1,
3,
8,
8,
1,
2,
1,
3,
1,
4,
1,
2,
1,
3,
1,
4,
9,
9,
9,
1,
1,
1,
4,
1,
1,
1,
4,
9,
12,
11,
1,
2,
1,
3,
1,
4,
1,
2,
1,
3,
1,
4,
9,
9,
9,
1,
1,
1,
4,
1,
1,
1,
4,
9,
12,
11,
1,
2,
1,
3,
1,
2,
1,
3,
8,
8,
1,
2,
1,
3,
1,
2,
1,
3,
8,
8,
1,
2,
1,
3,
1,
4,
1,
2,
1,
3,
1,
4,
9,
9,
9,
1,
1,
1,
4,
10,
9,
1,
1,
1,
4,
10,
9,
1,
3,
1,
2,
1,
3,
1,
2,
1,
3,
1,
1,
1,
1,
1,
2,
1,
2,
6,
6,
1,
3,
1,
1,
1,
1,
1,
2,
1,
2,
6,
6,
1,
1,
1,
4,
1,
4,
1,
3,
1,
2,
1,
4,
1,
3,
1,
2,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
2,
3,
4,
6,
6,
6,
6,
8,
8,
8,
8,
11,
10,
11,
10,
1,
1,
1,
4,
1,
4,
1,
3,
1,
2,
1,
4,
1,
3,
1,
2,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
2,
3,
4,
6,
6,
6,
6,
8,
8,
8,
8,
11,
10,
13,
12,
12,
12,
12,
12,
12,
12,
16,
15,
14,
17,
16,
17,
16,
16,
15,
14,
16,
15,
14,
14,
13,
12,
16,
15,
14,
17,
16,
17,
16,
16,
15,
14,
16,
15,
14,
14,
13,
12,
19,
18,
17,
16,
17,
16,
15,
14,
13,
12,
11,
10,
9,
8,
7,
6,
5,
4,
3,
2,
1,
0,
};
QDLDL_int   linsys_solver_iwork[1428];
QDLDL_bool  linsys_solver_bwork[476];
QDLDL_float linsys_solver_fwork[476];
qdldl_solver linsys_solver = {QDLDL_SOLVER, &solve_linsys_qdldl, &update_linsys_solver_matrices_qdldl, &update_linsys_solver_rho_vec_qdldl, &linsys_solver_L, linsys_solver_Dinv, linsys_solver_P, linsys_solver_bp, linsys_solver_sol, linsys_solver_rho_inv_vec, (c_float)0.00000100000000000000, 172, 304, linsys_solver_Pdiag_idx, 117, &linsys_solver_KKT, linsys_solver_PtoKKT, linsys_solver_AtoKKT, linsys_solver_rhotoKKT, linsys_solver_D, linsys_solver_etree, linsys_solver_Lnz, linsys_solver_iwork, linsys_solver_bwork, linsys_solver_fwork, };

// Define solution
c_float xsolution[172];
c_float ysolution[304];

OSQPSolution solution = {xsolution, ysolution};

// Define info
OSQPInfo info = {0, "Unsolved", OSQP_UNSOLVED, (c_float)0.0, (c_float)0.0, (c_float)0.0};

// Define workspace
c_float work_rho_vec[304] = {
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)100.00000000000000000000,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.00000100000000000000,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
};
c_float work_rho_inv_vec[304] = {
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)0.01000000000000000021,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)1000000.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
};
c_int work_constr_type[304] = {
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
0,
0,
-1,
-1,
-1,
0,
-1,
-1,
-1,
-1,
-1,
-1,
0,
0,
-1,
-1,
-1,
0,
-1,
-1,
-1,
-1,
-1,
-1,
0,
0,
-1,
-1,
-1,
0,
-1,
-1,
-1,
-1,
-1,
-1,
0,
0,
-1,
-1,
-1,
0,
-1,
-1,
-1,
-1,
-1,
-1,
0,
0,
-1,
-1,
-1,
0,
-1,
-1,
-1,
-1,
-1,
-1,
0,
0,
-1,
-1,
-1,
0,
-1,
-1,
-1,
-1,
-1,
-1,
0,
0,
-1,
-1,
-1,
0,
-1,
-1,
-1,
-1,
-1,
-1,
0,
0,
-1,
-1,
-1,
0,
-1,
-1,
-1,
-1,
-1,
-1,
0,
0,
-1,
-1,
-1,
0,
-1,
-1,
-1,
-1,
-1,
-1,
0,
0,
-1,
-1,
-1,
0,
-1,
-1,
-1,
-1,
-1,
-1,
0,
0,
-1,
-1,
-1,
0,
-1,
-1,
-1,
-1,
-1,
-1,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
};
c_float work_x[172];
c_float work_y[304];
c_float work_z[304];
c_float work_xz_tilde[476];
c_float work_x_prev[172];
c_float work_z_prev[304];
c_float work_Ax[304];
c_float work_Px[172];
c_float work_Aty[172];
c_float work_delta_y[304];
c_float work_Atdelta_y[172];
c_float work_delta_x[172];
c_float work_Pdelta_x[172];
c_float work_Adelta_x[304];
c_float work_D_temp[172];
c_float work_D_temp_A[172];
c_float work_E_temp[304];

OSQPWorkspace workspace = {
&data, (LinSysSolver *)&linsys_solver,
work_rho_vec, work_rho_inv_vec,
work_constr_type,
work_x, work_y, work_z, work_xz_tilde,
work_x_prev, work_z_prev,
work_Ax, work_Px, work_Aty,
work_delta_y, work_Atdelta_y,
work_delta_x, work_Pdelta_x, work_Adelta_x,
work_D_temp, work_D_temp_A, work_E_temp,
&settings, &scaling, &solution, &info};

c_float l_new[304] = {
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)-0.523600,
(c_float)-0.523600,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-1.000000,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-0.523600,
(c_float)-0.523600,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-1.000000,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-0.523600,
(c_float)-0.523600,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-1.000000,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-0.523600,
(c_float)-0.523600,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-1.000000,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-0.523600,
(c_float)-0.523600,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-1.000000,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-0.523600,
(c_float)-0.523600,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-1.000000,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-0.523600,
(c_float)-0.523600,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-1.000000,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-0.523600,
(c_float)-0.523600,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-1.000000,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-0.523600,
(c_float)-0.523600,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-1.000000,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-0.523600,
(c_float)-0.523600,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-1.000000,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-0.523600,
(c_float)-0.523600,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-1.000000,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-INFINITY,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
(c_float)-0.991600,
};

c_float u_new[304] = {
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.000000,
(c_float)0.523600,
(c_float)0.523600,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)0.523600,
(c_float)0.523600,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)0.523600,
(c_float)0.523600,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)0.523600,
(c_float)0.523600,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)0.523600,
(c_float)0.523600,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)0.523600,
(c_float)0.523600,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)0.523600,
(c_float)0.523600,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)0.523600,
(c_float)0.523600,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)0.523600,
(c_float)0.523600,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)0.523600,
(c_float)0.523600,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)0.523600,
(c_float)0.523600,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)INFINITY,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
(c_float)2.408400,
};

c_float x_test[192] = {
// 1
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
// 2
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.08330000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.01600000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)1.67080000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.31900000000000000000,
// 3
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.29820000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.03900000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)2.63010000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.141200000000000000000,
// 4
(c_float)0.000000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.57440000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.044100000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)2.89490000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.037000000000000000000,
// 5
(c_float)0.000000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.81750000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.039900000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)1.96390000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.04760000000000000000,
// 6
(c_float)0.000000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.96370000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.035400000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.95750000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.04340000000000000000,
// 7
(c_float)0.000000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)1.02570000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.031300000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.28050000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.03850000000000000000,
// 8
(c_float)-0.000000000000000000000,
(c_float)0.00000000000000000000,
(c_float)1.03750000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.027600000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.04640000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.03410000000000000000,
// 9
(c_float)-0.000000000000000000000,
(c_float)0.00000000000000000000,
(c_float)1.02820000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.024400000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.13990000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.03010000000000000000,
// 10
(c_float)-0.000000000000000000000,
(c_float)0.00000000000000000000,
(c_float)1.01510000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.021600000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.12190000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.02660000000000000000,
// 11
(c_float)-0.000000000000000000000,
(c_float)0.00000000000000000000,
(c_float)1.00540000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.019100000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.07270000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.02350000000000000000,
// 12
(c_float)0.000000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)1.00020000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.016900000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.03050000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.02080000000000000000,
// 13
(c_float)0.000000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.99840000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.014900000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00550000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.01840000000000000000,
// 14
(c_float)0.000000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.99830000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.013200000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00480000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.01630000000000000000,
// 15
(c_float)0.000000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.99890000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.011700000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00660000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.01440000000000000000,
// 16
(c_float)0.000000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.99950000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.010300000000000000000,
(c_float)-0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00490000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.00000000000000000000,
(c_float)-0.01270000000000000000,
};


void update_and_solve(StateSpace *in, ControlInputs *out) {

    l_new[0] = -(in->x1);
    u_new[0] = -(in->x1);
    l_new[1] = -(in->x2);
    u_new[1] = -(in->x2);
    l_new[2] = -(in->x3);
    u_new[2] = -(in->x3);
    l_new[3] = -(in->x4);
    u_new[3] = -(in->x4);
    l_new[4] = -(in->x5);
    u_new[4] = -(in->x5);
    l_new[5] = -(in->x6);
    u_new[5] = -(in->x6);
    l_new[6] = -(in->x7);
    u_new[6] = -(in->x7);
    l_new[7] = -(in->x8);
    u_new[7] = -(in->x8);
    l_new[8] = -(in->x9);
    u_new[8] = -(in->x9);
    l_new[9] = -(in->x10);
    u_new[9] = -(in->x10);
    l_new[10] = -(in->x11);
    u_new[10] = -(in->x11);
    l_new[11] = -(in->x12);
    u_new[11] = -(in->x12);

    osqp_update_bounds(&workspace, l_new, u_new);

    // Solve Problem
    osqp_solve(&workspace);

    out->u1 = (&workspace)->solution->x[132];
    out->u2 = (&workspace)->solution->x[133];
    out->u3 = (&workspace)->solution->x[134];
    out->u4 = (&workspace)->solution->x[135];

}


///////////////////////////////////////////////////////// int MAIN ///////////////////////////////////////
//int main(int argc, char **argv) {
//
//    struct timeval stop, start;
//    //gettimeofday(&start, NULL);
//    // Solve Problem
//    //osqp_solve(&workspace);
//    //gettimeofday(&stop, NULL);
//    //printf("took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
//
//    int loop;
//
//
////    for(loop = 0; loop < sizeof(udata)/sizeof(*udata); ++loop) {
////        printf("%d --- ", loop);
////        printf("%f \n", udata[loop]);
////    }
//
////    printf("%f \n", (&workspace)->solution->x[132]);
////    printf("%f \n", (&workspace)->solution->x[133]);
////    printf("%f \n", (&workspace)->solution->x[134]);
////    printf("%f \n", (&workspace)->solution->x[135]);
//
////    int init_loop;
////
////    c_float u_min[4] = { (c_float)-0.9916, (c_float)-0.9916, (c_float)-0.9916, (c_float)-0.9916 };
////    c_float u_max[4] = { (c_float)2.4084, (c_float)2.4084, (c_float)2.4084, (c_float)2.4084 };
////    c_float x_min[12] = { (c_float)-0.5236, (c_float)-0.5236, (c_float)-INFINITY, (c_float)-INFINITY, (c_float)-INFINITY, (c_float)-1, (c_float)-INFINITY, (c_float)-INFINITY, (c_float)-INFINITY, (c_float)-INFINITY, (c_float)-INFINITY, (c_float)-INFINITY };
////    c_float x_max[12] = { (c_float)0.5236, (c_float)0.5236, (c_float)INFINITY, (c_float)INFINITY, (c_float)INFINITY, (c_float)INFINITY, (c_float)INFINITY, (c_float)INFINITY, (c_float)INFINITY, (c_float)INFINITY, (c_float)INFINITY, (c_float)INFINITY };
////    int N = 10;
////    int nx = 12;
////    int nu = 4;
////
////    c_float l_new[304] = {(c_float)0.00000000000000000000};
////    c_float u_new[304] = {(c_float)0.00000000000000000000};
////
////    for(init_loop = 0; init_loop < (1+N)*nx; ++init_loop) {
////        l_new[init_loop] = (c_float)0.00000000000000000000;
////        u_new[init_loop] = (c_float)0.00000000000000000000;
////    }
////
////    for(init_loop = (1+N)*nx; init_loop < 2*(1+N)*nx; init_loop = init_loop+nx) {
////        l_new[init_loop] = (c_float)x_min[0];
////        l_new[init_loop+1] = (c_float)x_min[1];
////        l_new[init_loop+2] = (c_float)x_min[2];
////        l_new[init_loop+3] = (c_float)x_min[3];
////        l_new[init_loop+4] = (c_float)x_min[4];
////        l_new[init_loop+5] = (c_float)x_min[5];
////        l_new[init_loop+6] = (c_float)x_min[6];
////        l_new[init_loop+7] = (c_float)x_min[7];
////        l_new[init_loop+8] = (c_float)x_min[8];
////        l_new[init_loop+9] = (c_float)x_min[9];
////        l_new[init_loop+10] = (c_float)x_min[10];
////        l_new[init_loop+11] = (c_float)x_min[11];
////
////        u_new[init_loop] = (c_float)x_max[0];
////        u_new[init_loop+1] = (c_float)x_max[1];
////        u_new[init_loop+2] = (c_float)x_max[2];
////        u_new[init_loop+3] = (c_float)x_max[3];
////        u_new[init_loop+4] = (c_float)x_max[4];
////        u_new[init_loop+5] = (c_float)x_max[5];
////        u_new[init_loop+6] = (c_float)x_max[6];
////        u_new[init_loop+7] = (c_float)x_max[7];
////        u_new[init_loop+8] = (c_float)x_max[8];
////        u_new[init_loop+9] = (c_float)x_max[9];
////        u_new[init_loop+10] = (c_float)x_max[10];
////        u_new[init_loop+11] = (c_float)x_max[11];
////    }
////
////    for(init_loop = 2*(1+N)*nx; init_loop < 2*(1+N)*nx+N*nu; init_loop = init_loop+nu) {
////        l_new[init_loop] = (c_float)u_min[0];
////        l_new[init_loop+1] = (c_float)u_min[1];
////        l_new[init_loop+2] = (c_float)u_min[2];
////        l_new[init_loop+3] = (c_float)u_min[3];
////
////        u_new[init_loop] = (c_float)u_max[0];
////        u_new[init_loop+1] = (c_float)u_max[1];
////        u_new[init_loop+2] = (c_float)u_max[2];
////        u_new[init_loop+3] = (c_float)u_max[3];
////    }
////
////
////    for(loop = 0; loop < sizeof(u_new)/sizeof(u_new[0]); ++loop) {
////        printf("(c_float)%f,\n", l_new[loop]);
////    }
//
//    for(loop = 0; loop < 15; ++loop) {
//
//        l_new[0] = -x_test[loop*12];
//        u_new[0] = -x_test[loop*12];
//        l_new[1] = -x_test[loop*12+1];
//        u_new[1] = -x_test[loop*12+1];
//        l_new[2] = -x_test[loop*12+2];
//        u_new[2] = -x_test[loop*12+2];
//        l_new[3] = -x_test[loop*12+3];
//        u_new[3] = -x_test[loop*12+3];
//        l_new[4] = -x_test[loop*12+4];
//        u_new[4] = -x_test[loop*12+4];
//        l_new[5] = -x_test[loop*12+5];
//        u_new[5] = -x_test[loop*12+5];
//        l_new[6] = -x_test[loop*12+6];
//        u_new[6] = -x_test[loop*12+6];
//        l_new[7] = -x_test[loop*12+7];
//        u_new[7] = -x_test[loop*12+7];
//        l_new[8] = -x_test[loop*12+8];
//        u_new[8] = -x_test[loop*12+8];
//        l_new[9] = -x_test[loop*12+9];
//        u_new[9] = -x_test[loop*12+9];
//        l_new[10] = -x_test[loop*12+10];
//        u_new[10] = -x_test[loop*12+10];
//        l_new[11] = -x_test[loop*12+11];
//        u_new[11] = -x_test[loop*12+11];
//
//        osqp_update_bounds(&workspace, l_new, u_new);
//        osqp_solve(&workspace);
//        printf("%d \n", loop+1);
//        printf("Status:                %s\n", (&workspace)->info->status);
//        printf("%f \n", (&workspace)->solution->x[132]);
//        printf("%f \n", (&workspace)->solution->x[133]);
//        printf("%f \n", (&workspace)->solution->x[134]);
//        printf("%f \n", (&workspace)->solution->x[135]);
//    }
//
//
//    // Print status
////    printf("Status:                %s\n", (&workspace)->info->status);
////    printf("Number of iterations:  %d\n", (int)((&workspace)->info->iter));
////    printf("Objective value:       %.4e\n", (&workspace)->info->obj_val);
////    printf("Primal residual:       %.4e\n", (&workspace)->info->pri_res);
////    printf("Dual residual:         %.4e\n", (&workspace)->info->dua_res);
//
//    return 0;
//}
