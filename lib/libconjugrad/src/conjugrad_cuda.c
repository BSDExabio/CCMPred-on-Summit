#include "conjugrad.h"
#include "conjugrad_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

/*
int init_cg_mem_cuda(int nvar);

int destroy_cg_mem_cuda();
*/

int linesearch_gpu(
	int n,
	conjugrad_float_t *d_x,
	conjugrad_float_t *fx,
	conjugrad_float_t *alpha,
	conjugrad_evaluate_gpu_t proc_evaluate,
	void *instance,
	devicedata *d_instance,
	conjugrad_parameter_t *param
);

// global GPU pointer
/*
conjugrad_float_t *d_g;
conjugrad_float_t *d_s;
*/

int conjugrad_gpu(
	int n,
	conjugrad_float_t *d_x,
	conjugrad_float_t *fx,
	conjugrad_evaluate_gpu_t proc_evaluate,
	conjugrad_progress_t proc_progress,
	void *instance,
	devicedata *dd,
	conjugrad_parameter_t *param
) {
	conjugrad_float_t alpha, alphaprev, dg, dgprev, beta, gnorm, gprevnorm, xnorm;
	// allocate memory on device
//	init_cg_mem_cuda(n);
	int n_linesearch, n_iter = 0;
	int ret = CONJUGRADERR_UNKNOWN;
	conjugrad_float_t *k_last_fx = (conjugrad_float_t *)malloc(sizeof(conjugrad_float_t) * param->k);
	conjugrad_float_t *check_fx = k_last_fx;


	CHECK_ERR(cudaMemset(dd->d_s, 0, sizeof(conjugrad_float_t) * n));

	//call evaluate
	*fx = proc_evaluate(instance, dd, d_x, n);

	// get xnorm and gnorm
	vecnorm_gpu(dd->d_g, &gnorm, n);
	vecnorm_gpu(d_x, &xnorm, n);

	if (gnorm / xnorm <= param->epsilon) {
		ret = CONJUGRAD_ALREADY_MINIMIZED;
		goto conjugrad_exit;
	}

	alpha = F1 / fsqrt(gnorm); // F1 is 1.0

	while (true) {
		if (n_iter >= param->max_iterations) {
			ret = CONJUGRADERR_MAXIMUMITERATION;
			break;
		}



		if (n_iter > 0) {
			// fletcher-reeves: beta_n = ||x_n|| / ||x_{n-1}||
			//                         = ||g_n|| / ||g_{n-1}||
			beta = gnorm / gprevnorm;

			// s_n = \delta x_n + \beta_n * s_{n-1}
			//     = \beta_n * s_{n-1} - g_n
			update_s_gpu(dd->d_s, dd->d_g, beta, n);
			vecdot_gpu(dd->d_s, dd->d_g, &dg, n);
			alpha = alphaprev * dgprev / dg;

		} else {
			// s_0 = \delta x_0
			initialize_s_gpu(dd->d_s, dd->d_g, n);
			vecdot_gpu(dd->d_s, dd->d_g, &dg, n);
		}

		// linesearch
		n_linesearch = linesearch_gpu(n, d_x, fx, &alpha, proc_evaluate, instance, dd, param);

		gprevnorm = gnorm;
		vecnorm_gpu(dd->d_g, &gnorm, n);
		vecnorm_gpu(d_x, &xnorm, n);
		alphaprev = alpha;
		dgprev = dg;

		if(n_linesearch < 0) {
			ret = n_linesearch;
			break;
		}

		int pos = n_iter % param->k;
		check_fx = k_last_fx + pos;
		
		if (n_iter >= param->k) {
			conjugrad_float_t rel_change = (*check_fx - *fx) / *fx;
			if (rel_change < param->epsilon) {
				ret = CONJUGRAD_SUCCESS;
				break;
			}
		}

		*check_fx = *fx;

		n_iter++;
		proc_progress(instance, d_x, dd->d_g, *fx, xnorm, gnorm, alpha, n, n_iter, n_linesearch);

		// convergence check
		//if (xnorm < F1) { xnorm = F1; }
		//if (gnorm / xnorm <= 1e-3f) {
		//	ret = CONJUGRAD_SUCCESS;
		//	break;
		//}
	}
	
	conjugrad_exit:
	
	//destroy_cg_mem_cuda();
	free(k_last_fx);
	return ret;
}


int linesearch_gpu(
	int n,
	conjugrad_float_t *d_x,
	conjugrad_float_t *fx,
	conjugrad_float_t *alpha,
	conjugrad_evaluate_gpu_t proc_evaluate,
	void *instance,
	devicedata *dd,
	conjugrad_parameter_t *param
) {
	conjugrad_float_t fx_step;
	conjugrad_float_t prevalpha;
	prevalpha = F0; // 0.0

	int n_linesearch = 0;

	conjugrad_float_t dginit;
	vecdot_gpu(dd->d_s, dd->d_g, &dginit, n);
	conjugrad_float_t dgtest = dginit * param->ftol;
	conjugrad_float_t dg;
	conjugrad_float_t finit = *fx;

	while (true) {
		if (n_linesearch >= param->max_linesearch) { return CONJUGRADERR_MAXIMUMLINESEARCH; }
		n_linesearch++;

		// do step
		update_x_gpu(d_x, dd->d_s, *alpha, prevalpha, n);

		// evaluate
		fx_step = proc_evaluate(instance, dd, d_x, n);

		// armijo condition
		if (fx_step <= finit + *alpha * dgtest) {
			vecdot_gpu(dd->d_s, dd->d_g, &dg, n);
			if (dg < param->wolfe * dginit) {
				*fx = fx_step;
				return n_linesearch;
			}
		}
		prevalpha = *alpha;
		*alpha *= param->alpha_mul;
	}
}

/*
int init_cg_mem_cuda(int nvar) {
	CHECK_ERR(cudaMalloc((void **) &d_g, sizeof(conjugrad_float_t) * nvar));
	CHECK_ERR(cudaMalloc((void **) &d_s, sizeof(conjugrad_float_t) * nvar));
	return EXIT_SUCCESS;
}


int destroy_cg_mem_cuda() {
	CHECK_ERR(cudaFree(d_g));
	CHECK_ERR(cudaFree(d_s));
	return EXIT_SUCCESS;
}
*/
