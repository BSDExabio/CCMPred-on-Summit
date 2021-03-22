/** Callback for LBFGS optimization to calculate function value and gradient
 * @param[in] instance The user data passed to the LBFGS optimizer
 * @param[in] x The current variable assignments
 * @param[out] g The current gradient
 * @param[in] nvar The number of variables
 * @param[in] step The step size for the current iteration
 */

conjugrad_float_t evaluate_cuda (
	void *instance,
	devicedata *d_instance,
	const conjugrad_float_t *x,
//	conjugrad_float_t *g,
	const int nvar
);


int init_cuda( int nvar_padded, void *instance, 
	       devicedata *d_instance );
int destroy_cuda( devicedata *d_instance );
