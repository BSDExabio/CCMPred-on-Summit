#include <stdbool.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <stdint.h>
#include <libgen.h>
#include <string.h>
#include <locale.h>
#include <math.h>
#include <time.h>

#include "ccmpred.h"
#include "sequence.h"
#include "io.h"
#include "conjugrad.h"
#include "util.h"
#include "parseopt.h"
#include "meta.h"

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "evaluate_cuda.h"
#endif

#ifdef OPENMP
#include <omp.h>
//#include "evaluate_cpu_omp.h"
#endif

#ifdef CURSES
#include <unistd.h>
#include <curses.h>
#include <term.h>
#endif

#include "evaluate_cpu.h"

#define MAXFILES 100


#ifndef _WIN32
// Time measurement
#include <sys/time.h>
#endif

inline double seconds_since( struct timeval *time_start)
{
#ifndef _WIN32
	struct timeval time_end;
	gettimeofday(&time_end,NULL);
        double num_sec     = time_end.tv_sec  - time_start->tv_sec;
        double num_usec    = time_end.tv_usec - time_start->tv_usec;
        return (num_sec + (num_usec/1000000));
#else
	return 0.0;
#endif
}

inline void start_timer(struct timeval *time_start)
{
#ifndef _WIN32
	gettimeofday(time_start,NULL);
#endif
}



/** Callback for LBFGS optimization to show progress
 * @param[in] instance The user data passed to the LBFGS optimizer
 * @param[in] x The current variable assignments
 * @param[in] g The current gradients
 * @param[in] fx The current negative log-likelihood value
 * @param[in] xnorm The euclidean norm of the variables
 * @param[in] gnorm The euclidean norm of the gradients
 * @param[in] step The step size for the current iteration
 * @param[in] nvar The number of variables
 * @param[in] k The number of the current iteration
 * @param[in] ls The number of evaluations called for the current iteration
 */
static int progress(
	void *instance,
	const conjugrad_float_t *x,
	const conjugrad_float_t *g,
	const conjugrad_float_t fx,
	const conjugrad_float_t xnorm,
	const conjugrad_float_t gnorm,
	const conjugrad_float_t step,
	int n,
	int k,
	int ls
) {
	//printf("iter\teval\tf(x)    \t║x║     \t║g║     \tstep\n");
	//printf("%-4d\t%-4d\t%-8g\t%-8g\t%-8.8g\t%-3.3g\n", k, ls, fx, xnorm, gnorm, step);

#ifdef JANSSON
	userdata *ud = (userdata *)instance;
	json_t *meta_steps = (json_t *)ud->meta_steps;

	json_t *ms = json_object();
	json_object_set(ms, "iteration", json_integer(k));
	json_object_set(ms, "eval", json_integer(ls));
	json_object_set(ms, "fx", json_real(fx));
	json_object_set(ms, "xnorm", json_real(xnorm));
	json_object_set(ms, "gnorm", json_real(gnorm));
	json_object_set(ms, "step", json_real(step));

	json_array_append(meta_steps, ms);

#endif

	return 0;
}

/** Init the pairwise emission potentials with single emission potentials as observed in the MSA
 * @param[out] x The matrix to initialize
 * @param[in] ncol The number of columns in the MSA (i.e. L)
 * @param[in] nrow The number of rows in the MSA (i.e. N)
 * @param[in] msa The MSA to read column frequencies from
 */
void init_bias(conjugrad_float_t *x, userdata *ud) {

	int ncol = ud->ncol;
	int nrow = ud->nrow;
	int nsingle = ncol * (N_ALPHA - 1);
	int nsingle_padded = nsingle + N_ALPHA_PAD - (nsingle % N_ALPHA_PAD);
	int nvar_padded = nsingle_padded + ncol * ncol * N_ALPHA * N_ALPHA_PAD;

	conjugrad_float_t *x1 = x;

	//memset(x, 0, sizeof(conjugrad_float_t) * ud->nvar);
	memset(x, 0, sizeof(conjugrad_float_t) * nvar_padded);

	for(int j = 0; j < ncol; j++) {

		int aacounts[N_ALPHA];
		for(int a = 0; a < N_ALPHA; a++) {
			// init with pseudocounts
			aacounts[a] = 1;
		}

		// calculate weights for column
		for(int i = 0; i < nrow; i++) {
			aacounts[X(i,j)]++;
		}

		int aasum = nrow + N_ALPHA;

		conjugrad_float_t aafrac[N_ALPHA];
		for(int a = 0; a < N_ALPHA; a++) {
			aafrac[a] = ((conjugrad_float_t)aacounts[a]) / aasum;
		}

		// we set the weights of the other amino acids relative to the gap column at index 20 (to avoid degeneracy)
		conjugrad_float_t aanorm = flog(aafrac[20]);
		for(int a = 0; a < N_ALPHA - 1; a++) {
			V(j, a) = flog( aafrac[a] ) - aanorm;
		}

	}

}

#ifdef CURSES
/**
 * Detect if we can generate colored output
 */
bool detect_colors() {
	int erret = 0;
	if(setupterm(NULL, 1, &erret) == ERR) {
		return false;
	}

	// colorize if the terminal supports colors and we're writing to a terminal
	return has_colors() && isatty(STDOUT_FILENO);
}
#endif

void logo(bool color) {


	if(color) {
		printf(" _____ _____ _____               _ \n");
		printf("|\x1b[30;42m     |     |     |\x1b[0m___ ___ ___ _|\x1b[30;44m |\x1b[0m\n");
		printf("|\x1b[30;42m   --|   --| | | |\x1b[44m . |  _| -_| . |\x1b[0m\n");
		printf("|\x1b[30;42m_____|_____|_|_|_|\x1b[44m  _|_|\x1b[0m \x1b[30;44m|___|___|\x1b[0m version %s\n", __VERSION);
		printf("                  |\x1b[30;44m_|\x1b[0m\n\n");
	} else {
		printf(" _____ _____ _____               _ \n");
		printf("|     |     |     |___ ___ ___ _| |\n");
		printf("|   --|   --| | | | . |  _| -_| . |\n");
		printf("|_____|_____|_|_|_|  _|_| |___|___|\n");
		printf("                  |_|              \n\n");
	}
}

char* concat(char *s1, char *s2) {
	size_t l1 = strlen(s1);
	size_t l2 = strlen(s2);
	char *result = malloc(l1 + l2 + 1);
	if(result == NULL) {
		die("Cannot malloc new string!");
	}
	memcpy(result, s1, l1);
	memcpy(result+l1, s2, l2+1);
	return result;
}

/**
 * Print a pretty usage message
 * @param[in] exename The name of the executable
 * @param[in] long_usage The length of the usage message to display: 0: short usage, 1: usage and options
 */
void usage(char* exename, int long_usage) {
	printf("Usage: %s [options] input.aln output.mat\n\n", exename);

	if(long_usage) {
		printf("Options:\n");
#ifdef CUDA
		printf("\t-d DEVICE \tCalculate on CUDA device number DEVICE (set to -1 to use CPU) [default: 0]\n");
#endif
#ifdef OPENMP
		printf("\t-t THREADS\tCalculate using THREADS threads on the CPU (automatically disables CUDA if available) [default: 1]\n");
#endif
		printf("\t-n NUMITER\tCompute a maximum of NUMITER operations [default: 50]\n");
		printf("\t-e EPSILON\tSet convergence criterion for minimum decrease in the last K iterations to EPSILON [default: 0.01]\n");
		printf("\t-k LASTK  \tSet K parameter for convergence criterion to LASTK [default: 5]\n");

		printf("\n");
		printf("\t-f FILELIST\tRead list of the sequence files from FILELIST\n");
		printf("\t-i INIFILE\tRead initial weights from INIFILE\n");
		printf("\t-r RAWFILE\tStore raw prediction matrix in RAWFILE\n");

#ifdef MSGPACK
		printf("\t-b BRAWFLE\tStore raw prediction matrix in msgpack format in BRAWFLE\n");
#endif
		printf("\n");
		printf("\t-w IDTHRES\tSet sequence reweighting identity threshold to IDTHRES [default: 0.8]\n");
		printf("\t-l LFACTOR\tSet pairwise regularization coefficients to LFACTOR * (L-1) [default: 0.2]\n");
		printf("\t-A        \tDisable average product correction (APC)\n");
		printf("\t-R        \tRe-normalize output matrix to [0,1]\n");
		printf("\t-h        \tDisplay help\n");

		printf("\n");
		printf("\n\n");
	}

	exit(1);
}

int main(int argc, char **argv)
{
	// Timer initializations
#ifndef _WIN32
	struct timeval time_start, idle_timer;
	start_timer(&time_start);
	start_timer(&idle_timer);
#else
	// Dummy variables if timers off
//	double time_start, idle_timer;
#endif
	double total_setup_time=0;
	double total_resulting_time=0;
	double total_exec_time=0;
//	double idle_time;

	char *rawfilename = NULL;
	int numiter = 250;
	int use_apc = 1;
	int use_normalization = 0;
	conjugrad_float_t lambda_single = F001; // 0.01
	conjugrad_float_t lambda_pair = FInf;
	conjugrad_float_t lambda_pair_factor = F02; // 0.2
	int conjugrad_k = 5;
	conjugrad_float_t conjugrad_eps = 0.01;

	parse_option *optList, *thisOpt;

	char *optstr;
	char *old_optstr = malloc(1);
	old_optstr[0] = 0;
	optstr = concat("f:r:i:n:w:k:e:l:ARh?", old_optstr);
	free(old_optstr);

	int use_def_gpu = 0;
	old_optstr = optstr;
	optstr = concat("d:", optstr);
	free(old_optstr);

#ifdef MSGPACK
	char* msgpackfilename = NULL;
	old_optstr = optstr;
	optstr = concat("b:", optstr);
	free(old_optstr);
#endif

	optList = parseopt(argc, argv, optstr);
	free(optstr);
	
	char* filelistname = NULL;
	char* msafilename = NULL;
	char* matfilename = "out.mat";
	char* initfilename = NULL;

	conjugrad_float_t reweighting_threshold = F08; // 0.8


	//double run_time=0;
	//double total_exec_time=0;
	//double resulting_time=0;
/*
#ifndef _WIN32
                struct timeval setup_timer, exec_timer,  resulting_timer;
#else
               // double setup_timer, exec_timer, resulting_timer;
#endif
*/
	unsigned int numthreads = 1;
	unsigned short use_filelist = 0;

	while(optList != NULL) {
		thisOpt = optList;
		optList = optList->next;

		switch(thisOpt->option) {
#ifdef OPENMP
			case 't':
				numthreads = atoi(thisOpt->argument);

//#ifdef CUDA
//				use_def_gpu = -1; // automatically disable GPU if number of threads specified
//#endif
				break;
#endif
#ifdef CUDA
			case 'd':
				use_def_gpu = atoi(thisOpt->argument);
				break;
#endif
#ifdef MSGPACK
			case 'b':
				msgpackfilename = thisOpt->argument;
				break;
#endif
			case 'f':
				filelistname = thisOpt->argument;
				use_filelist = 1;
				break;
			case 'r':
				rawfilename = thisOpt->argument;
				break;
			case 'i':
				initfilename = thisOpt->argument;
				break;
			case 'n':
				numiter = atoi(thisOpt->argument);
				break;
			case 'w':
				reweighting_threshold = (conjugrad_float_t)atof(thisOpt->argument);
				break;
			case 'l':
				lambda_pair_factor = (conjugrad_float_t)atof(thisOpt->argument);
				break;
			case 'k':
				conjugrad_k = (int)atoi(thisOpt->argument);
				break;
			case 'e':
				conjugrad_eps = (conjugrad_float_t)atof(thisOpt->argument);
				break;
			case 'A':
				use_apc = 0;
				break;
			case 'R':
				use_normalization = 1;
				break;
			case 'h':
			case '?':
				usage(argv[0], 1);
				break;

			case 0:
				if(msafilename == NULL) {
					msafilename = thisOpt->argument;
				} else if(matfilename == NULL) {
					matfilename = thisOpt->argument;
				} 
				break;
			default:
				die("Unknown argument"); 
		}

		free(thisOpt);
	}

	
	char** file_list;
	char** resfile;
	int nfiles;

	if(use_filelist){
		file_list = malloc(MAXFILES*sizeof(char*));
		resfile = malloc(MAXFILES*sizeof(char*));
		nfiles = get_files(filelistname, file_list, resfile);
	}else
	if(msafilename == NULL){
		 usage(argv[0], 0);
	} else {
		file_list = malloc(sizeof(char*));
                resfile = malloc(sizeof(char*));
                nfiles = 1;
		file_list[0] = msafilename;
		resfile[0] = matfilename;
	}

	double exec_time[nfiles];
	double idle_time[nfiles];
	double setup_time[nfiles];
	double result_time[nfiles];
	
#ifdef CURSES
        bool color = detect_colors();
#else
        bool color = false;
#endif
	logo(color);
        printf("\n number of files : %d \n", nfiles);

        int num_devices = 0;
/*#ifdef OPENMP
        num_devices = omp_get_num_devices() ; 
        printf("Found %d CUDA devices, ", num_devices);
#else
*/
        cudaError_t sts = cudaGetDeviceCount(&num_devices);
        if(sts != CUDA_SUCCESS) {
               printf("No CUDA devices available, ");
               cudaDeviceReset();
               exit(-1);
        }
        else {
               printf("Found %d CUDA devices, ", num_devices);
        }
//#//endif

#ifdef OPENMP
	//omp_set_num_threads(numthreads);
        #pragma omp parallel
        {
                int thread_id  = omp_get_thread_num();
                #pragma omp master
                { 
		   numthreads = omp_get_num_threads();
                   printf("\n Using %d openMP threads", numthreads);
#ifdef JANSSON
                   json_object_set(meta_parameters, "cpu_threads", json_integer(numthreads));
#endif 
                }
                #pragma omp barrier
#else
        {
                int thread_id = 0;
#endif


       		//userdata *ud = (userdata *)malloc( sizeof(userdata) );
        	//conjugrad_parameter_t *param = conjugrad_init();
#ifndef _WIN32
                struct timeval setup_timer, exec_timer,  resulting_timer;
#else
//                double setup_timer, exec_timer, resulting_timer;
#endif

#ifdef OPENMP
		#pragma omp for schedule(dynamic, 1)
#endif
	   for(unsigned int job=0; job <nfiles; job++){
		//printf("\n Job %d is assigned to thread %d ", job, thread_id);
		start_timer(&setup_timer);
        	FILE *msafile = fopen(file_list[job], "r");
        	if( msafile == NULL) {
                	printf("Cannot open %s!\n\n", file_list[job]);
                	//return 2;
        	}

#ifdef JANSSON
       	 	char* metafilename = malloc(2048);
        	snprintf(metafilename, 2048, "%s.meta.json", msafilename);
        
        	FILE *metafile = fopen(metafilename, "r");
        	json_t *meta;
        	if(metafile == NULL) {
                // Cannot find .meta.json file - create new empty metadata
                	meta = meta_create();
       		 } else {
               	 // Load metadata from matfile.meta.json
                	meta = meta_read_json(metafile);
                	fclose(metafile);
        	}
        
        	json_object_set(meta, "method", json_string("ccmpred"));
        
        	json_t *meta_step = meta_add_step(meta, "ccmpred");
        	json_object_set(meta_step, "version", json_string(__VERSION));
                
        	json_t *meta_parameters = json_object();
        	json_object_set(meta_step, "parameters", meta_parameters);
                
        	json_t *meta_steps = json_array();
        	json_object_set(meta_step, "iterations", meta_steps);
        
        	json_t *meta_results = json_object();
	        json_object_set(meta_step, "results", meta_results);

#endif
		int ncol, nrow;
		unsigned char* msa = read_msa(msafile, &ncol, &nrow);
		fclose(msafile);

		int nsingle = ncol * (N_ALPHA - 1);
		int nvar = nsingle + ncol * ncol * N_ALPHA * N_ALPHA;
		int nsingle_padded = nsingle + N_ALPHA_PAD - (nsingle % N_ALPHA_PAD);
		int nvar_padded = nsingle_padded + ncol * ncol * N_ALPHA * N_ALPHA_PAD;

#ifdef JANSSON
		json_object_set(meta_parameters, "device", json_string("gpu"));
#endif

		conjugrad_float_t *x = conjugrad_malloc(nvar_padded);
        	if( x == NULL) {
                	die("ERROR: Not enough memory to allocate variables!");
        	}
        	memset(x, 0, sizeof(conjugrad_float_t) * nvar_padded);

        	// Auto-set lambda_pair
        	if(isnan(lambda_pair)) {
                	lambda_pair = lambda_pair_factor * (ncol - 1);
        	}

        	// fill up user data struct for passing to evaluate
       		userdata *ud = (userdata *)malloc( sizeof(userdata) );
        	if(ud == 0) { die("Cannot allocate memory for user data!"); }
        	ud->msa = msa;
        	ud->ncol = ncol;
        	ud->nrow = nrow;
        	ud->nsingle = nsingle;
        	ud->nvar = nvar;
        //	ud->nvar_padded = nvar_padded;
        	ud->lambda_single = lambda_single;
        	ud->lambda_pair = lambda_pair;
        	ud->weights = conjugrad_malloc(nrow);
        	ud->reweighting_threshold = reweighting_threshold;

        	if(initfilename == NULL) {
                	// Initialize emissions to pwm
                	init_bias(x, ud);
        	} else {
                	// Load potentials from file
                	read_raw(initfilename, ud, x);
        	}

        	// optimize with default parameters
        	conjugrad_parameter_t *param = conjugrad_init();

        	param->max_iterations = numiter;
        	param->epsilon = conjugrad_eps;
        	param->k = conjugrad_k;
        	param->max_linesearch = 5;
        	param->alpha_mul = F05;
       	 	param->ftol = 1e-4;
	        param->wolfe = F02;

                unsigned int dev ;  
#ifdef OPENMP
                dev = thread_id%num_devices;  // Can be changed based on the schedule
		cudaError_t status = cudaSetDevice(dev);
                if(cudaSuccess != status) {
                        printf("Error setting device: %d\n", status);
                        exit(-1);
                }
#else
		if(use_def_gpu >= num_devices){
			printf("Requested device %i does not avialable, onle %i devices available. ", use_def_gpu+1, num_devices);
                	exit(-1);
		}
		if (use_def_gpu <0)
			status = cudaFree(NULL);
		else
			status = cudaSetDevice(use_def_gpu);

                if(cudaSuccess != status) {
                        printf("Error setting device: %d\n", status);
                        exit(-1);
                }
#endif
		printf("\n Job %d process the sequence %d using thread %d and device %d \n", job, file_list[job], thread_id, dev);
/*
                struct cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, use_def_gpu);
                printf("using device #%d: %s\n", use_def_gpu, prop.name);

                size_t mem_free, mem_total;
                status = cudaMemGetInfo(&mem_free, &mem_total);
                if(cudaSuccess != status) {
                        printf("Error getting memory info: %d\n", status);
                        exit(1);
                }
                size_t mem_needed = nrow * ncol * 2 + // MSAs
                            sizeof(conjugrad_float_t) * nrow * ncol * 2 + // PC, PCS
                            sizeof(conjugrad_float_t) * nrow * ncol * N_ALPHA_PAD + // PCN
                            sizeof(conjugrad_float_t) * nrow + // Weights
                            (sizeof(conjugrad_float_t) * ((N_ALPHA - 1) * ncol + ncol * ncol * N_ALPHA * N_ALPHA_PAD)) * 4;

                setlocale(LC_NUMERIC, "");
                printf("Total GPU RAM:  %'17lu\n", mem_total);
                printf("Free GPU RAM:   %'17lu\n", mem_free);
                printf("Needed GPU RAM: %'17lu ", mem_needed);

                if(mem_needed <= mem_free) {
                        printf("\n GPU has enough memory for the execution");
                } else {
                        printf("\n GPU memory is not enough for the execution");
               }

#ifdef JANSSON
		json_object_set(meta_parameters, "device", json_string("gpu"));
		json_t* meta_gpu = json_object();
		json_object_set(meta_parameters, "gpu_info", meta_gpu);

		json_object_set(meta_gpu, "name", json_string(prop.name));
		json_object_set(meta_gpu, "mem_total", json_integer(mem_total));
		json_object_set(meta_gpu, "mem_free", json_integer(mem_free));
		json_object_set(meta_gpu, "mem_needed", json_integer(mem_needed));
#endif
*/
        	setup_time[job] = seconds_since(&setup_timer);
#ifdef OPENMP
//	#pragma omp atomic update
#endif
		total_setup_time += setup_time[job];

#ifdef OPENMP
//	#pragma omp critical
//	#pragma omp barrier
#endif
	{

		//printf("\nRunning Job #%d:\n", job);
		idle_time[job] = seconds_since(&idle_timer);
		start_timer(&exec_timer);
		//int (*init)(void *)  = init_cuda;
		//int (*destroy)(void *) = destroy_cuda;
		conjugrad_evaluate_gpu_t evaluate = evaluate_cuda;

       		devicedata dd; // = (devicedata *)malloc( sizeof(devicedata) );
		init_cuda(nvar_padded, ud, &dd);
/*
#ifdef JANSSON
		json_object_set(meta_parameters, "reweighting_threshold", json_real(ud->reweighting_threshold));
		json_object_set(meta_parameters, "apc", json_boolean(use_apc));
		json_object_set(meta_parameters, "normalization", json_boolean(use_normalization));

		json_t *meta_regularization = json_object();
		json_object_set(meta_parameters, "regularization", meta_regularization);

		json_object_set(meta_regularization, "type", json_string("l2")); 
		json_object_set(meta_regularization, "lambda_single", json_real(lambda_single));
		json_object_set(meta_regularization, "lambda_pair", json_real(lambda_pair));
		json_object_set(meta_regularization, "lambda_pair_factor", json_real(lambda_pair_factor));

		json_t *meta_opt = json_object();
		json_object_set(meta_parameters, "optimization", meta_opt);

		json_object_set(meta_opt, "method", json_string("libconjugrad"));
		json_object_set(meta_opt, "float_bits", json_integer((int)sizeof(conjugrad_float_t) * 8));
		json_object_set(meta_opt, "max_iterations", json_integer(param->max_iterations));
		json_object_set(meta_opt, "max_linesearch", json_integer(param->max_linesearch));
		json_object_set(meta_opt, "alpha_mul", json_real(param->alpha_mul));
		json_object_set(meta_opt, "ftol", json_real(param->ftol));
		json_object_set(meta_opt, "wolfe", json_real(param->wolfe));

		json_t *meta_msafile = meta_file_from_path(msafilename);
		json_object_set(meta_parameters, "msafile", meta_msafile);
		json_object_set(meta_msafile, "ncol", json_integer(ncol));
		json_object_set(meta_msafile, "nrow", json_integer(nrow));

		if(initfilename != NULL) {
			json_t *meta_initfile = meta_file_from_path(initfilename);
			json_object_set(meta_parameters, "initfile", meta_initfile);
			json_object_set(meta_initfile, "ncol", json_integer(ncol));
			json_object_set(meta_initfile, "nrow", json_integer(nrow));
		}

		double neff = 0;
		for(int i = 0; i < nrow; i++) {
			neff += ud->weights[i];
		}

		json_object_set(meta_msafile, "neff", json_real(neff));

		ud->meta_steps = meta_steps;

#endif

*/
		printf("\nWill optimize %d %ld-bit variables\n", nvar, sizeof(conjugrad_float_t) * 8);
/*
		if(color) { printf("\x1b[1m"); }
		printf("iter\teval\tf(x)    \t||x||     \t||g||     \tstep\n");
		if(color) { printf("\x1b[0m"); }
*/

		conjugrad_float_t fx;
		int ret;

		conjugrad_float_t *d_x;
//*
		status = cudaMalloc((void **) &d_x, sizeof(conjugrad_float_t) * nvar_padded);
		if (cudaSuccess != status) {
			printf("CUDA error No. %d while allocation memory for d_x\n", status);
			exit(1);
		}
		status = cudaMemcpy(d_x, x, sizeof(conjugrad_float_t) * nvar_padded, cudaMemcpyHostToDevice);
		if (cudaSuccess != status) {
			printf("CUDA error No. %d while copying parameters to GPU\n", status);
			exit(1);
		}
		ret = conjugrad_gpu(nvar_padded, d_x, &fx, evaluate, progress, ud, &dd, param);
		status = cudaMemcpy(x, d_x, sizeof(conjugrad_float_t) * nvar_padded, cudaMemcpyDeviceToHost);
		if (cudaSuccess != status) {
			printf("CUDA error No. %d while copying parameters back to CPU\n", status);
			exit(1);
		}
		/*status = cudaFree(d_x);
		if (cudaSuccess != status) {
			printf("CUDA error No. %d while freeing memory for d_x\n", status);
			exit(1);
		}*/
		exec_time[job] = seconds_since(&exec_timer);
		
		printf("\n %s with status code %d - ", (ret < 0 ? "Exit" : "Done"), ret);

		if(ret == CONJUGRAD_SUCCESS) {
			printf("Success!\n");
		} else if(ret == CONJUGRAD_ALREADY_MINIMIZED) {
			printf("Already minimized!\n");
		} else if(ret == CONJUGRADERR_MAXIMUMITERATION) {
			printf("Maximum number of iterations reached.\n");
		} else {
			printf("Unknown status code!\n");
		}
//*/
		printf("\nFinal fx = %f\n\n", fx);

		status = cudaFree(d_x);
		 if (cudaSuccess != status) {
                        printf("CUDA error No. %d while freeing memory for d_x\n", status);
                        exit(1);
                }
		destroy_cuda(&dd);
		start_timer(&idle_timer);

 	} // end critical region
#ifdef OPENMP
	//#pragma omp barrier
                #pragma omp atomic update
#endif

                total_exec_time += exec_time[job];
//*       
	 // Post-processing ------
		start_timer(&resulting_timer);
                //matfilename = resfile[job];
		FILE* out = fopen(resfile[job], "w");
		if(out == NULL) {
			printf("Cannot open %s for writing!\n\n", matfilename);
			//return 3;
		}

		conjugrad_float_t *outmat = conjugrad_malloc(ncol * ncol);

		FILE *rawfile = NULL;
		if(rawfilename != NULL) {
			printf("Writing raw output to %s\n", rawfilename);
			rawfile = fopen(rawfilename, "w");

			if(rawfile == NULL) {
			printf("Cannot open %s for writing!\n\n", rawfilename);
			//return 4;
			}

			write_raw(rawfile, x, ncol);
		}

#ifdef MSGPACK

		FILE *msgpackfile = NULL;
		if(msgpackfilename != NULL) {
			printf("Writing msgpack raw output to %s\n", msgpackfilename);
			msgpackfile = fopen(msgpackfilename, "w");

			if(msgpackfile == NULL) {
				printf("Cannot open %s for writing!\n\n", msgpackfilename);
				return 4;
			}
#ifndef JANSSON
			void *meta = NULL;
#endif

		}
#endif

		sum_submatrices(x, outmat, ncol);

		if(use_apc) {
			apc(outmat, ncol);
		}

		if(use_normalization) {
			normalize(outmat, ncol);
		}

		write_matrix(out, outmat, ncol, ncol);
//*/
/*
#ifdef JANSSON
		json_object_set(meta_results, "fx_final", json_real(fx));
		json_object_set(meta_results, "num_iterations", json_integer(json_array_size(meta_steps)));
		json_object_set(meta_results, "opt_code", json_integer(ret));

		json_t *meta_matfile = meta_file_from_path(matfilename);
		json_object_set(meta_results, "matfile", meta_matfile);

		if(rawfilename != NULL) {
			json_object_set(meta_results, "rawfile", meta_file_from_path(rawfilename));
		}

#ifdef MSGPACK
		if(msgpackfilename != NULL) {
			json_object_set(meta_results, "msgpackfile", meta_file_from_path(msgpackfilename));
		}
#endif

		fprintf(out, "#>META> %s", json_dumps(meta, JSON_COMPACT));
		if(rawfile != NULL) {
			fprintf(rawfile, "#>META> %s", json_dumps(meta, JSON_COMPACT));
		}
#endif

		if(rawfile != NULL) {
			fclose(rawfile);
		}

#ifdef MSGPACK
		if(msgpackfile != NULL) {
			write_raw_msgpack(msgpackfile, x, ncol, meta);
			fclose(msgpackfile);
		}

#endif
*/
	        result_time[job]=seconds_since(&resulting_timer);
#ifdef OPENMP
	         #pragma omp atomic update
#endif
		total_resulting_time+=result_time[job]; 

		printf("\nJob #%d took %.3f sec for the execution after taking %.3f sec for setup and took %.3f for resulting \n", job, exec_time[job], setup_time[job], result_time[job]);
		fflush(out);
		fclose(out);
		conjugrad_free(outmat);
		conjugrad_free(x);
		free(msa);
		conjugrad_free(ud->weights);
		free(ud);
		free(param);
	}//End of the for loop - Run over the list of files
	} // End of parallel section

//Print the run time

#ifndef _WIN32
	// Total time measurement
	printf("\nSetup time of entire job set (%d files): %.3f sec", nfiles, total_setup_time);
	printf("\nProcessing time of entire job set (%d files): %.3f sec", nfiles, total_exec_time);
	printf("\nResulting time of entire job set (%d files): %.3f sec", nfiles, total_resulting_time);
	printf("\nRun time of entire job set (%d files): %.3f sec", nfiles, seconds_since(&time_start));
	if((numthreads > 1) && use_filelist)	printf("\nSavings from multithreading: %.3f sec ",(total_setup_time+total_resulting_time+total_exec_time) - seconds_since(&time_start));
#endif
	printf("\n");
	//printf("Output can be found in %s\n", matfilename);

	return 0;
}
