//===---- reduction.cu - NVPTX OpenMP reduction implementation ---- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of reduction with KMPC interface.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <complex.h>
#include "omptarget-nvptx.h"

//EXTERN int32_t __gpu__warpBlockRedu_fixed4_add(int32_t);

EXTERN void omp_reduction_op_gpu(char *, char *);


//cannot implement atomic_start and atomic_end for GPU. Report runtime error
EXTERN void __kmpc_atomic_start() {
		printf("__kmpc_atomic_start not supported\n"); \
		asm("trap;"); \
		return; \
}

EXTERN void __kmpc_atomic_end() {
		printf("__kmpc_atomic_end not supported\n"); \
		asm("trap;"); \
		return; \
}

//EXTERN  kmp_ReductFctPtr *gpu_callback = (kmp_ReductFctPtr *)omp_reduction_op_gpu;

EXTERN 
int32_t __gpu_block_reduce(){
        if (omp_get_num_threads() != blockDim.x)
		return 0;
	unsigned tnum = __ballot(1);
	if (tnum != (~0x0)) { //assume swapSize is 32
		return 0;
	}
	return 1;
}

EXTERN 
int32_t __kmpc_reduce_gpu(kmp_Indent *loc, int32_t global_tid, int32_t num_vars, size_t reduce_size, void *reduce_data, kmp_ReductFctPtr *reductFct, kmp_CriticalName *lck) {
  int globalThreadId = GetGlobalThreadId();
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(globalThreadId);
	int numthread;
  	if (currTaskDescr->IsParallelConstruct()) {
		numthread = omp_get_num_threads();
	} else {
		numthread = omp_get_num_teams();
	}
	

	if (numthread == 1)
                return 1;
        else if (!__gpu_block_reduce())
                return 2;
	else {
		if (threadIdx.x == 0)
			return 1;
		else
			return 0;
	}
//	return 2;
	/**
	 * Only when all the threads in a block are doing reduction,
	 * the warpBlockRedu is used. Otherwise atomic.
	 * check the data type, too.
	 * A special case: when the size of thread group is one,
	 * do reduction directly.
	 **/
	
	// Note: this code provokes warning because it follows a "return"
	
	//since there is no thread interface yet, just infer from the 
	// result of ballot
#if 0
	unsigned tnum = __ballot(1);
	if (tnum != (~0x0)) { //assume swapSize is 32
		return 2;
	}
	
#if 0
	if (threadIdx.x == 0) {
		if ((void *)reductFct != (void *)omp_reduction_op_gpu) {
			printf("function pointer value is not correct\n");
		} else {
			printf("function pointer value is correct\n");
		}
	}
#endif

	//printf("function pointer %p %d %p\n", reductFct, reduce_size, omp_reduction_op_gpu);
	if (reduce_size == 0) {
	(*reductFct)((char*)reduce_data, (char*)reduce_data);
	} else {
	//omp_reduction_op_gpu((char*)reduce_data, (char*)reduce_data);
	 (*gpu_callback)((char*)reduce_data, (char*)reduce_data);
	}
	
	//int **myp = (int **) reduce_data;
	// the results are with thread 0. Reduce to the shared one
	if (threadIdx.x == 0) {
		//printf("function pointer %p %p\n", reductFct, omp_reduction_op);
	 //   	printf("my result %d\n", *myp[0]);
		return 1;
	} else {
		return 0;
	}
#endif
}

EXTERN 
int32_t __kmpc_reduce(kmp_Indent *loc, int32_t global_tid, int32_t num_vars, size_t reduce_size, void *reduce_data, kmp_ReductFctPtr *reductFct, kmp_CriticalName *lck) {
	return __kmpc_reduce_gpu(loc, global_tid, num_vars, reduce_size, reduce_data, reductFct, lck);
}

EXTERN 
int32_t __kmpc_reduce_nowait(kmp_Indent *loc, int32_t global_tid, int32_t num_vars, size_t reduce_size, void *reduce_data, kmp_ReductFctPtr *reductFct, kmp_CriticalName *lck) {
  int globalThreadId = GetGlobalThreadId();
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(globalThreadId);
	int numthread;
  	if (currTaskDescr->IsParallelConstruct()) {
		numthread = omp_get_num_threads();
	} else {
		numthread = omp_get_num_teams();
	}

	if (numthread == 1)
                return 1;
        else if (!__gpu_block_reduce())
                return 2;
	else {
		if (threadIdx.x == 0)
			return 1;
		else
			return 0;
	}

	// Notice: as above, uncomment if 0 once this code below is ready for shipping
#if 0
	unsigned tnum = __ballot(1);
	if (tnum != (~0x0)) { //assume swapSize is 32
		return 2;
	}

	if (threadIdx.x == 0) {
		printf("choose block reduction\n");
	}

	(*reductFct)(reduce_data, reduce_data);
	//omp_reduction_op((char*)reduce_data, (char*)reduce_data);

	int **myp = (int **) reduce_data;
	// the results are with thread 0. Reduce to the shared one
	if (threadIdx.x == 0) {
	    	printf("my result %d\n", *myp[0]);
		return 1;
	} else {
		return 0;
	}
#endif
}

EXTERN
void __kmpc_end_reduce( kmp_Indent *loc, int32_t global_tid, kmp_CriticalName *lck ) {
}

EXTERN
void __kmpc_end_reduce_nowait( kmp_Indent *loc, int32_t global_tid, kmp_CriticalName *lck ) {
}




/* implement different data type or operations  with atomicCAS
 */

#define omptarget_nvptx_add(x, y) ((x)+(y))
#define omptarget_nvptx_sub(x, y) ((x)-(y))
#define omptarget_nvptx_sub_rev(y, x) ((x)-(y))
#define omptarget_nvptx_mul(x, y) ((x)*(y))
#define omptarget_nvptx_div(x, y) ((x)/(y))
#define omptarget_nvptx_div_rev(y, x) ((x)/(y))
#define omptarget_nvptx_min(x, y) ((x)>(y)?(y):(x))
#define omptarget_nvptx_max(x, y) ((x)<(y)?(y):(x))
#define omptarget_nvptx_andb(x, y) ((x) & (y))
#define omptarget_nvptx_orb(x, y) ((x) | (y))
#define omptarget_nvptx_xor(x, y) ((x) ^ (y))
#define omptarget_nvptx_shl(x,y) ((x) << (y))
#define omptarget_nvptx_shr(x,y) ((x) >> (y))
#define omptarget_nvptx_andl(x, y) ((x) && (y))
#define omptarget_nvptx_orl(x, y) ((x) || (y))
#define omptarget_nvptx_eqv(x, y) ((x) == (y))
#define omptarget_nvptx_neqv(x, y) ((x) != (y))

#if 0
// keep for debugging 
EXTERN 
void __kmpc_atomic_fixed4_add(kmp_Indent *id_ref, int32_t gtid, int32_t * lhs, int32_t rhs) {
	//if (gtid < 64)
        PRINT(LD_LOOP, "thread %d participating in reduction, lhs = %p, rhs = %d\n", gtid, lhs, rhs);
        atomicAdd(lhs, rhs);
}
#endif


INLINE __device__ float atomicCAS(float *_addr, float _compare, float _val) {
	int *addr = (int *) _addr;
	int compare = __float_as_int(_compare);
	int val = __float_as_int(_val);
	return __int_as_float(atomicCAS(addr, compare, val));
}

INLINE __device__ double atomicCAS(double *_addr, double _compare, double _val) {
	unsigned long long int  *addr = (unsigned long long int *) _addr;
	unsigned long long int  compare = __double_as_longlong(_compare);
	unsigned long long int  val = __double_as_longlong(_val);
	return __longlong_as_double(atomicCAS(addr, compare, val));
}

INLINE __device__ long long int atomicCAS(long long int *_addr, long long int _compare, long long int _val) {
	unsigned long long int  *addr = (unsigned long long int *) _addr;
	unsigned long long int  compare = (unsigned long long int)(_compare);
	unsigned long long int  val = (unsigned long long int)(_val);
	return (long long int) (atomicCAS(addr, compare, val));
}

INLINE __device__ int64_t atomicCAS(int64_t *_addr, int64_t _compare, int64_t _val) {
	unsigned long long int  *addr = (unsigned long long int *) _addr;
	unsigned long long int  compare = (unsigned long long int)(_compare);
	unsigned long long int  val = (unsigned long long int)(_val);
	return (int64_t) (atomicCAS(addr, compare, val));
}

INLINE __device__ uint64_t atomicCAS(uint64_t *_addr, uint64_t _compare, uint64_t _val) {
	unsigned long long int  *addr = (unsigned long long int *) _addr;
	unsigned long long int  compare = (unsigned long long int)(_compare);
	unsigned long long int  val = (unsigned long long int)(_val);
	return (uint64_t) (atomicCAS(addr, compare, val));
}

INLINE __device__ float complex atomicCAS(float complex *_addr, float complex _compare, float complex _val) {
	double  *addr = (double *) _addr;
	double  compare = (double)(_compare);
	double  val = (double)(_val);
	return (float complex) (atomicCAS(addr, compare, val));
}

#define ATOMIC_GENOP_NATIVE(_name, _dtype, _op, _cudaop) \
        EXTERN void __kmpc_atomic_##_name##_##_op\
        (kmp_Indent *id_ref, int32_t gtid, _dtype * lhs, _dtype rhs) { \
	PRINT(LD_LOOP, "Reduction: thead %d\n", gtid); \
	atomic##_cudaop(lhs, rhs); \
} \
  \
        EXTERN _dtype __kmpc_atomic_##_name##_##_op##_cpt\
        (kmp_Indent *id_ref, int32_t gtid, _dtype * lhs, _dtype rhs, int flag) { \
	_dtype old =  atomic##_cudaop(lhs, rhs); \
	if (flag) { \
		return omptarget_nvptx_##_op(old, rhs); \
	} else {\
		return old; \
	} \
}

/*for types that are supported directly by atomicCAS */
#define ATOMIC_GENOP_DIRECT(_name, _dtype, _op) \
        EXTERN void __kmpc_atomic_##_name##_##_op\
        (kmp_Indent *id_ref, int32_t gtid, _dtype * lhs, _dtype rhs) { \
	PRINT(LD_LOOP, "Reduction: thead %d\n", gtid); \
	_dtype *temp_lhs = lhs; \
	_dtype oldvalue = *temp_lhs; \
	_dtype saved ; \
	_dtype newvalue ; \
	do { \
		saved = oldvalue; \
		newvalue = (_dtype)omptarget_nvptx_##_op(saved, rhs); \
		oldvalue = atomicCAS(temp_lhs, saved, newvalue); \
	} while (saved != oldvalue) ;\
} \
  \
        EXTERN _dtype __kmpc_atomic_##_name##_##_op##_cpt\
        (kmp_Indent *id_ref, int32_t gtid, _dtype * lhs, _dtype rhs, int flag) { \
	_dtype *temp_lhs = lhs; \
	_dtype oldvalue = *temp_lhs; \
	_dtype saved ; \
	_dtype newvalue ; \
	do { \
		saved = oldvalue; \
		newvalue = (_dtype)omptarget_nvptx_##_op(saved, rhs); \
		oldvalue = atomicCAS(temp_lhs, saved, newvalue); \
	} while (saved != oldvalue) ;\
	if (flag) return newvalue; \
	else return oldvalue ; \
}

#define ATOMIC_GENOP_DIRECT_REV(_name, _dtype, _op) \
        EXTERN void __kmpc_atomic_##_name##_##_op##_rev\
        (kmp_Indent *id_ref, int32_t gtid, _dtype * lhs, _dtype rhs) { \
	_dtype *temp_lhs = lhs; \
	_dtype oldvalue = *temp_lhs; \
	_dtype saved ; \
	_dtype newvalue ; \
	do { \
		saved = oldvalue; \
		newvalue = (_dtype)omptarget_nvptx_##_op(rhs, saved); \
		oldvalue = atomicCAS(temp_lhs, saved, newvalue); \
	} while (saved != oldvalue) ;\
} \
  \
        EXTERN _dtype __kmpc_atomic_##_name##_##_op##_cpt##_rev\
        (kmp_Indent *id_ref, int32_t gtid, _dtype * lhs, _dtype rhs, int flag) { \
	_dtype *temp_lhs = lhs; \
	_dtype oldvalue = *temp_lhs; \
	_dtype saved ; \
	_dtype newvalue ; \
	do { \
		saved = oldvalue; \
		newvalue = (_dtype)omptarget_nvptx_##_op(rhs, saved); \
		oldvalue = atomicCAS(temp_lhs, saved, newvalue); \
	} while (saved != oldvalue) ;\
	if (flag) return newvalue; \
	else return oldvalue ; \
}


INLINE __device__  void dc_add(double complex *lhs, double complex rhs) {
	double *ptrl = (double *)lhs;
	double *ptrr = (double *) &rhs;
	ptrl[0] += ptrr[0];
	ptrl[1] += ptrr[1];
}

INLINE __device__  void dc_sub(double complex *lhs, double complex rhs) {
	double *ptrl = (double *)lhs;
	double *ptrr = (double *) &rhs;
	ptrl[0] -= ptrr[0];
	ptrl[1] -= ptrr[1];
}

INLINE __device__  void dc_mul(double complex *lhs, double complex rhs) {
	double *ptrl = (double *)lhs;
	double *ptrr = (double *) &rhs;
	double r1 = ptrl[0], r2 = ptrr[0];
	double i1 = ptrl[1], i2 = ptrr[1];
	ptrl[0] = r1*r2-i1*i2;
	ptrl[1] = r1*i2+r2*i1;
}

INLINE __device__  void dc_div(double complex *lhs, double complex rhs) {
	double *ptrl = (double *)lhs;
	double *ptrr = (double *) &rhs;
	double r1 = ptrl[0], r2 = ptrr[0];
	double i1 = ptrl[1], i2 = ptrr[1];
	ptrl[0] = (r1*r2+i1*i2)/(r2*r2+i2*i2);
	ptrl[1] = (i1*r2-r1*i2)/(r2*r2+i2*i2);
}

#define ATOMIC_GENOP_DC(_op) \
        EXTERN void __kmpc_atomic_cmplx8_##_op\
        (kmp_Indent *id_ref, int32_t gtid, double _Complex * lhs, double _Complex rhs) { \
		printf("double complex atomic opertion not supported\n"); \
		asm("trap;"); \
		return; \
	}\
	EXTERN double _Complex __gpu_warpBlockRedu_cmplx8_##_op(double _Complex rhs) { \
	__shared__ double _Complex lhs; \
	if (threadIdx.x == 0 ) \
		lhs = rhs; \
	__syncthreads(); \
	for (int i= 1; i<blockDim.x; i++)  { \
		if (threadIdx.x == i) { \
			dc_##_op(&lhs, rhs); \
		}\
		__syncthreads(); \
	}\
	return lhs; \
}

//implementation with shared
#define ATOMIC_GENOP_DC_obsolete(_op) \
        EXTERN void __kmpc_atomic_cmplx16_##_op\
        (kmp_Indent *id_ref, int32_t gtid, double _Complex * lhs, double _Complex rhs) { \
	__shared__ unsigned int  stepinblock; \
	unsigned tnum = __ballot(1); \
	if (tnum != (~0x0)) {  \
		return ; \
	} \
	if (threadIdx.x == 0)  stepinblock = 0; \
	__syncthreads(); \
	while (stepinblock < blockDim.x) { \
		if (threadIdx.x == stepinblock) { \
		dc_##_op(lhs, rhs); \
		stepinblock ++ ; \
		}\
		__syncthreads(); \
	} \
} 

ATOMIC_GENOP_DC(add);
ATOMIC_GENOP_DC(sub);
ATOMIC_GENOP_DC(mul);
ATOMIC_GENOP_DC(div);


INLINE __device__  uint64_t fc_add(float r1, float i1, float r2, float i2) {
	uint64_t result;
	float *rr = (float *)&result;
	float *ri = rr+1;
	*rr = r1+r2;
	*ri = i1+i2;
	return result;
}

INLINE __device__  uint64_t fc_sub(float r1, float i1, float r2, float i2) {
	uint64_t result;
	float *rr = (float *)&result;
	float *ri = rr+1;
	*rr = r1-r2;
	*ri = i1-i2;
	return result;
}

INLINE __device__  uint64_t fc_mul(float r1, float i1, float r2, float i2) {
	uint64_t result;
	float *rr = (float *)&result;
	float *ri = rr+1;
	*rr = r1*r2-i1*i2;
	*ri = r1*i2+r2*i1;
	return result;
}

INLINE __device__  uint64_t fc_div(float r1, float i1, float r2, float i2) {
	uint64_t result;
	float *rr = (float *)&result;
	float *ri = rr+1;
	*rr = (r1*r2+i1*i2)/(r2*r2+i2*i2);
	*ri = (i1*r2-r1*i2)/(r2*r2+i2*i2);
	return result;
}

#define ATOMIC_GENOP_FC(_op) \
        EXTERN void __kmpc_atomic_cmplx4_##_op\
        (kmp_Indent *id_ref, int32_t gtid, float complex * lhs, float complex rhs) { \
	uint64_t *temp_lhs = (uint64_t *)lhs; \
	uint64_t oldvalue = *temp_lhs; \
	uint64_t saved ; \
	float *pr1 = (float *) &rhs; \
	float *pi1 = pr1+1; \
	float r1 = *pr1 ; \
	float i1 = *pi1 ; \
	uint64_t newvalue ; \
	do { \
		saved = oldvalue; \
		float *pr2 = (float *) &saved ; \
		float *pi2 = pr2+1; \
		newvalue = fc_##_op(*pr2, *pi2, r1, i1); \
		oldvalue = atomicCAS(temp_lhs, saved, newvalue); \
	} while (saved != oldvalue) ;\
} \
\
        EXTERN void  __kmpc_atomic_cmplx4_##_op##_cpt\
        (kmp_Indent *id_ref, int32_t gtid, float complex * lhs, float complex rhs, float complex *outp, int flag) { \
	uint64_t *temp_lhs = (uint64_t *)lhs; \
	uint64_t oldvalue = *temp_lhs; \
	uint64_t saved ; \
	float *pr1 = (float *) &rhs; \
	float *pi1 = pr1+1; \
	float r1 = *pr1 ; \
	float i1 = *pi1 ; \
	uint64_t newvalue ; \
	do { \
		saved = oldvalue; \
		float *pr2 = (float *) &saved ; \
		float *pi2 = pr2+1; \
		newvalue = fc_##_op(*pr2, *pi2, r1, i1); \
		oldvalue = atomicCAS(temp_lhs, saved, newvalue); \
	} while (saved != oldvalue) ;\
	if (flag) {float complex *temp = (float complex *)&newvalue; *outp =  *temp; } \
	else {float complex *temp = (float complex *) &saved; *outp =  *temp; } \
}

#define ATOMIC_GENOP_FC_REV(_op) \
        EXTERN void __kmpc_atomic_cmplx4_##_op##_rev\
        (kmp_Indent *id_ref, int32_t gtid, float complex * lhs, float complex rhs) { \
	uint64_t *temp_lhs = (uint64_t *)lhs; \
	uint64_t oldvalue = *temp_lhs; \
	uint64_t saved ; \
	float *pr1 = (float *) &rhs; \
	float *pi1 = pr1+1; \
	float r1 = *pr1 ; \
	float i1 = *pi1 ; \
	uint64_t newvalue ; \
	do { \
		saved = oldvalue; \
		float *pr2 = (float *) &saved ; \
		float *pi2 = pr2+1; \
		newvalue = fc_##_op(r1, i1, *pr2, *pi2); \
		oldvalue = atomicCAS(temp_lhs, saved, newvalue); \
	} while (saved != oldvalue) ;\
} \
\
        EXTERN void  __kmpc_atomic_cmplx4_##_op##_cpt##_rev\
        (kmp_Indent *id_ref, int32_t gtid, float complex * lhs, float complex rhs, float complex *outp, int flag) { \
	uint64_t *temp_lhs = (uint64_t *)lhs; \
	uint64_t oldvalue = *temp_lhs; \
	uint64_t saved ; \
	float *pr1 = (float *) &rhs; \
	float *pi1 = pr1+1; \
	float r1 = *pr1 ; \
	float i1 = *pi1 ; \
	uint64_t newvalue ; \
	do { \
		saved = oldvalue; \
		float *pr2 = (float *) &saved ; \
		float *pi2 = pr2+1; \
		newvalue = fc_##_op(r1, i1, *pr2, *pi2); \
		oldvalue = atomicCAS(temp_lhs, saved, newvalue); \
	} while (saved != oldvalue) ;\
	if (flag) {float complex *temp = (float complex *)&newvalue; *outp =  *temp; } \
	else {float complex *temp = (float complex *) &saved; *outp =  *temp; } \
}


ATOMIC_GENOP_FC(add);
ATOMIC_GENOP_FC(sub);
ATOMIC_GENOP_FC_REV(sub);
ATOMIC_GENOP_FC(mul);
ATOMIC_GENOP_FC(div);
ATOMIC_GENOP_FC_REV(div);

/* for test */
//ATOMIC_GENOP_DIRECT(fixed4, int32_t, add);

//for int and unit
#define ATOMIC_GENOP_ALL_MIXED(_name, _dirname, _tname, _optype) \
	_dirname(_tname, _optype, add, Add) ;\
	_dirname(_tname, _optype, sub, Sub) ;\
	_name##_REV(_tname, _optype, sub) ;\
	_name(_tname, _optype, mul) ;\
	_name(_tname, _optype, div) ;\
	_name##_REV(_tname, _optype, div) ;\
	_dirname(_tname, _optype, min, Min) ;\
	_dirname(_tname, _optype, max, Max) ;\
	_dirname(_tname, _optype, andb, And) ;\
	_dirname(_tname, _optype, orb, Or) ;\
	_dirname(_tname, _optype, xor, Xor) ;\
	_name(_tname, _optype, shl) ;\
	_name(_tname, _optype, shr) ;\
	_name(_tname, _optype, andl) ;\
	_name(_tname, _optype, orl) ; \
	_name(_tname, _optype, eqv) ; \
	_name(_tname, _optype, neqv) ;


#define ATOMIC_GENOP_ALL_MIXED_FIXED8U(_name, _dirname, _tname, _optype) \
	_dirname(_tname, _optype, add, Add) ;\
	_name(_tname, _optype, sub) ;\
	_name##_REV(_tname, _optype, sub) ;\
	_name(_tname, _optype, mul) ;\
	_name(_tname, _optype, div) ;\
	_name##_REV(_tname, _optype, div) ;\
	_dirname(_tname, _optype, min, Min) ;\
	_dirname(_tname, _optype, max, Max) ;\
	_dirname(_tname, _optype, andb, And) ;\
	_dirname(_tname, _optype, orb, Or) ;\
	_dirname(_tname, _optype, xor, Xor) ;\
	_name(_tname, _optype, shl) ;\
	_name(_tname, _optype, shr) ;\
	_name(_tname, _optype, andl) ;\
	_name(_tname, _optype, orl) ; \
	_name(_tname, _optype, eqv) ; \
	_name(_tname, _optype, neqv) ;

#define ATOMIC_GENOP_ALL(_name, _tname, _optype) \
	_name(_tname, _optype, add) ;\
	_name(_tname, _optype, sub) ;\
	_name##_REV(_tname, _optype, sub) ;\
	_name(_tname, _optype, mul) ;\
	_name(_tname, _optype, div) ;\
	_name##_REV(_tname, _optype, div) ;\
	_name(_tname, _optype, min) ;\
	_name(_tname, _optype, max) ;\
	_name(_tname, _optype, andb) ;\
	_name(_tname, _optype, orb) ;\
	_name(_tname, _optype, xor) ;\
	_name(_tname, _optype, shl) ;\
	_name(_tname, _optype, shr) ;\
	_name(_tname, _optype, andl) ;\
	_name(_tname, _optype, orl) ; \
	_name(_tname, _optype, eqv) ; \
	_name(_tname, _optype, neqv) ; \

#define ATOMIC_GENOP_FLOAT(_name, _tname, _optype) \
	_name(_tname, _optype, add) ;\
	_name(_tname, _optype, sub) ;\
	_name##_REV(_tname, _optype, sub) ;\
	_name(_tname, _optype, mul) ;\
	_name(_tname, _optype, div) ;\
	_name##_REV(_tname, _optype, div) ;\
	_name(_tname, _optype, min) ;\
	_name(_tname, _optype, max) ;




//ATOMIC_GENOP_ALL(ATOMIC_GENOP_DIRECT, fixed4, int32_t);
//ATOMIC_GENOP_ALL(ATOMIC_GENOP_DIRECT, fixed4u, uint32_t);
ATOMIC_GENOP_ALL_MIXED(ATOMIC_GENOP_DIRECT, ATOMIC_GENOP_NATIVE, fixed4, int32_t);
ATOMIC_GENOP_ALL_MIXED(ATOMIC_GENOP_DIRECT, ATOMIC_GENOP_NATIVE, fixed4u, uint32_t);


ATOMIC_GENOP_ALL(ATOMIC_GENOP_DIRECT, fixed8, int64_t);

ATOMIC_GENOP_ALL(ATOMIC_GENOP_DIRECT, fixed8u, uint64_t);
//ATOMIC_GENOP_ALL_MIXED_FIXED8U(ATOMIC_GENOP_DIRECT, ATOMIC_GENOP_NATIVE, fixed8u, uint64_t);

ATOMIC_GENOP_FLOAT(ATOMIC_GENOP_DIRECT, float4, float);
ATOMIC_GENOP_FLOAT(ATOMIC_GENOP_DIRECT, float8, double);


/*******************************************************
 * data type of size not 32 nor 64
 *******************************************************/

typedef enum {
  omptarget_nvptx_inc,
  omptarget_nvptx_dec,
  omptarget_nvptx_add,
  omptarget_nvptx_sub,
  omptarget_nvptx_sub_rev,
  omptarget_nvptx_mul,
  omptarget_nvptx_div,
  omptarget_nvptx_div_rev,
  omptarget_nvptx_min,
  omptarget_nvptx_max,
  omptarget_nvptx_rd,
  omptarget_nvptx_wr,
  omptarget_nvptx_swp,
  omptarget_nvptx_andb,
  omptarget_nvptx_orb,
  omptarget_nvptx_xor,
  omptarget_nvptx_andl,
  omptarget_nvptx_orl,
  omptarget_nvptx_eqv,
  omptarget_nvptx_neqv,
  omptarget_nvptx_shl,
  omptarget_nvptx_shl_rev,
  omptarget_nvptx_shr,
  omptarget_nvptx_shr_rev,
} omptarget_nvptx_BINOP_t;



template <
  typename OpType,            // type of the operation performed
  omptarget_nvptx_BINOP_t binop          // enum describing the operation
>
INLINE __device__ OpType Compute(OpType a, OpType b) // a is old value, b is new value
{
  OpType res = 0;
  if (binop == omptarget_nvptx_inc)     res = a + b;
  if (binop == omptarget_nvptx_dec)     res = a - b;
  if (binop == omptarget_nvptx_add)     res = a + b;
  if (binop == omptarget_nvptx_sub)     res = a - b;
  if (binop == omptarget_nvptx_sub_rev) res = b - a;
  if (binop == omptarget_nvptx_mul)     res = a * b;
  if (binop == omptarget_nvptx_div)     res = a / b;
  if (binop == omptarget_nvptx_div_rev) res = b / a;
  if (binop == omptarget_nvptx_min)     res = a < b ? a : b;
  if (binop == omptarget_nvptx_max)     res = a > b ? a : b;
  if (binop == omptarget_nvptx_rd)      res = a; // read
  if (binop == omptarget_nvptx_wr)      res = b; // write and swap are the same
  if (binop == omptarget_nvptx_swp)     res = b; // write and swap are the same
  if (binop == omptarget_nvptx_andb)    res = a &  b;
  if (binop == omptarget_nvptx_orb)     res = a |  b;
  if (binop == omptarget_nvptx_xor)     res = a ^  b;
  if (binop == omptarget_nvptx_andl)    res = a && b;
  if (binop == omptarget_nvptx_orl)     res = a || b;
  if (binop == omptarget_nvptx_eqv)     res = a == b;
  if (binop == omptarget_nvptx_neqv)    res = a != b;
  if (binop == omptarget_nvptx_shl)     res = a << b;
  if (binop == omptarget_nvptx_shl_rev) res = b << a;
  if (binop == omptarget_nvptx_shr)     res = a >> b;
  if (binop == omptarget_nvptx_shr_rev) res = b >> a;

  return res;
}


template<> 
INLINE __device__ float Compute<float, omptarget_nvptx_add>(float a, float b) 
{
	return a+b;
}

template<> 
INLINE __device__ float Compute<float, omptarget_nvptx_sub>(float a, float b) 
{
	return a-b;
}

template<> 
INLINE __device__ float Compute<float, omptarget_nvptx_mul>(float a, float b) 
{
	return a*b;
}

template<> 
INLINE __device__ float Compute<float, omptarget_nvptx_div>(float a, float b) 
{
	return a/b;
}

template<> 
INLINE __device__ float Compute<float, omptarget_nvptx_min>(float a, float b) 
{
	return a<b?a:b;
}

template<> 
INLINE __device__ float Compute<float, omptarget_nvptx_max>(float a, float b) 
{
	return a>b?a:b;
}

template<> 
INLINE __device__ double Compute<double, omptarget_nvptx_add>(double a, double b) 
{
	return a+b;
}

template<> 
INLINE __device__ double Compute<double, omptarget_nvptx_sub>(double a, double b) 
{
	return a-b;
}

template<> 
INLINE __device__ double Compute<double, omptarget_nvptx_mul>(double a, double b) 
{
	return a*b;
}

template<> 
INLINE __device__ double Compute<double, omptarget_nvptx_div>(double a, double b) 
{
	return a/b;
}

template<> 
INLINE __device__ double Compute<double, omptarget_nvptx_min>(double a, double b) 
{
	return a<b?a:b;
}

template<> 
INLINE __device__ double Compute<double, omptarget_nvptx_max>(double a, double b) 
{
	return a>b?a:b;
}

#if 0
template <
  omptarget_nvptx_BINOP_t binop          // enum describing the operation
>
INLINE __device__ float Compute<float, binop>(float a, float b) // a is old value, b is new value
{
  OpType res = 0;
  if (binop == omptarget_nvptx_add)     res = a + b;
  if (binop == omptarget_nvptx_sub)     res = a - b;
  if (binop == omptarget_nvptx_mul)     res = a * b;
  if (binop == omptarget_nvptx_div)     res = a / b;
  if (binop == omptarget_nvptx_min)     res = a < b ? a : b;
  if (binop == omptarget_nvptx_max)     res = a > b ? a : b;
  return res;
}
#endif


////////////////////////////////////////////////////////////////////////////////
// common atomic slicing functions (modifying only a part of a word)
////////////////////////////////////////////////////////////////////////////////

template <
  typename MemType,      // type of the underlying atomic memory operation 
  typename OpType        // type of the operation performed
>
INLINE __device__ void ComputeAtomic_PrepareSlice(
  OpType *addr,          // original address
  MemType **memAddrPtr,           // truncated address to MemType boundary
  MemType *memBitShiftRightPtr,   // bits to shift to move val to rightmost position
  MemType *memValMaskInPlacePtr)  // mask of val in proper position
{
  //ASSERT(LT_FUSSY, sizeof(OpType)<sizeof(MemType), "expected smaller type");
  // compute the mask that corresponds to the natural alignment of memType
  // int -> 0x3; long long -> 0x7
  unsigned long memAddrMask = sizeof(MemType) - 1;
  // compute the addr of the atomic variable truncated to alignment of memType
  *memAddrPtr = (MemType *) 
    ((unsigned long) addr & ~ memAddrMask);
  // compute the number of bit shift to move the target atomic value in 
  // the rightmost position
  unsigned long byteOffsetInMem = (unsigned long) addr & memAddrMask;

  /* for big-endian */
  //unsigned long lastByteOffsetInMem = byteOffsetInMem + sizeof(OpType) -1;
//  unsigned long byteShiftRight = (sizeof(MemType) - 1) - lastByteOffsetInMem;
  //*memBitShiftRightPtr = (MemType) (byteShiftRight << 3);  // 3: byte to bits


   /* for little-endian */
  unsigned long byteShiftRight = byteOffsetInMem;
  *memBitShiftRightPtr = (MemType) (byteShiftRight << 3);  // 3: byte to bits

  // mask to isolate target atomic value located in rightmost position
  MemType memValMask = ((MemType) 1 << (sizeof(OpType) << 3)) -1;  
  // mask to isolate target atomic value located in place
  *memValMaskInPlacePtr = memValMask << *memBitShiftRightPtr;
/*
  printf( 
    "Atomic of size %d in mem size %d: addr 0x%llx, truncated addr 0x%llx, shift right %lld, mask in place 0x%llx\n",
    sizeof(MemType), sizeof(OpType), (uint64_t) addr, (uint64_t) *memAddrPtr, 
    (uint64_t) *memBitShiftRightPtr, (uint64_t) *memValMaskInPlacePtr);
*/
}

template <
  typename MemType,      // type of the underlying atomic memory operation 
  typename OpType,       // type of the operation performed
  omptarget_nvptx_BINOP_t binop     // enum describing the operation
>
INLINE __device__  MemType  ComputeAtomic_ComputeSlice(
  MemType oldMemVal,          // old value
  OpType val,                 // value to compute with
  MemType memBitShiftRight,   // bits to shift to move val to rightmost position
  MemType memValMaskInPlace  // mask of val in proper position 
  ) 
{
  OpType oldValtmp;
  OpType newValtmp;
  // select target atomic val
  MemType oldMemVal_targetVal = oldMemVal &   memValMaskInPlace;
  MemType oldMemVal_otherVal  = oldMemVal & ~ memValMaskInPlace;
  // shift target atomic val to rightmost place: this is the old value

  //type conversion??
  oldValtmp = (OpType) (oldMemVal_targetVal >> memBitShiftRight);
  // perform op

  newValtmp = Compute<OpType, binop>(oldValtmp, val);

  // insert new value in old world mem

   //type conversion??
  MemType newMemVal_targetVal = ((MemType) newValtmp) << memBitShiftRight;
  newMemVal_targetVal &= memValMaskInPlace;
  MemType newMemVal = oldMemVal_otherVal | newMemVal_targetVal;
  return newMemVal;
}



#define ATOMIC_GENOP_PARTIAL(_name, _dtype, _op, _memType) \
        EXTERN void __kmpc_atomic_##_name##_##_op\
        (kmp_Indent *id_ref, int32_t gtid, _dtype * lhs, _dtype rhs) { \
      _memType *memAddr ; \
      _memType memBitShiftRightPtr; \
      _memType memValMaskInPlacePtr; \
      ComputeAtomic_PrepareSlice<_memType, _dtype> \
        (lhs, &memAddr, &memBitShiftRightPtr, &memValMaskInPlacePtr); \
      _memType oldMemVal, newMemVal; \
      oldMemVal = *memAddr; \
      _memType savedMemVal; \
      do { \
        savedMemVal = oldMemVal; \
        newMemVal = ComputeAtomic_ComputeSlice  <_memType, _dtype, omptarget_nvptx_##_op > \
          (oldMemVal, rhs, memBitShiftRightPtr, memValMaskInPlacePtr); \
	oldMemVal = atomicCAS(memAddr, savedMemVal, newMemVal); \
      } while (savedMemVal != oldMemVal); \
} \
\
        EXTERN _dtype __kmpc_atomic_##_name##_##_op##_cpt\
        (kmp_Indent *id_ref, int32_t gtid, _dtype * lhs, _dtype rhs, int flag) { \
      _memType *memAddr ; \
      _memType memBitShiftRightPtr; \
      _memType memValMaskInPlacePtr; \
      ComputeAtomic_PrepareSlice<_memType, _dtype> \
        (lhs, &memAddr, &memBitShiftRightPtr, &memValMaskInPlacePtr); \
      _memType oldMemVal, newMemVal; \
      oldMemVal = *memAddr; \
      _memType savedMemVal; \
      do { \
        savedMemVal = oldMemVal; \
        newMemVal = ComputeAtomic_ComputeSlice  <_memType, _dtype, omptarget_nvptx_##_op > \
          (oldMemVal, rhs, memBitShiftRightPtr, memValMaskInPlacePtr); \
	oldMemVal = atomicCAS(memAddr, savedMemVal, newMemVal); \
      } while (savedMemVal != oldMemVal); \
      if (flag) \
	return (_dtype) ((newMemVal & memValMaskInPlacePtr) >> memBitShiftRightPtr); \
      else  \
	return (_dtype) ((oldMemVal & memValMaskInPlacePtr) >> memBitShiftRightPtr); \
}


#define ATOMIC_GENOP_PARTIAL_REV(_name, _dtype, _op, _memType) \
        EXTERN void __kmpc_atomic_##_name##_##_op##_rev\
        (kmp_Indent *id_ref, int32_t gtid, _dtype * lhs, _dtype rhs) { \
      _memType *memAddr ; \
      _memType memBitShiftRightPtr; \
      _memType memValMaskInPlacePtr; \
      ComputeAtomic_PrepareSlice<_memType, _dtype> \
        (lhs, &memAddr, &memBitShiftRightPtr, &memValMaskInPlacePtr); \
      _memType oldMemVal, newMemVal; \
      oldMemVal = *memAddr; \
      _memType savedMemVal; \
      do { \
        savedMemVal = oldMemVal; \
        newMemVal = ComputeAtomic_ComputeSlice  <_memType, _dtype, omptarget_nvptx_##_op > \
          (oldMemVal, rhs, memBitShiftRightPtr, memValMaskInPlacePtr); \
	oldMemVal = atomicCAS(memAddr, savedMemVal, newMemVal); \
      } while (savedMemVal != oldMemVal); \
} \
\
        EXTERN _dtype __kmpc_atomic_##_name##_##_op##_cpt_rev\
        (kmp_Indent *id_ref, int32_t gtid, _dtype * lhs, _dtype rhs, int flag) { \
      _memType *memAddr ; \
      _memType memBitShiftRightPtr; \
      _memType memValMaskInPlacePtr; \
      ComputeAtomic_PrepareSlice<_memType, _dtype> \
        (lhs, &memAddr, &memBitShiftRightPtr, &memValMaskInPlacePtr); \
      _memType oldMemVal, newMemVal; \
      oldMemVal = *memAddr; \
      _memType savedMemVal; \
      do { \
        savedMemVal = oldMemVal; \
        newMemVal = ComputeAtomic_ComputeSlice  <_memType, _dtype, omptarget_nvptx_##_op > \
          (oldMemVal, rhs, memBitShiftRightPtr, memValMaskInPlacePtr); \
	oldMemVal = atomicCAS(memAddr, savedMemVal, newMemVal); \
      } while (savedMemVal != oldMemVal); \
      if (flag) \
	return (_dtype) ((newMemVal & memValMaskInPlacePtr) >> memBitShiftRightPtr); \
      else  \
	return (_dtype) ((oldMemVal & memValMaskInPlacePtr) >> memBitShiftRightPtr); \
}



#define ATOMIC_GENOP_ALL4(_name, _tname, _optype, _memtype) \
	_name(_tname, _optype, add, _memtype); \
	_name(_tname, _optype, sub, _memtype); \
	_name##_REV(_tname, _optype, sub_rev, _memtype); \
	_name(_tname, _optype, mul, _memtype); \
	_name(_tname, _optype, div, _memtype); \
	_name##_REV(_tname, _optype, div_rev, _memtype); \
	_name(_tname, _optype, min, _memtype); \
	_name(_tname, _optype, max, _memtype); \
	_name(_tname, _optype, andb, _memtype); \
	_name(_tname, _optype, orb, _memtype); \
	_name(_tname, _optype, xor, _memtype); \
	_name(_tname, _optype, andl, _memtype); \
	_name(_tname, _optype, orl, _memtype); \
	_name(_tname, _optype, eqv, _memtype); \
	_name(_tname, _optype, neqv, _memtype); \
	_name(_tname, _optype, shl, _memtype); \
	_name(_tname, _optype, shr, _memtype);


ATOMIC_GENOP_ALL4(ATOMIC_GENOP_PARTIAL, fixed1, int8_t, int32_t);
ATOMIC_GENOP_ALL4(ATOMIC_GENOP_PARTIAL, fixed1u, uint8_t, int32_t);
ATOMIC_GENOP_ALL4(ATOMIC_GENOP_PARTIAL, fixed2u, uint16_t, int32_t);
ATOMIC_GENOP_ALL4(ATOMIC_GENOP_PARTIAL, fixed2, int16_t, int32_t);

/** cooperative reduction 
 *  make use of warp, shared variable, and __syncthreads
 **/




template<typename T>
INLINE __device__ T myshfldown(T val, unsigned int delta, int size=warpSize) {
	return __shfl_down(val, delta, size);
#if 0
	T ret = 0;
	int localv;
	int remotev;
	/* not finished */
	switch(sizeof(T)) {
	case 1:
	case 2:
		localv = reinterpret_cast<int>(val);
		remotev = __shfl_down(localv, delta, size);
		ret = reinterpret_cast<T>(remotev);
		break;
	break;
	
	}
	return ret;
#endif
}

#if 0
template<>
INLINE __device__ float myshfldown<float>(float val, unsigned int delta, int size) {
	int t = __float_as_int(val);
	int t1 = __shfl_down(t, delta, size);
	float ret = __int_as_float<float>(t1);
	return ret;
}
#endif

template<>
INLINE __device__ int myshfldown<int>(int val, unsigned int delta, int size) {
	return __shfl_down(val, delta, size);
}

template<>
INLINE __device__ unsigned int myshfldown<unsigned int>(unsigned int val, unsigned int delta, int size) {
	return __shfl_down(val, delta, size);
}

template<>
INLINE __device__ int64_t myshfldown<int64_t>(int64_t val, unsigned int delta, int size) {
	return __shfl_down(val, delta, size);
}

template<>
INLINE __device__ uint64_t myshfldown<uint64_t>(uint64_t val, unsigned int delta, int size) {
	return __shfl_down(val, delta, size);
}


template<>
INLINE __device__ float myshfldown<float>(float val, unsigned int delta, int size) {
	return __shfl_down(val, delta, size);
}

template<>
INLINE __device__ double myshfldown<double>(double val, unsigned int delta, int size) {
	return __shfl_down(val, delta, size);
}

template<>
INLINE __device__ unsigned long long  myshfldown<unsigned long long >(unsigned long long  val, unsigned int delta, int size) {
	return __shfl_down(val, delta, size);
}



template <typename T, omptarget_nvptx_BINOP_t binop>
__inline__ __device__
T reduInitVal() {
	switch(binop) {
   	case omptarget_nvptx_inc:
	case omptarget_nvptx_dec:
	case omptarget_nvptx_add:
	case omptarget_nvptx_sub:
	case omptarget_nvptx_sub_rev:
		return (T) 0;
	case omptarget_nvptx_mul:
	case omptarget_nvptx_div:
		return (T) 1;
	default:
		//ASSERT(0);
		return (T) 0;
	}
}

template <typename T, omptarget_nvptx_BINOP_t binop>
__inline__ __device__
T  warpReduceSum(T val, unsigned int size) {
  for (int offset = size/2; offset > 0; offset /= 2)
    val = Compute<T, binop>(val, myshfldown<T>(val, offset, size));
  return val;
}

//#define MYGSIZE warpSize
#define MYGSIZE 32

template<typename T, omptarget_nvptx_BINOP_t binop>
__inline__ __device__ T warpBlockReduction(T inputval) {
  	__shared__ T shared[MYGSIZE]; 
	
	unsigned int remainder  = blockDim.x & (MYGSIZE-1);;
	unsigned int start_r = blockDim.x - remainder;
  	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	if (blockDim.x < MYGSIZE) {
		shared[threadIdx.x] = inputval;
	} else {
		if (threadIdx.x >= start_r) {
			shared[threadIdx.x - start_r] = inputval;
		} else if (threadIdx.x < MYGSIZE && threadIdx.x >= remainder) {
			shared[threadIdx.x] = reduInitVal<T, binop>();
		}
	}
	__syncthreads();

	if (blockDim.x < MYGSIZE) {
		if (threadIdx.x == 0) {
			T val = shared[0];
			for(unsigned i= 1; i < blockDim.x; i++) {
				val = Compute<T, binop>(val, shared[i]);
			}
			return val;
		}
		return (T) 0;
	}
		
	if (threadIdx.x < start_r) {
		T val = warpReduceSum<T, binop>(inputval, MYGSIZE);
		if (lane == 0) {
			shared[wid] = Compute<T, binop>(shared[wid], val);
		}
	}
	__syncthreads();

	if (wid == 0) {
		T val = warpReduceSum<T, binop>(shared[threadIdx.x], MYGSIZE);
		if (threadIdx.x == 0) {
//			printf("inside %d\n", val);
			return val;
		}
	}
	return (T) 0;
}


#define WARPBLOCK_GENREDU(_name, _dtype, _op) \
        EXTERN _dtype __gpu_warpBlockRedu_##_name##_##_op\
        (_dtype rhs) { \
	return warpBlockReduction<_dtype, omptarget_nvptx_##_op>(rhs); \
}

#define WARPBLOCK_GENREDU_ALLOP(_name, _dtype) \
	WARPBLOCK_GENREDU(_name, _dtype, add); \
	WARPBLOCK_GENREDU(_name, _dtype, sub); \
	WARPBLOCK_GENREDU(_name, _dtype, mul); \
	WARPBLOCK_GENREDU(_name, _dtype, div); \
	WARPBLOCK_GENREDU(_name, _dtype, min); \
	WARPBLOCK_GENREDU(_name, _dtype, max); \
	WARPBLOCK_GENREDU(_name, _dtype, andb); \
	WARPBLOCK_GENREDU(_name, _dtype, orb); \
	WARPBLOCK_GENREDU(_name, _dtype, xor); \
	WARPBLOCK_GENREDU(_name, _dtype, andl); \
	WARPBLOCK_GENREDU(_name, _dtype, orl); \
	WARPBLOCK_GENREDU(_name, _dtype, eqv); \
	WARPBLOCK_GENREDU(_name, _dtype, neqv); \
	WARPBLOCK_GENREDU(_name, _dtype, shl); \
	WARPBLOCK_GENREDU(_name, _dtype, shr); 
	

WARPBLOCK_GENREDU_ALLOP(fixed1, int8_t);
WARPBLOCK_GENREDU_ALLOP(fixed1u, uint8_t);
WARPBLOCK_GENREDU_ALLOP(fixed2, int16_t);
WARPBLOCK_GENREDU_ALLOP(fixed2u, uint16_t);
WARPBLOCK_GENREDU_ALLOP(fixed4, int32_t);
WARPBLOCK_GENREDU_ALLOP(fixed4u, uint32_t);
WARPBLOCK_GENREDU_ALLOP(fixed8, int64_t);
WARPBLOCK_GENREDU_ALLOP(fixed8u, uint64_t);

#define WARPBLOCK_GENREDU_ALLOP_F(_name, _dtype) \
	WARPBLOCK_GENREDU(_name, _dtype, add); \
	WARPBLOCK_GENREDU(_name, _dtype, sub); \
	WARPBLOCK_GENREDU(_name, _dtype, mul); \
	WARPBLOCK_GENREDU(_name, _dtype, div); \
	WARPBLOCK_GENREDU(_name, _dtype, min); \
	WARPBLOCK_GENREDU(_name, _dtype, max); 
WARPBLOCK_GENREDU_ALLOP_F(float4, float);
WARPBLOCK_GENREDU_ALLOP_F(float8, double);


/**************************************
* runtime support for array reduction *
***************************************/

#define ARRAYATOMIC_GENOP(_name, _dtype, _op) \
        EXTERN void __array_atomic_##_name##_##_op\
        (kmp_Indent *id_ref, int32_t gtid, _dtype * lhs, _dtype *rhs, int n) { \
	PRINT(LD_LOOP, "Reduction: thead %d\n", gtid); \
	for(int i = 0; i < n; i++) { \
        	__kmpc_atomic_##_name##_##_op(id_ref, gtid, lhs+i, rhs[i]);\
	 }\
	}\
        EXTERN void __gpu_array_warpBlockRedu_##_name##_##_op\
	(_dtype *ldata, int n) {\
	for(int i = 0; i < n; i++) { \
		ldata[i] = __gpu_warpBlockRedu_##_name##_##_op(ldata[i]); \
	} \
} 

#define ARRAY_GEN_ALLOP_INTEGER(_name, _tname, _optype) \
	_name(_tname, _optype, add) ;\
	_name(_tname, _optype, sub) ;\
	_name(_tname, _optype, mul) ;\
	_name(_tname, _optype, div) ;\
	_name(_tname, _optype, min) ;\
	_name(_tname, _optype, max) ;\
	_name(_tname, _optype, andb) ;\
	_name(_tname, _optype, orb) ;\
	_name(_tname, _optype, xor) ;\
	_name(_tname, _optype, shl) ;\
	_name(_tname, _optype, shr) ;\
	_name(_tname, _optype, andl) ;\
	_name(_tname, _optype, orl) ; \
	_name(_tname, _optype, eqv) ; \
	_name(_tname, _optype, neqv) ; \

#define ARRAY_GEN_ALLOP_FLOAT(_name, _tname, _optype) \
	_name(_tname, _optype, add) ;\
	_name(_tname, _optype, sub) ;\
	_name(_tname, _optype, mul) ;\
	_name(_tname, _optype, div) ;\
	_name(_tname, _optype, min) ;\
	_name(_tname, _optype, max) ;


//ARRAYATOMIC_GENOP(fixed4, int32_t, add);

#if 1
ARRAY_GEN_ALLOP_INTEGER(ARRAYATOMIC_GENOP, fixed1, int8_t);
ARRAY_GEN_ALLOP_INTEGER(ARRAYATOMIC_GENOP, fixed2, int16_t);
ARRAY_GEN_ALLOP_INTEGER(ARRAYATOMIC_GENOP, fixed4, int32_t);
ARRAY_GEN_ALLOP_INTEGER(ARRAYATOMIC_GENOP, fixed8, int64_t);
ARRAY_GEN_ALLOP_FLOAT(ARRAYATOMIC_GENOP,float4, float);
ARRAY_GEN_ALLOP_FLOAT(ARRAYATOMIC_GENOP,float8, double);
#endif

