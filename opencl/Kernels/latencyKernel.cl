#ifndef LATENCY_KERNEL_C
#define LATENCY_KERNEL_C

#define CHASING		myPtr = *((ptr_type*)myPtr);
#define CHASING_20	CHASING CHASING CHASING CHASING CHASING \
					CHASING CHASING CHASING CHASING CHASING \
					CHASING CHASING CHASING CHASING CHASING \
					CHASING CHASING CHASING CHASING CHASING

#define CHASING_400  	CHASING_20 CHASING_20 CHASING_20 CHASING_20 CHASING_20	\ 
						CHASING_20 CHASING_20 CHASING_20 CHASING_20 CHASING_20	\
						CHASING_20 CHASING_20 CHASING_20 CHASING_20 CHASING_20	\
						CHASING_20 CHASING_20 CHASING_20 CHASING_20 CHASING_20	\	

typedef unsigned long ptr_type;

kernel void add_address(global ptr_type * restrict d_source, int length)
{
	int globalId = get_global_id(0);
	int globalSize = get_global_size(0);

	while (globalId < length) {
		d_source[globalId] += (ptr_type)(&d_source[0]);
		globalId += globalSize;
	}
}

kernel void latency (global ptr_type* restrict d_source, const int length)
{
	//only one workitem is functioning
	ptr_type myPtr = (ptr_type)(&d_source);
	for(int i = 0; i < 1000; i++) {		//chasing for 2000000 times
		CHASING_400 CHASING_400 CHASING_400 CHASING_400 CHASING_400
	}
	d_source[length] = myPtr;	//write to the last place of d_source and should be a cache hit
}

#endif
