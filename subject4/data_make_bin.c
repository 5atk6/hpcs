#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "data_util_bin.c"

#define N 1024

void init_r(double *r, int num)
{
	int i;
	for( i=0; i < num; i++)
		r[i] = rand()/(RAND_MAX+1.0)*100.0-50.0;
}

void init_m(double *m, int num)
{
	int i;
	for(i=0; i < num; i++)
		m[i] = 1.0+rand()/(RAND_MAX+1.0);
}

void init_v(double *v, int num)
{
	init_r(v,num);
}

int main(int argc, char* argv[])
{
	int size;
	double *m,*x,*y,*z,*vx, *vy, *vz;

	if(argc == 1){
		size = N;
	}
	else {
		size = atoi(argv[1]);
		if(size == 0){
			size = atoi(argv[1]);
		}
	}

	m  = (double*)malloc(sizeof(double)*size);
	x  = (double*)malloc(sizeof(double)*size);
	y  = (double*)malloc(sizeof(double)*size);
	z  = (double*)malloc(sizeof(double)*size);
	vx = (double*)malloc(sizeof(double)*size);
	vy = (double*)malloc(sizeof(double)*size);
	vz = (double*)malloc(sizeof(double)*size);

	init_m(m,size);
	init_r(x,size);
	init_r(y,size);
	init_r(z,size);
	init_v(x,size);
	init_v(y,size);
	init_v(z,size);


	write_data(m,"m.dat",size);
	write_data(x,"x.dat",size);
	write_data(y,"y.dat",size);
	write_data(z,"z.dat",size);
	write_data(vx,"vx.dat",size);
	write_data(vy,"vy.dat",size);
	write_data(vx,"vz.dat",size);

	free(m);
	free(x);
	free(y);
	free(z);
	free(vx);
	free(vy);
	free(vz);

	printf("success\n");
	return 0;
}