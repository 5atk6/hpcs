#include <stdio.h>
#include <omp.h>
#include <math.h>

#define N 1000

double sum(double *a,int n);

int i;
double A[N],B[N],C[N],j,k,l;

main(){
  double t1,t2;
  for(i=1;i<N;i++) A[i]=(double)i;
  for(i=1;i<N;i++) B[i]=1.0/A[i];
  for(i=1;i<N;i++) C[i]=sin(B[i]);
#pragma omp barrier
#pragma omp master
  {
    t1=omp_get_wtime();
  }
  j=sum(A,N);
  k=sum(B,N);
  l=sum(C,N);
#pragma omp barrier
#pragma omp master
  {
    t2=omp_get_wtime();
  }
  printf("sum = %lf  time = %lf 1/m = %lf sin(1/m) = %lf\n",j,t2-t1,k,l);
  printf("%lf\n",M_PI);
}

double sum(double *a,int n){
  double s=0.0;
#pragma omp parallel
  {
#pragma omp for reduction(+:s)
    for(i=1;i<n;i++) s+=1.0*a[i];
  }
  return s;
}
