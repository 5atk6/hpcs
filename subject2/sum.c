#include <stdio.h>
#include <omp.h>
#define N 100000000

int A[N];
int i,j;


int sum(int *a,int n){
  int s=0;
#pragma omp parallel
  {
#pragma omp for reduction(+:s)
    for(i=0;i<n;i++) s+=a[i];
  }
  return s;
}

main(){
  double t1,t2;
  for(i=0;i<N;i++) A[i]=i;
#pragma omp barrier
#pragma omp master
  {
  t1=omp_get_wtime();
}
  j=sum(A,N);
#pragma omp barrier
#pragma omp master
  {
  t2=omp_get_wtime();
}
  printf("sum = %d  time = %lf\n",j,t2-t1);
}
