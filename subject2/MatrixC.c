#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

#define S 1000
#define T 1000
#define U 1000
#define N 3.0 //試行回数

void makeMatrix(int *matrix,int n,int m){
  int i,j,k;
  k=0;
  for(i=0;i<n;i++){
    for(j=0;j<m;j++,k++){
      matrix[j+i*m]=j+i*m;
    }
  }
}

void multiplMatrix(int a[S*T],int b[T*U],int c[S*U]){
  int i,j,k;
#pragma omp parallel for private(i,j)
  for(i=0;i<S;i++){
    for(j=0;j<U;j++){
      c[j+i*U]=0;
    }
  }
  #pragma omp parallel for private(i,j,k)
  for(i=0;i<S;i++){
    for(j=0;j<U;j++){
      for(k=0;k<T;k++){
	c[j+i*U]+=a[k+i*T]*b[j+k*U];
      }
    }
  }
}

main(){
  int matrixA[S*T],matrixB[T*U],matrixC[S*U],i;
  double t1,t2,time;
  makeMatrix(matrixA,S,T);
  makeMatrix(matrixB,T,U);
  
  
  for(i=0;i<N;i++){
    t1=omp_get_wtime();
    multiplMatrix(matrixA,matrixB,matrixC);
    t2=omp_get_wtime();
    time+=t2-t1;
  }
  printf("time = %lf\n",time/N);
  return 0;
}
