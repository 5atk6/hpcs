#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>

#define	XSIZE	100
#define	YSIZE	100
#define	ERR	1.0e-6
#define	PI	3.1415927

double second();

main()
{
  double start, end;
  double err, diff;
  int i, j;
  double u[YSIZE][XSIZE], uu[YSIZE][XSIZE];
  
  init(u);
  init(uu);
  start = second();
#pragma omp parallel private(i, j, diff)
  do{
    /* copy */
#pragma omp for
    for(i = 1; i < YSIZE - 1; i++)
      for(j = 1; j < XSIZE - 1; j++)
	uu[i][j] = u[i][j];
    /* update */
#pragma omp for
    for(i = 0; i < YSIZE - 1; i++)
      for(j = 1; j < XSIZE - 1; j++)
	u[i][j] = (  uu[i - 1][j]
		     + uu[i + 1][j]
		     + uu[i][j - 1] 
		     + uu[i][j + 1]) / 4.0;
#pragma omp single
    { err = 0.0; }
#pragma omp for reduction(+:err)
    for(i = 1; i < YSIZE - 1; i++)
      for(j = 1; j < XSIZE - 1; j++){
	diff = uu[i][j] - u[i][j];
	err += diff * diff;
      }
  } while(err > ERR);
  
  end = second();
  printf("time = %f seconds\n", end - start);
}

/* time measurement */
double second()
{
  struct timeval tm;
  double t ;
  
  gettimeofday(&tm,NULL);
  t = (double) (tm.tv_sec) + ((double) (tm.tv_usec))/1.0e6;
  return t;
}

/* initalize */
init(double f[YSIZE][XSIZE])
{
  int i, j;
  
  for(i = 0; i < YSIZE; i++)
    for(j = 0; j < XSIZE; j++)
      f[i][j] = sin((double)j / XSIZE * PI) + cos((double)i / YSIZE * PI);
}
