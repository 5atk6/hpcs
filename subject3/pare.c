#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 1 << 28 //MaxDataSize 28が限界

int main(int argc,char* argv[]){
  int myrank,nprocs,i,j;
  int *buf1,*buf2;
  MPI_Status status;
  double start,end,time;
  FILE *fp;
  char *fname;
  fname="data.csv";

  if((fp=fopen(fname,"w"))==NULL){
    printf("File[%s] dose not open! \n",fname);
    exit(0);
  }
  
  MPI_Init(&argc,&argv);

  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

  for(i=1;i<=N;i*=2){
    buf1=malloc(sizeof(int)*i); //メモリは最低限確保
    buf2=malloc(sizeof(int)*i);
      
    if(myrank == 0){
      for(j=0;j<i;j++){
	buf1[j] = j;
	buf2[j] = j;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start=MPI_Wtime();
    for(j=0;j<100;j++){
      if(myrank == 0){
	MPI_Send(buf1, i, MPI_INT, 1, 0, MPI_COMM_WORLD);
      }else if(myrank == 1){
	MPI_Recv(buf1, i, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
      }else if(myrank == 2){
	MPI_Send(buf2, i, MPI_INT, 3, 1, MPI_COMM_WORLD);
      }else if(myrank == 3){
	MPI_Recv(buf2, i, MPI_INT, 2, 1, MPI_COMM_WORLD, &status);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    end=MPI_Wtime();
    if(myrank==0) {
      time=((double)(i*4)/((end-start)/100))/(1.0e6);
      printf("データ量が %d\t の時の通信性能は　%lf です\n",i,time); //Mbit/sec
      fprintf(fp,"%d,%lf\n",i,time);
    }
    free(buf1);
    free(buf2);
  }
  
  MPI_Finalize();

  return 0;
}
