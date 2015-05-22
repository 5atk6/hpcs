#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
  int myrank, nprocs, i;
  int *buf;
  MPI_Status status;
  
  MPI_Init(&argc, &argv);
  
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  
  buf = malloc(sizeof(int)*20);
  
  if(myrank == 0){
    for(i=0;i<20;i++){
      buf[i] = i;
    }
  }
  
  if(myrank == 0){
    MPI_Send(buf, 20, MPI_INT, 1, 0, MPI_COMM_WORLD);
  }else if(myrank == 1){
    MPI_Recv(buf, 20, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
  }
  
  if(myrank == 0){
    printf("[%d/%d] send data = {", myrank, nprocs);
    for(i=0;i<20;i++){
      printf("%d,", buf[i]);
    }
    printf("}\n");
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  if(myrank == 1){
    printf("[%d/%d] recv data = {", myrank, nprocs);
    for(i=0;i<20;i++){
      printf("%d,", buf[i]);
    }
    printf("}\n");
  }
  
  MPI_Finalize();
}
