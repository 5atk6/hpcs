#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include <malloc.h>

#include "data_util_bin.c"

#define G 1.0
#define dt 1.0
#define size 1024
#define step 10

int main(int argc,char* argv[]){
  double r,startTime,endTime,local_time,
    axij,ayij,azij,
    *vxiNext,*vyiNext,*vziNext,
    *xiNext,*yiNext,*ziNext,
    *global_m,*global_x,*global_y,*global_z,*global_vx,*global_vy,*global_vz,
    *local_m,*local_x,*local_y,*local_z,*local_vx,*local_vy,*local_vz;
  int i,j,k,t;

  MPI_Status status;
  int my_rank, nprocs;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  
  if(argc != 8){
    printf("m x y z vx vy vzのデータを入力してね\n");
    exit(1);
  }

  local_m=(double*)malloc(sizeof(double)*(size/nprocs));
  local_x=(double*)malloc(sizeof(double)*(size/nprocs));
  local_y=(double*)malloc(sizeof(double)*(size/nprocs));
  local_z=(double*)malloc(sizeof(double)*(size/nprocs));
  local_vx=(double*)malloc(sizeof(double)*(size/nprocs));
  local_vy=(double*)malloc(sizeof(double)*(size/nprocs));
  local_vz=(double*)malloc(sizeof(double)*(size/nprocs));
  
  //データを読み込む
  if(my_rank==0){  
    global_m=(double*)malloc(sizeof(double)*size);
    global_x=(double*)malloc(sizeof(double)*size);
    global_y=(double*)malloc(sizeof(double)*size);
    global_z=(double*)malloc(sizeof(double)*size);
    global_vx=(double*)malloc(sizeof(double)*size);
    global_vy=(double*)malloc(sizeof(double)*size);
    global_vz=(double*)malloc(sizeof(double)*size);
        
    read_data(argv[1],global_m,size);
    read_data(argv[2],global_x,size);
    read_data(argv[3],global_y,size);
    read_data(argv[4],global_z,size);
    read_data(argv[5],global_vx,size);
    read_data(argv[6],global_vy,size);
    read_data(argv[7],global_vz,size);
  }
    
  MPI_Barrier(MPI_COMM_WORLD);
  //他プロセスにランク0で読み込んだデータを分配
  MPI_Scatter(global_m,size/nprocs,MPI_DOUBLE,local_m,
	      size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_x,size/nprocs,MPI_DOUBLE,local_x,
	      size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_y,size/nprocs,MPI_DOUBLE,local_y,
	      size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_z,size/nprocs,MPI_DOUBLE,local_z,
	      size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_vx,size/nprocs,MPI_DOUBLE,local_vx,
	      size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_vy,size/nprocs,MPI_DOUBLE,local_vy,
	      size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_vz,size/nprocs,MPI_DOUBLE,local_vz,
	      size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  
  vxiNext=(double*)malloc(sizeof(double)*size/nprocs);
  vyiNext=(double*)malloc(sizeof(double)*size/nprocs);
  vziNext=(double*)malloc(sizeof(double)*size/nprocs);
  xiNext=(double*)malloc(sizeof(double)*size/nprocs);
  yiNext=(double*)malloc(sizeof(double)*size/nprocs);
  ziNext=(double*)malloc(sizeof(double)*size/nprocs);
  
  //初期化
  for(i=0;i<size/nprocs;i++){
    vxiNext[i]=0;
    vyiNext[i]=0;
    vziNext[i]=0;
    xiNext[i]=0;
    yiNext[i]=0;
    ziNext[i]=0;
  }
  
  //時間計測について
  local_time=0.0;
  
  double global_axij=0;
  double global_ayij=0;
  double global_azij=0;
  //重力の計算 
  for(t=0;t<step;t++){
    MPI_Barrier(MPI_COMM_WORLD);
    startTime=MPI_Wtime();

    for(i=0;i<size/nprocs;i++){
      
      axij=0.0; ayij=0.0; azij=0.0; //aijの初期化
      for(j=0;j<size/nprocs;j++){
	if(i==j){
	  continue;
	}
	
	r=sqrt((local_x[i]-local_x[j])*(local_x[i]-local_x[j])
	       +(local_y[i]-local_y[j])*(local_y[i]-local_y[j])
	       +(local_z[i]-local_z[j])*(local_z[i]-local_z[j]));
	axij+=G*(local_m[j]/(r*r))*((local_x[j]-local_x[i])/r);
	ayij+=G*(local_m[j]/(r*r))*((local_y[j]-local_y[i])/r);
	azij+=G*(local_m[j]/(r*r))*((local_z[j]-local_z[i])/r);

	
	/*for(k=0;k<nprocs;k++){
	  if(k==my_rank){
	    continue;
	  }
	  MPI_Send(&axij+my_rank,size/nprocs,MPI_DOUBLE,k,0,MPI_COMM_WORLD);  
	  MPI_Recv(&axij+k,size/nprocs,MPI_DOUBLE,k,0,MPI_COMM_WORLD,&status);
	  }*/
      }
      
      //他プロセスのaijの情報を受け取り足し合わせてglobal_aijに入れる
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Allreduce(&axij,&global_axij,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(&ayij,&global_ayij,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(&azij,&global_azij,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      
      vxiNext[i]=local_vx[i]+global_axij*dt; 
      vyiNext[i]=local_vy[i]+global_ayij*dt; 
      vziNext[i]=local_vz[i]+global_azij*dt; 
    
      xiNext[i]=local_x[i]+vxiNext[i]*dt;
      yiNext[i]=local_y[i]+vyiNext[i]*dt;
      ziNext[i]=local_z[i]+vziNext[i]*dt;
      
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    endTime=MPI_Wtime();
    local_time+=endTime-startTime;
    
    memcpy(local_x,xiNext,sizeof(local_x)*size/nprocs);
    memcpy(local_y,yiNext,sizeof(local_y)*size/nprocs);
    memcpy(local_z,ziNext,sizeof(local_z)*size/nprocs);
    memcpy(local_vx,vxiNext,sizeof(local_vx)*size/nprocs);
    memcpy(local_vy,vyiNext,sizeof(local_vy)*size/nprocs);
    memcpy(local_vz,vziNext,sizeof(local_vz)*size/nprocs);
  }
  printf("time = %lf\n",local_time/step);

  MPI_Gather(local_x,size/nprocs,MPI_DOUBLE,global_x,size/nprocs,
	     MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Gather(local_y,size/nprocs,MPI_DOUBLE,global_y,size/nprocs,
	     MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Gather(local_z,size/nprocs,MPI_DOUBLE,global_z,size/nprocs,
	     MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Gather(local_vx,size/nprocs,MPI_DOUBLE,global_vx,size/nprocs,
	     MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Gather(local_vy,size/nprocs,MPI_DOUBLE,global_vy,size/nprocs,
	     MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Gather(local_vz,size/nprocs,MPI_DOUBLE,global_vz,size/nprocs,
	     MPI_DOUBLE,0,MPI_COMM_WORLD);
  if(my_rank==0){
    
    write_data(global_x,"xex.dat",size);
    /*write_data(output_y,"yex.dat",size/nprocs);
    write_data(output_z,"zex.dat",size/nprocs);
    write_data(output_vx,"vxex.dat",size/nprocs);
    write_data(output_vy,"vyex.dat",size/nprocs);
    write_data(output_vz,"vzex.dat",size/nprocs);
    */
    free(global_m);
    free(global_x);
    free(global_y);
    free(global_z);
    free(global_vx);
    free(global_vy);
    free(global_vz);
  }

  free(local_m);
  free(local_x);
  free(local_y);
  free(local_z);
  free(local_vx);
  free(local_vy);
  free(local_vz);
  free(xiNext);
  free(yiNext);
  free(ziNext);
  free(vxiNext);
  free(vyiNext);
  free(vziNext);
  
  printf("success\n");
  MPI_Finalize();
  
  return 0;
}
