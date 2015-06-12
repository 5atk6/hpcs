#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include <malloc.h>

#include "data_util_bin.c"

#define G 1.0f
#define dt 1.0f
#define size 1000
#define step 10

int main(int argc,char* argv[]){
  double r,axij,ayij,azij,*m,*x,*y,*z,*vx,*vy,*vz,*vxiNext,*vyiNext,*vziNext,*xiNext,*yiNext,*ziNext,startTime,endTime,local_time,*global_m,*global_x,*global_y,*global_z,*global_vx,*global_vy,*global_vz,*local_m,*local_x,*local_y,*local_z,*local_vx,*local_vy,*local_vz;
  int i,j,k,t;

  MPI_Status recv_status;
  int my_rank, nprocs;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  
  if(argc != 8){
    printf("m x y z vx vy vzのデータを入力してね\n");
    exit(1);
  }

  local_m=(double*)malloc(sizeof(double)*size/nprocs);
  local_x=(double*)malloc(sizeof(double)*size/nprocs);
  local_y=(double*)malloc(sizeof(double)*size/nprocs);
  local_z=(double*)malloc(sizeof(double)*size/nprocs);
  local_vx=(double*)malloc(sizeof(double)*size/nprocs);
  local_vy=(double*)malloc(sizeof(double)*size/nprocs);
  local_vz=(double*)malloc(sizeof(double)*size/nprocs);
  
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
    //printf("hogeee %lf\n",global_x[500]);  
  }
  MPI_Barrier(MPI_COMM_WORLD);
  //他プロセスにランク0で読み込んだデータを分配
  MPI_Scatter(global_m,size/nprocs,MPI_DOUBLE,local_m,size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_x,size/nprocs,MPI_DOUBLE,local_x,size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_y,size/nprocs,MPI_DOUBLE,local_y,size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_z,size/nprocs,MPI_DOUBLE,local_z,size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_vx,size/nprocs,MPI_DOUBLE,local_vx,size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_vy,size/nprocs,MPI_DOUBLE,local_vy,size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_vz,size/nprocs,MPI_DOUBLE,local_vz,size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  //printf("hoge %d %lf\n",my_rank,local_x[0]);
  
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
    //for(i=my_rank*(size/nprocs);i<my_rank*(size/nprocs)+(size/nprocs);i++){
      //aijについて
      axij=0; ayij=0; azij=0; //aijの初期化
      for(j=0;j<size;j++){
      if(i==j){
	  continue;
      }
      
	r=sqrt((local_x[i]-local_x[j])*(local_x[i]-local_x[j])+(local_y[i]-local_y[j])*(local_y[i]-local_y[j])+(local_z[i]-local_z[j])*(local_z[i]-local_z[j]));
	axij+=G*(local_m[j]/(r*r))*((local_x[j]-local_x[i])/r);
	ayij+=G*(local_m[j]/(r*r))*((local_y[j]-local_y[i])/r);
	azij+=G*(local_m[j]/(r*r))*((local_z[j]-local_z[i])/r);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      
      //他プロセスのaijの情報を受け取り足し合わせてrxijに入れる
      MPI_Allreduce(&axij,&global_axij,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(&ayij,&global_ayij,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(&azij,&global_azij,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      //printf("ここまではおっけー %d \n",my_rank);
      
      vxiNext[i]=local_vx[i]+global_axij; 
      vyiNext[i]=local_vy[i]+global_axij; 
      vziNext[i]=local_vz[i]+global_axij; 
    
      xiNext[i]=local_x[i]+vxiNext[i]*dt;
      yiNext[i]=local_y[i]+vyiNext[i]*dt;
      ziNext[i]=local_z[i]+vziNext[i]*dt;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    endTime=MPI_Wtime();
    local_time+=endTime-startTime;
    
    memcpy(local_x,xiNext,sizeof(x)*size/nprocs);
    memcpy(local_y,yiNext,sizeof(y)*size/nprocs);
    memcpy(local_z,ziNext,sizeof(z)*size/nprocs);
    memcpy(local_vx,vxiNext,sizeof(vx)*size/nprocs);
    memcpy(local_vy,vyiNext,sizeof(vy)*size/nprocs);
    memcpy(local_vz,vziNext,sizeof(vz)*size/nprocs);
    
  }
  printf("time = %lf\n",local_time/step);
  
  /*write_data(x,"xex.dat",size);
  write_data(y,"yex.dat",size);
  write_data(z,"zex.dat",size);
  write_data(vxiNext,"vxex.dat",size);
  write_data(vyiNext,"vyex.dat",size);
  write_data(vxiNext,"vzex.dat",size);*/

  if(my_rank==0){
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




/*if(x==NULL){
    printf("xerror\n");  
    return 1;
  }if(y==NULL){
    printf("error\n");
    return 1;
  }if(z==NULL){
    printf("error\n");
    return 1;
  }if(m==NULL){
    printf("merror\n");
    return 1;
  }if(vx==NULL){
    printf("error\n");
    return 1;
  }if(vy==NULL){
    printf("error\n");
    return 1;
  }if(vz==NULL){
    printf("error\n");
    return 1;
  }
  if(vxiNext==NULL){
    printf("error\n");
    return 1;
  }if(vyiNext==NULL){
    printf("error\n");
    return 1;
  }if(vziNext==NULL){
    printf("error\n");
    return 1;
  }if(xiNext==NULL){
    printf("error\n");
    return 1;
  }if(yiNext==NULL){
    printf("error\n");
    return 1;
  }if(ziNext==NULL){
    printf("error\n");
    return 1;
    }*/
					  

  /*sendXbuf=(double*)malloc(sizeof(double)*nprocs);
  sendYbuf=(double*)malloc(sizeof(double)*nprocs);
  sendZbuf=(double*)malloc(sizeof(double)*nprocs);
  receiveXbuf=(double*)malloc(sizeof(double)*nprocs);
  receiveYbuf=(double*)malloc(sizeof(double)*nprocs);
  receiveZbuf=(double*)malloc(sizeof(double)*nprocs);*/  
