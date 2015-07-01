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
    global_axij,global_ayij,global_azij,
    *vxiNext,*vyiNext,*vziNext,
    *xiNext,*yiNext,*ziNext,
    *global_m,*global_x,*global_y,*global_z,*global_vx,*global_vy,*global_vz,
    *basicLocal_m,*basicLocal_x,*basicLocal_y,*basicLocal_z,*basicLocal_vx,*basicLocal_vy,*basicLocal_vz,
    *recvLocal_m,*recvLocal_x,*recvLocal_y,*recvLocal_z,*recvLocal_vx,*recvLocal_vy,*recvLocal_vz;
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

  basicLocal_m=(double*)malloc(sizeof(double)*(size/nprocs));
  basicLocal_x=(double*)malloc(sizeof(double)*(size/nprocs));
  basicLocal_y=(double*)malloc(sizeof(double)*(size/nprocs));
  basicLocal_z=(double*)malloc(sizeof(double)*(size/nprocs));
  basicLocal_vx=(double*)malloc(sizeof(double)*(size/nprocs));
  basicLocal_vy=(double*)malloc(sizeof(double)*(size/nprocs));
  basicLocal_vz=(double*)malloc(sizeof(double)*(size/nprocs));
  recvLocal_m=(double*)malloc(sizeof(double)*(size/nprocs));
  recvLocal_x=(double*)malloc(sizeof(double)*(size/nprocs));
  recvLocal_y=(double*)malloc(sizeof(double)*(size/nprocs));
  recvLocal_z=(double*)malloc(sizeof(double)*(size/nprocs));
  recvLocal_vx=(double*)malloc(sizeof(double)*(size/nprocs));
  recvLocal_vy=(double*)malloc(sizeof(double)*(size/nprocs));
  recvLocal_vz=(double*)malloc(sizeof(double)*(size/nprocs));
  vxiNext=(double*)malloc(sizeof(double)*size/nprocs);
  vyiNext=(double*)malloc(sizeof(double)*size/nprocs);
  vziNext=(double*)malloc(sizeof(double)*size/nprocs);
  xiNext=(double*)malloc(sizeof(double)*size/nprocs);
  yiNext=(double*)malloc(sizeof(double)*size/nprocs);
  ziNext=(double*)malloc(sizeof(double)*size/nprocs);
  
  if(my_rank==0){  
    global_m=(double*)malloc(sizeof(double)*size);
    global_x=(double*)malloc(sizeof(double)*size);
    global_y=(double*)malloc(sizeof(double)*size);
    global_z=(double*)malloc(sizeof(double)*size);
    global_vx=(double*)malloc(sizeof(double)*size);
    global_vy=(double*)malloc(sizeof(double)*size);
    global_vz=(double*)malloc(sizeof(double)*size);

    //データを読み込む      
    read_data(argv[1],global_m,size);
    read_data(argv[2],global_x,size);
    read_data(argv[3],global_y,size);
    read_data(argv[4],global_z,size);
    read_data(argv[5],global_vx,size);
    read_data(argv[6],global_vy,size);
    read_data(argv[7],global_vz,size);
  }

  //他プロセスにランク0で読み込んだデータを分配
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Scatter(global_m,size/nprocs,MPI_DOUBLE,basicLocal_m,
	      size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_x,size/nprocs,MPI_DOUBLE,basicLocal_x,
	      size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_y,size/nprocs,MPI_DOUBLE,basicLocal_y,
	      size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_z,size/nprocs,MPI_DOUBLE,basicLocal_z,
	      size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_vx,size/nprocs,MPI_DOUBLE,basicLocal_vx,
	      size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_vy,size/nprocs,MPI_DOUBLE,basicLocal_vy,
	      size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Scatter(global_vz,size/nprocs,MPI_DOUBLE,basicLocal_vz,
	      size/nprocs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  
  //初期化
  local_time=0.0;
  global_axij=0;
  global_ayij=0;
  global_azij=0;
  
  //重力の計算 //rank数でfor文
  for(t=0;t<step;t++){
    MPI_Barrier(MPI_COMM_WORLD);
    startTime=MPI_Wtime();

    //自分を除くノードに送信
    for(i=0;i<nprocs;i++){
      if(i==my_rank){
	continue;
      }
      MPI_Send(basicLocal_x+my_rank,size/nprocs,MPI_DOUBLE,
	       i,0,MPI_COMM_WORLD);
    }

    //basicLocalサイズ分のループ
    for(i=0;i<size/nprocs;i++){
      
      axij=0.0; ayij=0.0; azij=0.0; //aijの初期化

      
      for(k=0;k<nprocs;k++){
	
	  
	  for(j=0;j<size/nprocs;j++){ //1024までやりたい
	    if(i==j){
	      continue;
	    }
	    r=sqrt((basicLocal_x[i]-recvLocal_x[j])*(basicLocal_x[i]-recvLocal_x[j])
		   +(basicLocal_y[i]-recvLocal_y[j])*(basicLocal_y[i]-recvLocal_y[j])
		   +(basicLocal_z[i]-recvLocal_z[j])*(basicLocal_z[i]-recvLocal_z[j]));
	    axij+=G*(recvLocal_m[j]/(r*r))*((recvLocal_x[j]-basicLocal_x[i])/r);
	    ayij+=G*(recvLocal_m[j]/(r*r))*((recvLocal_y[j]-basicLocal_y[i])/r);
	    azij+=G*(recvLocal_m[j]/(r*r))*((recvLocal_z[j]-basicLocal_z[i])/r);
	  }
      }
    
      //他プロセスのaijの情報を受け取り足し合わせてglobal_aijに入れる
      MPI_Barrier(MPI_COMM_WORLD);
      //printf("rank%d axij=%lf\n",my_rank,axij);      
      MPI_Allreduce(&axij,&global_axij,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(&ayij,&global_ayij,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(&azij,&global_azij,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      //printf("my_rankhoge %d\n",my_rank);
      
      vxiNext[i]=basicLocal_vx[i]+global_axij*dt; 
      vyiNext[i]=basicLocal_vy[i]+global_ayij*dt; 
      vziNext[i]=basicLocal_vz[i]+global_azij*dt; 
    
      xiNext[i]=basicLocal_x[i]+vxiNext[i]*dt;
      yiNext[i]=basicLocal_y[i]+vyiNext[i]*dt;
      ziNext[i]=basicLocal_z[i]+vziNext[i]*dt;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    endTime=MPI_Wtime();
    local_time+=endTime-startTime;
    
    memcpy(basicLocal_x,xiNext,sizeof(basicLocal_x)*size/nprocs);
    memcpy(basicLocal_y,yiNext,sizeof(basicLocal_y)*size/nprocs);
    memcpy(basicLocal_z,ziNext,sizeof(basicLocal_z)*size/nprocs);
    memcpy(basicLocal_vx,vxiNext,sizeof(basicLocal_vx)*size/nprocs);
    memcpy(basicLocal_vy,vyiNext,sizeof(basicLocal_vy)*size/nprocs);
    memcpy(basicLocal_vz,vziNext,sizeof(basicLocal_vz)*size/nprocs);
  }
  printf("time = %lf\n",local_time/step);

  MPI_Gather(basicLocal_x,size/nprocs,MPI_DOUBLE,global_x,size/nprocs,
	     MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Gather(basicLocal_y,size/nprocs,MPI_DOUBLE,global_y,size/nprocs,
	     MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Gather(basicLocal_z,size/nprocs,MPI_DOUBLE,global_z,size/nprocs,
	     MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Gather(basicLocal_vx,size/nprocs,MPI_DOUBLE,global_vx,size/nprocs,
	     MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Gather(basicLocal_vy,size/nprocs,MPI_DOUBLE,global_vy,size/nprocs,
	     MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Gather(basicLocal_vz,size/nprocs,MPI_DOUBLE,global_vz,size/nprocs,
	     MPI_DOUBLE,0,MPI_COMM_WORLD);

  if(my_rank==0){  
    write_data(global_x,"xex.dat",size);
    /*write_data(global_y,"yex.dat",size);
    write_data(global_z,"zex.dat",size);
    write_data(global_vx,"vxex.dat",size);
    write_data(global_vy,"vyex.dat",size);
    write_data(global_vz,"vzex.dat",size);
    */
    free(global_m);
    free(global_x);
    free(global_y);
    free(global_z);
    free(global_vx);
    free(global_vy);
    free(global_vz);
  }

  free(basicLocal_m);
  free(basicLocal_x);
  free(basicLocal_y);
  free(basicLocal_z);
  free(basicLocal_vx);
  free(basicLocal_vy);
  free(basicLocal_vz);
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
