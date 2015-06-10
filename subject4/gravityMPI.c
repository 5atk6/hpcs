#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

#include "data_util_bin.c"

#define G 1.0f
#define dt 1.0f
#define size 1000
#define step 10

int main(int argc,char* argv[]){
  double r,axij,ayij,azij,*m,*x,*y,*z,*vx,*vy,*vz,*vxiNext,*vyiNext,*vziNext,*xiNext,*yiNext,*ziNext,startTime,endTime,time,*sendXbuf,*sendYbuf,*sendZbuf,*receiveXbuf,*receiveYbuf,*receiveZbuf;
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
  //データを読み込む
  m=(double*)malloc(sizeof(double)*size/nprocs);
  x=(double*)malloc(sizeof(double)*size/nprocs);
  y=(double*)malloc(sizeof(double)*size/nprocs);
  z=(double*)malloc(sizeof(double)*size/nprocs);
  vx=(double*)malloc(sizeof(double)*size/nprocs);
  vy=(double*)malloc(sizeof(double)*size/nprocs);
  vz=(double*)malloc(sizeof(double)*size/nprocs);
  vxiNext=(double*)malloc(sizeof(double)*size/nprocs);
  vyiNext=(double*)malloc(sizeof(double)*size/nprocs);
  vziNext=(double*)malloc(sizeof(double)*size/nprocs);
  xiNext=(double*)malloc(sizeof(double)*size/nprocs);
  yiNext=(double*)malloc(sizeof(double)*size/nprocs);
  ziNext=(double*)malloc(sizeof(double)*size/nprocs);
  /*sendXbuf=(double*)malloc(sizeof(double)*nprocs);
  sendYbuf=(double*)malloc(sizeof(double)*nprocs);
  sendZbuf=(double*)malloc(sizeof(double)*nprocs);
  receiveXbuf=(double*)malloc(sizeof(double)*nprocs);
  receiveYbuf=(double*)malloc(sizeof(double)*nprocs);
  receiveZbuf=(double*)malloc(sizeof(double)*nprocs);*/
  read_data(argv[1],m,size);
  read_data(argv[2],x,size);
  read_data(argv[3],y,size);
  read_data(argv[4],z,size);
  read_data(argv[5],vx,size);
  read_data(argv[6],vy,size);
  read_data(argv[7],vz,size);
  
  //初期化
  for(i=0;i<size;i++){
    vxiNext[i]=0;
    vyiNext[i]=0;
    vziNext[i]=0;
    xiNext[i]=0;
    yiNext[i]=0;
    ziNext[i]=0;
  }
  
  printf("%d\n",my_rank);
  //時間計測について
  time=0;
  int start,end;

  //
  double global_axij=0;
  double global_ayij=0;
  double global_azij=0;
  //重力の計算 
  for(t=0;t<step;t++){
    
    //
    
    startTime=MPI_Wtime();
      
    for(i=my_rank*(size/nprocs);i<my_rank*(size/nprocs)+(size/nprocs);i++){
      //aijについて
      axij=0; ayij=0; azij=0; //aijの初期化
      for(j=0;j<size;j++){
	if(i==j){
	  continue;
	}
	r=sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j])+(z[i]-z[j])*(z[i]-z[j]));
	axij+=G*(m[j]/(r*r))*((x[j]-x[i])/r);
	ayij+=G*(m[j]/(r*r))*((y[j]-y[i])/r);
	azij+=G*(m[j]/(r*r))*((z[j]-z[i])/r);
	axij=my_rank;
      }
      //MPI_Barrier(MPI_COMM_WORLD);
      printf("%lf\n",axij);
      //他プロセスのaijの情報を受け取り足し合わせてrxijに入れる
      MPI_Allreduce(&axij,&global_axij,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      //MPI_Allreduce(&ayij,&global_ayij,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      //MPI_Allreduce(&azij,&global_azij,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      printf("ここまではおっけー\n");      		      
      vxiNext[i]=vx[i]+global_axij; 
      vyiNext[i]=vy[i]+global_axij; 
      vziNext[i]=vz[i]+global_axij; 
    
      xiNext[i]=x[i]+vxiNext[i]*dt;
      yiNext[i]=y[i]+vyiNext[i]*dt;
      ziNext[i]=z[i]+vziNext[i]*dt;
    }
    endTime=MPI_Wtime();
    time+=end-start;
    
    memcpy(x,xiNext,sizeof(x)*size/nprocs);
    memcpy(y,yiNext,sizeof(y)*size/nprocs);
    memcpy(z,ziNext,sizeof(z)*size/nprocs);
    memcpy(vx,vxiNext,sizeof(vx)*size/nprocs);
    memcpy(vy,vyiNext,sizeof(vy)*size/nprocs);
    memcpy(vz,vziNext,sizeof(vz)*size/nprocs);
    
  }
  printf("time = %lf\n",time/step);
  
  write_data(x,"xex.dat",size);
  write_data(y,"yex.dat",size);
  write_data(z,"zex.dat",size);
  write_data(vxiNext,"vxex.dat",size);
  write_data(vyiNext,"vyex.dat",size);
  write_data(vxiNext,"vzex.dat",size);
  
  free(m);
  free(x);
  free(y);
  free(z);
  free(vx);
  free(vy);
  free(vz);
  free(xiNext);
  free(yiNext);
  free(ziNext);
  free(vxiNext);
  free(vyiNext);
  free(vziNext);

  MPI_Finalize();
  printf("success\n");
  return 0;
}



			 
					  
