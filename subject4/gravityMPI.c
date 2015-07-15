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
#ifndef step
#define step 10
#endif

int main(int argc,char* argv[]){
	double r, startTime, endTime, local_time,
		*axij, *ayij, *azij,
		*global_m, *global_x, *global_y, *global_z, *global_vx, *global_vy, *global_vz,
		*basicLocal_m, *basicLocal_x, *basicLocal_y, *basicLocal_z, *basicLocal_vx, *basicLocal_vy, *basicLocal_vz,
		*recvLocal_m, *recvLocal_x, *recvLocal_y, *recvLocal_z, *recvLocal_vx, *recvLocal_vy, *recvLocal_vz;
	int i, j, k, t;
	MPI_Status status;
	MPI_Request ireqX, ireqY, ireqZ, ireqM;
	ireqX = MPI_REQUEST_NULL;
	ireqY = MPI_REQUEST_NULL;
	ireqZ = MPI_REQUEST_NULL;
	ireqM = MPI_REQUEST_NULL;
	int my_rank, nprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  
	if(argc != 8){
		printf("m x y z vx vy vzのデータを入力してね\n");
		exit(1);
	}

	basicLocal_m  = (double*)malloc(sizeof(double)*(size/nprocs));
	basicLocal_x  = (double*)malloc(sizeof(double)*(size/nprocs));
	basicLocal_y  = (double*)malloc(sizeof(double)*(size/nprocs));
	basicLocal_z  = (double*)malloc(sizeof(double)*(size/nprocs));
	basicLocal_vx = (double*)malloc(sizeof(double)*(size/nprocs));
	basicLocal_vy = (double*)malloc(sizeof(double)*(size/nprocs));
	basicLocal_vz = (double*)malloc(sizeof(double)*(size/nprocs));
	
	recvLocal_m   = (double*)malloc(sizeof(double)*(size/nprocs));
	recvLocal_x   = (double*)malloc(sizeof(double)*(size/nprocs));
	recvLocal_y   = (double*)malloc(sizeof(double)*(size/nprocs));
	recvLocal_z   = (double*)malloc(sizeof(double)*(size/nprocs));
	recvLocal_vx  = (double*)malloc(sizeof(double)*(size/nprocs));
	recvLocal_vy  = (double*)malloc(sizeof(double)*(size/nprocs));
	recvLocal_vz  = (double*)malloc(sizeof(double)*(size/nprocs));

	axij          = (double*)malloc(sizeof(double)*size/nprocs);
	ayij          = (double*)malloc(sizeof(double)*size/nprocs);
	azij          = (double*)malloc(sizeof(double)*size/nprocs);
  
	global_m      = (double*)malloc(sizeof(double)*size);
	global_x      = (double*)malloc(sizeof(double)*size);
	global_y      = (double*)malloc(sizeof(double)*size);
	global_z      = (double*)malloc(sizeof(double)*size);
	global_vx     = (double*)malloc(sizeof(double)*size);
	global_vy     = (double*)malloc(sizeof(double)*size);
	global_vz     = (double*)malloc(sizeof(double)*size);
	
	//データを読み込む
	if(my_rank == 0){
		read_data(argv[1], global_m, size);
		read_data(argv[2], global_x, size);
		read_data(argv[3], global_y, size);
		read_data(argv[4], global_z, size);
		read_data(argv[5], global_vx, size);
		read_data(argv[6], global_vy, size);
		read_data(argv[7], global_vz, size);
	}

	//他プロセスにランク0で読み込んだデータを分配
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Scatter(global_m,size/nprocs,MPI_DOUBLE, basicLocal_m,
		    size/nprocs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
	
	//重力の計算 
	for(t=0;t<step;t++){
		MPI_Barrier(MPI_COMM_WORLD);
		startTime=MPI_Wtime();

		//自分を除くノードに送信
		for(i=0;i<nprocs;i++){
			if(i==my_rank){
				continue;
			}
			MPI_Isend(basicLocal_x,size/nprocs,MPI_DOUBLE,
				  i,0,MPI_COMM_WORLD,&ireqX);
			MPI_Isend(basicLocal_y,size/nprocs,MPI_DOUBLE,
				  i,1,MPI_COMM_WORLD,&ireqY);
			MPI_Isend(basicLocal_z,size/nprocs,MPI_DOUBLE,
				  i,2,MPI_COMM_WORLD,&ireqZ);
			MPI_Isend(basicLocal_m,size/nprocs,MPI_DOUBLE,
				  i,3,MPI_COMM_WORLD,&ireqM); 
		}
		MPI_Barrier(MPI_COMM_WORLD);
		int flag = 0;
		
		for(i=0;i<size/nprocs;i++){
			axij[i] = 0;
			ayij[i] = 0;
			azij[i] = 0;
		}
		for(k = 0; k < nprocs; k++){
			//受け取り
			if(k!=my_rank){
				flag=0;
				MPI_Irecv(recvLocal_x,size/nprocs,MPI_DOUBLE,k,0,MPI_COMM_WORLD,&ireqX);
				MPI_Irecv(recvLocal_y,size/nprocs,MPI_DOUBLE,k,1,MPI_COMM_WORLD,&ireqY);
				MPI_Irecv(recvLocal_z,size/nprocs,MPI_DOUBLE,k,2,MPI_COMM_WORLD,&ireqZ);
				MPI_Irecv(recvLocal_m,size/nprocs,MPI_DOUBLE,k,3,MPI_COMM_WORLD,&ireqM);
				MPI_Wait(&ireqX,&status);
				MPI_Wait(&ireqY,&status);
				MPI_Wait(&ireqZ,&status);
				MPI_Wait(&ireqM,&status);
			}else{
				flag = 1;
				memcpy(recvLocal_x,basicLocal_x,sizeof(basicLocal_x)*size/nprocs);
				memcpy(recvLocal_y,basicLocal_y,sizeof(basicLocal_y)*size/nprocs);
				memcpy(recvLocal_z,basicLocal_z,sizeof(basicLocal_z)*size/nprocs);
				memcpy(recvLocal_m,basicLocal_m,sizeof(basicLocal_m)*size/nprocs);
			}

			//printf("my_rank=%d k=%d recvL=%lf basicL=%lf\n",my_rank,k,recvLocal_x[0],basicLocal_x[0]);
      
			for(i=0;i<size/nprocs;i++){
				
				for(j=0;j<size/nprocs;j++){
					if(i==j && flag == 1){
						continue;
					}
	  
					r=sqrt( (basicLocal_x[i]-recvLocal_x[j])*(basicLocal_x[i]-recvLocal_x[j])
						+(basicLocal_y[i]-recvLocal_y[j])*(basicLocal_y[i]-recvLocal_y[j])
						+(basicLocal_z[i]-recvLocal_z[j])*(basicLocal_z[i]-recvLocal_z[j]));
					axij[i]+=G*(recvLocal_m[j]/(r*r))*((recvLocal_x[j]-basicLocal_x[i])/r);
					ayij[i]+=G*(recvLocal_m[j]/(r*r))*((recvLocal_y[j]-basicLocal_y[i])/r);
					azij[i]+=G*(recvLocal_m[j]/(r*r))*((recvLocal_z[j]-basicLocal_z[i])/r); 
				}	
				MPI_Barrier(MPI_COMM_WORLD);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		for(i=0; i< size/nprocs; i++){
			basicLocal_vx[i] += axij[i]*dt;
			basicLocal_vy[i] += ayij[i]*dt;
			basicLocal_vz[i] += azij[i]*dt;
			basicLocal_x[i]   += basicLocal_vx[i]*dt;
			basicLocal_y[i]   += basicLocal_vy[i]*dt;
			basicLocal_z[i]   += basicLocal_vz[i]*dt;
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		
		endTime=MPI_Wtime();
		local_time+=endTime-startTime;
    
	}
	printf("time = %lf\n",local_time/nprocs);

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
	free(recvLocal_m);
	free(recvLocal_x);
	free(recvLocal_y);
	free(recvLocal_z);
	free(recvLocal_vx);
	free(recvLocal_vy);
	free(recvLocal_vz);

	free(axij);
	free(ayij);
	free(azij);
  
	printf("success step=%d nprocs=%d\n",step,nprocs);
	MPI_Finalize();
  
	return 0;
}
