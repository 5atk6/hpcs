#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "data_util_bin.c"

#define G 1.0f
#define dt 1.0f
#define size 1024
#define step 10

//iとjの距離の計算
double distanceCalc(double *x,double *y,double *z,int i,int j){
  return sqrt((x[i]-x[j])*(x[i]-x[j])
	      +(y[i]-y[j])*(y[i]-y[j])
	      +(z[i]-z[j])*(z[i]-z[j]));
}

int main(int argc,char* argv[]){
  double r,
    *m,*x,*y,*z,*vx,*vy,*vz,
    *vxiNext,*vyiNext,*vziNext,
    *xiNext,*yiNext,*ziNext,
    axij,ayij,azij;
  int i,j,t;
  
  if(argc != 8){
    printf("m x y z vx vy vzのデータを入力してね\n");
    exit(1);
  }

  //データを読み込む
  m=(double*)malloc(sizeof(double)*size);
  x=(double*)malloc(sizeof(double)*size);
  y=(double*)malloc(sizeof(double)*size);
  z=(double*)malloc(sizeof(double)*size);
  vx=(double*)malloc(sizeof(double)*size);
  vy=(double*)malloc(sizeof(double)*size);
  vz=(double*)malloc(sizeof(double)*size);
  
  vxiNext=(double*)malloc(sizeof(double)*size);
  vyiNext=(double*)malloc(sizeof(double)*size);
  vziNext=(double*)malloc(sizeof(double)*size);
  xiNext=(double*)malloc(sizeof(double)*size);
  yiNext=(double*)malloc(sizeof(double)*size);
  ziNext=(double*)malloc(sizeof(double)*size);
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
  
  //vxiNext,vyiNext,vziNextの計算
  for(t=0;t<step;t++){
    for(i=0;i<size;i++){
      axij=0;
      ayij=0;
      azij=0;
      for(j=0;j<size;j++){
	if(i==j){
	  continue;
	}
	r=distanceCalc(x,y,z,i,j);
	axij+=G*(m[j]/(r*r))*((x[j]-x[i])/r);
	ayij+=G*(m[j]/(r*r))*((y[j]-y[i])/r);
	azij+=G*(m[j]/(r*r))*((z[j]-z[i])/r);
      }
      
      vxiNext[i]=vx[i]+axij;
      vyiNext[i]=vy[i]+ayij;
      vziNext[i]=vz[i]+azij;
    
      xiNext[i]=x[i]+vxiNext[i]*dt;
      yiNext[i]=y[i]+vyiNext[i]*dt;
      ziNext[i]=z[i]+vziNext[i]*dt;
    }
    memcpy(x,xiNext,sizeof(x)*size);
    memcpy(y,yiNext,sizeof(y)*size);
    memcpy(z,ziNext,sizeof(z)*size);
    memcpy(vx,vxiNext,sizeof(vx)*size);
    memcpy(vy,vyiNext,sizeof(vy)*size);
    memcpy(vz,vziNext,sizeof(vz)*size);
  }

  write_data(x,"xexTikuzi.dat",size);
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
  
  printf("success\n");
  return 0;
}



			 
					  
