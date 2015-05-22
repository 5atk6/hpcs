#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <math.h>

#define DOUBLE_MIN (1e-10)

int main(int argc, char* argv[])
{
	char *first, *second;
	int i, ended, flag;
	FILE *fp1, *fp2;
	struct stat stat1, stat2;
	double *df, *ds;
	int dsize;

	double double_min;
	double_min = DOUBLE_MIN;

	if (argc < 3 || argc > 4) {
		printf("usage:  comparedouble file1 file2 [precision]\n");
	}

	if ((fp1 = fopen(argv[1], "r")) == NULL) {
		printf("file not found\n");
		exit(1);
	}

	if ((fp2 = fopen(argv[2], "r")) == NULL) {
		printf("file not found\n");
		exit(1);
	}

	if (argc == 4) {
		double_min = atof(argv[3]);
	}

	stat(argv[1], &stat1);
	stat(argv[2], &stat2);

	if (stat1.st_size != stat2.st_size) {
		printf("wrong file size\n");
		exit(1);
	}
	first = (char*)malloc(stat1.st_size);
	second = (char*)malloc(stat2.st_size);

	ended = 0;
	while (1) {
		ended += fread(first + ended, 1, stat1.st_size-ended, fp1);
		if (ended >= stat1.st_size)
			break;
	}
	fclose(fp1);

	ended = 0;
	while (1) {
		ended += fread(second + ended, 1, stat2.st_size-ended, fp2);
		if (ended>=stat2.st_size)
			break;
	}
	fclose(fp2);

	dsize = stat1.st_size / (sizeof(double) / sizeof(char));
	df = (double *)first;
	ds = (double *)second;
	flag = 0;
	for (i = 0; i < dsize; i++) {
		if(isnan(df[i]) || isnan(ds[i]) || (df[i] - ds[i]) / df[i] > double_min){
			printf("error a[%d]=%.20e\n      b[%d]=%.20e\n",i,df[i],i,ds[i]);
			flag ++;
		}
	}
	if (flag==0)
		printf("no error. same data.(different less than %1.20f)\n", double_min);
	else
		printf("%d errors in %d. different data. \n",flag,i);

	free(first);
	free(second);
	return 0;
}
