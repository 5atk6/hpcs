#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>

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
    printf("usage:\t./data_compare_bin file1 file2 [precision]\n");
    exit(1);
  }

  /* fopen <stdio.h> : fopen関数はfilenameが指す文字列を名前とするファイルを 
     modeが指すモードでオープンし，そのファイルにストリーム (戻り値になっている) を結び付ける */
  if ((fp1 = fopen(argv[1], "r")) == NULL) {
    printf("%s:\tfile1 not found\n", __FILE__);
    exit(1);
  }

  if ((fp2 = fopen(argv[2], "r")) == NULL) {
    printf("%s:\tfile2 not found\n", __FILE__);
    exit(1);
  }

  if (argc == 4) {
    /* atof() <stdlib.h> : 文字列で表現された数値をdouble型の数値に変換 */
    double_min = atof(argv[3]);
  }

  /* stat() <sys/stat.h> : ファイルやディレクトリの状態を取得 */
  stat(argv[1], &stat1);
  stat(argv[2], &stat2);

  if (stat1.st_size != stat2.st_size) {
    printf("%s:\twrong file size\n", __FILE__);
    exit(1);
  }
  first = (char *)malloc(stat1.st_size);
  second = (char *)malloc(stat2.st_size);

  ended = 0;
  while (1) {
    /* fread() <stdio.h> : ファイルfp1から1バイトのデータを
       (stat1.st_size - ended)個読み込み，バッファ(first + ended)に格納*/
    ended += fread(first + ended, 1, stat1.st_size - ended, fp1);
    if (ended >= stat1.st_size)
      break;
  }
  fclose(fp1);

  ended = 0;
  while (1) {
    ended += fread(second + ended, 1, stat2.st_size - ended, fp2);
    if (ended >= stat2.st_size)
      break;
  }
  fclose(fp2);

  dsize = stat1.st_size / (sizeof(double) / sizeof(char));
  df = (double *)first;
  ds = (double *)second;
  flag = 0;
  for (i = 0; i < dsize; ++i) {
    /* isnan() <math.h> : 引数の値がNaN(非数)かどうかを判定 */
    if (isnan(df[i]) || isnan(ds[i]) 
	|| (df[i] - ds[i]) / df[i] > double_min 
	|| (ds[i] - df[i]) / ds[i] > double_min) {
      printf("error:\n\ta[%d]=%.20e\n\tb[%d]=%.20e\n", i, df[i], i, ds[i]);
      ++flag;
    }
  }
  if (flag == 0) printf("No error. Same data. (different less than %1.20f.)\n", double_min);
  else printf("%d errors in %d. different data. \n", flag, i);

  free(first);
  free(second);
  return 0;
}
