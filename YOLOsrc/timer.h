#include <time.h>

void executeTime(clock_t start, clock_t end)
{

	double time_used;

	time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

	printf("executeTime %f\n", time_used );
}