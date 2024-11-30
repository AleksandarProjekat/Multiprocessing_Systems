/* 
Student : Aleksandar Vig

Zadatak 5.5 : Sumiranje prvih N prirodnih brojeva tako sto svaka nit racuna svoju parcijalnu sumu.
              Potom se konacna suma racuna sumiranjem tako dobijenih parcijalnih suma.

g++ -fopenmp -Wall -o z5_5 z5_5.cpp
./z5_5 4
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    int tc = strtol(argv[1], NULL, 10); // broj niti kao argument
    double n, sum = 0;

    printf("Number: ");
    scanf("%lf", &n);

    double s1 = omp_get_wtime();

    // Primer sa direktivom reduction
    #pragma omp parallel for num_threads(tc) reduction(+:sum)
    for (int i = 1; i <= (int)n; i++) {
        sum += (double)i;
    }

    s1 = omp_get_wtime() - s1;

    printf("\nSum using reduction is %lf\n", sum);
    printf("Executed in %lf seconds\n", s1);

    // Realizacija sa parcijalnim sumama
    sum = 0; 
    double s2 = omp_get_wtime();

    double *partial_sums = (double *)calloc(tc, sizeof(double));

    #pragma omp parallel num_threads(tc)
    {
        int tid = omp_get_thread_num(); // Identifikator niti
        int chunk = (int)n / tc;
        int start = tid * chunk + 1;
        int end = (tid == tc - 1) ? (int)n : (tid + 1) * chunk;

        for (int i = start; i <= end; i++) {
            partial_sums[tid] += (double)i;
        }
    }

    // Sabiranje parcijalnih suma
    for (int i = 0; i < tc; i++) {
        sum += partial_sums[i];
    }

    free(partial_sums);

    s2 = omp_get_wtime() - s2;

    printf("\nSum using partial sums is %lf\n", sum);
    printf("Executed in %lf seconds\n", s2);

    return 0;
}

