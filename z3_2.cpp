/* 
Student : Aleksandar Vig

Zadatak 3.2 : Sumiranje prvih N prirodnih brojeva upotrebom blokovske strategije

mpic++ -o z3_2 z3_2.cpp
mpiexec -n 4 ./z3_2
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int prank, csize;
    int N; // Ukupan broj prirodnih brojeva za sumiranje
    int local_sum = 0; // Parcijalna suma svakog procesa
    int global_sum = 0; // Konačna suma

    // Inicijalizacija MPI okruzenja
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &csize);

    // Proces 0 dobija ulaznu vrednost N
    if (prank == 0) {
        printf("Enter N (number of natural numbers to sum): ");
        fflush(stdout);
        scanf("%d", &N);
    }

    // Broadcast vrednosti N svim procesima
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Odredjivanje granica za blokove (blokovska strategija)
    int block_size = N / csize;
    int start = prank * block_size + 1;
    int end = (prank == csize - 1) ? N : start + block_size - 1;

    // Racunanje lokalne sume
    for (int i = start; i <= end; i++) {
        local_sum += i;
    }

    // Sakupljanje svih parcijalnih suma u globalnu sumu na procesu 0
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Proces 0 ispisuje rezultat
    if (prank == 0) {
        printf("The sum of the first %d natural numbers is %d\n", N, global_sum);
    }

    // Finalizacija MPI okruženja
    MPI_Finalize();
    return 0;
}

