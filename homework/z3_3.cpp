/* 
Student : Aleksandar Vig

Zadatak 3.3 : Skalarni proizvod dva vektora, vektore generise program na slucajan nacin

mpic++ -o z3_3 z3_3.cpp
mpiexec -n 4 ./z3_3
*/
#include <stdio.h>
#include <mpi.h>
#include <cmath>
#include <time.h>
#include <vector>

int getInput() {
    int res;
    printf("Vector length: ");
    fflush(stdout);
    scanf("%d", &res);
    return res;
}

int main(int argc, char* argv[]) {
    int n;                    // Duzina vektora
    int local_sum = 0;        // Lokalni rezultat skalarnog proizvoda
    int global_sum = 0;       // Globalni rezultat skalarnog proizvoda
    int csize, prank;         // Broj procesa i rang procesa
    int block_size, start, finish;

    std::vector<int> vector1, vector2; // Glavni vektori

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &csize);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    // Proces 0 dobija ulaznu velicinu vektora
    if (prank == 0) {
        n = getInput();
    }

    // Broadcast duzine vektora svim procesima
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Generisanje vektora na procesu 0
    if (prank == 0) {
        srand(time(NULL));
        vector1.resize(n);
        vector2.resize(n);

        for (int i = 0; i < n; i++) {
            vector1[i] = rand() % 20; 
            vector2[i] = rand() % 20;
        }
    } else {
        vector1.resize(n);
        vector2.resize(n);
    }

    // Broadcast vektora svim procesima
    MPI_Bcast(vector1.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector2.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime(); // Merenje vremena

    // Blokovska strategija
    block_size = ceil(static_cast<float>(n) / csize);
    start = prank * block_size;
    finish = std::min(start + block_size, n);

    // Lokalno mnozenje
    for (int i = start; i < finish; i++) {
        local_sum += vector1[i] * vector2[i];
    }

    // Redukcija lokalnih rezultata u globalni rezultat na procesu 0
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // Analiza najsporijeg procesa
    double max_time;
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Ispis rezultata na procesu 0
    if (prank == 0) {
        printf("\nScalar product of vectors:\n");
        printf("Vector 1: { ");
        for (int val : vector1) {
            printf("%d ", val);
        }
        printf("}\n");

        printf("Vector 2: { ");
        for (int val : vector2) {
            printf("%d ", val);
        }
        printf("}\n");

        printf("\nResult: %d\n", global_sum);
        printf("Elapsed time: %.6f seconds\n", max_time);
    }

    MPI_Finalize();
    return 0;
}

