/* 
Student : Aleksandar Vig

Zadatak 2.4 : Svaki proces salje svakom drugom procesu poruku od jednog prirodnog dvocifrenog broja

mpic++ -o z2_4 z2_4.cpp
mpiexec -n 4 ./z2_4
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    int prank, csize;

    // Inicijalizacija MPI okruzenja
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &csize);

    // Seed za generisanje random brojeva
    srand(time(NULL) + prank);

    // Kreiraj niz za cuvanje poruka
    int received[csize];
    for (int i = 0; i < csize; i++) {
        received[i] = -1; // Inicijalna vrednost (nije primljeno)
    }

    // Svaki proces salje poruku svim drugim procesima
    for (int i = 0; i < csize; i++) {
        if (i != prank) {
            int random_digit = rand() % 10; // Nasumičan broj od 0 do 9
            int message = prank * 10 + random_digit;

            MPI_Send(&message, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }

    // Svaki proces prima poruke od svih drugih procesa
    for (int i = 0; i < csize; i++) {
        if (i != prank) {
            int received_message;
            MPI_Recv(&received_message, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            received[i] = received_message;
        }
    }

    // Ispis poruka koje je proces primio
    printf("Process %d received:", prank);
    for (int i = 0; i < csize; i++) {
        if (i != prank) {
            printf(" %d", received[i]);
        }
    }
    printf("\n");

    // Finalizacija MPI okruzenja
    MPI_Finalize();
    return 0;
}

