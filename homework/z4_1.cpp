/* 
Student : Aleksandar Vig

Zadatak 4.1: Mnozenje matrice i vektora proizvoljne velicine n

mpic++ -o z4_1 z4_1.cpp

./z4_1 vector.txt matrix.txt
*/

#include <stdio.h>
#include <mpi.h>

// Funkcija za izracunavanje dimenzije ulaznog fajla
int returnSize(char *fname) {
    FILE *f = fopen(fname, "r");
    int dim = 0;
    double tmp;
    while (fscanf(f, "%lf", &tmp) != EOF)
        dim++;
    fclose(f);
    return dim;
}

// Funkcija za ucitavanje vektora
double *loadVec(char *fname, int n) {
    FILE *f = fopen(fname, "r");
    double *res = new double[n];
    double *it = res;
    while (fscanf(f, "%lf", it++) != EOF);
    fclose(f);
    return res;
}

// Funkcija za ucitavanje matrice
double *loadMat(char *fname, int n) {
    FILE *f = fopen(fname, "r");
    double *res = new double[n * n];
    double *it = res;
    while (fscanf(f, "%lf", it++) != EOF);
    fclose(f);
    return res;
}

// Funkcija za logovanje rezultata u fajl
void logRes(const char *fname, double *res, int n) {
    FILE *f = fopen(fname, "w");
    for (int i = 0; i != n; ++i)
        fprintf(f, "%lf ", res[i]);
    fclose(f);
}

int main(int argc, char *argv[]) {
    int csize;
    int prank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &csize);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    char *vfname = argv[1];
    char *mfname = argv[2];
    int dim;
    double *mat;
    double *vec;
    double *tmat;
    double *lres;
    double *res;

    if (prank == 0)
        dim = returnSize(vfname);

    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (prank == 0)
        vec = loadVec(vfname, dim);
    else
        vec = new double[dim];

    MPI_Bcast(vec, dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (prank == 0)
        tmat = loadMat(mfname, dim);

    int to = dim / csize; // Broj redova po procesu
    mat = new double[to * dim]; // Matrica koju svaki procesor dobija

    MPI_Scatter(tmat, to * dim, MPI_DOUBLE, mat, to * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    lres = new double[to]; // Lokalni rezultat

    for (int i = 0; i != to; ++i) {
        double s = 0;
        for (int j = 0; j != dim; ++j)
            s += mat[i * dim + j] * vec[j];
        lres[i] = s;
    }

    if (prank == 0)
        res = new double[dim];

    MPI_Gather(lres, to, MPI_DOUBLE, res, to, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (prank == 0) {
        // Obrada ostatka ako matrica nije deljiva sa brojem procesa
        for (int i = csize * to; i != dim; ++i) {
            double ostatak = 0;
            for (int j = 0; j != dim; ++j)
                ostatak += tmat[i * dim + j] * vec[j];
            res[i] = ostatak;
        }
        logRes("res.txt", res, dim);
    }

    if (prank == 0) {
        delete[] tmat;
        delete[] res;
    }

    delete[] vec;
    delete[] mat;
    delete[] lres;
    MPI_Finalize();
    return 0;
}

