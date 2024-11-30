/* 
Student : Aleksandar Vig

Zadatak 6.2 : Eratostenovo sito

g++ -fopenmp -Wall -o z6_2 z6_2.cpp
./z6_2
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <bits/stdc++.h>
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char* argv[])
{
    int n, strategy;
    std::vector<int> res;

    printf("Unesite gornju granicu za kalkulaciju prostih brojeva: ");
    scanf("%d", &n);

    printf("\nIzaberite strategiju:");
    printf("\n1 - Static");
    printf("\n2 - Dynamic");
    printf("\n3 - Guided");
    printf("\n4 - Auto");
    printf("\n5 - Runtime\n");
    scanf("%d", &strategy);

    const char* strategije[] = {"Static", "Dynamic", "Guided", "Auto", "Runtime"};

    if (strategy < 1 || strategy > 5) {
        printf("\nNeispravan izbor strategije.\n");
        return 1;
    }

    printf("\nIskoriscena je strategija: %s\n", strategije[strategy - 1]);

    bool prime[n + 1];
    memset(prime, true, sizeof(prime));
    double s = omp_get_wtime();
    int temp = (int)ceil(sqrt(n));

    switch (strategy)
    {
    case 1: // Static
        #pragma omp parallel for schedule(static, 1)
        for (int p = 2; p <= temp; p++)
        {
            if (prime[p] == true)
            {
                for (int i = p * p; i <= n; i += p)
                    prime[i] = false;
            }
        }
        break;
    case 2: // Dynamic
        #pragma omp parallel for schedule(dynamic, 1)
        for (int p = 2; p <= temp; p++)
        {
            if (prime[p] == true)
            {
                for (int i = p * p; i <= n; i += p)
                    prime[i] = false;
            }
        }
        break;
    case 3: // Guided
        #pragma omp parallel for schedule(guided, 1)
        for (int p = 2; p <= temp; p++)
        {
            if (prime[p] == true)
            {
                for (int i = p * p; i <= n; i += p)
                    prime[i] = false;
            }
        }
        break;
    case 4: // Auto
        #pragma omp parallel for schedule(auto)
        for (int p = 2; p <= temp; p++)
        {
            if (prime[p] == true)
            {
                for (int i = p * p; i <= n; i += p)
                    prime[i] = false;
            }
        }
        break;
    case 5: // Runtime
        #pragma omp parallel for schedule(runtime)
        for (int p = 2; p <= temp; p++)
        {
            if (prime[p] == true)
            {
                for (int i = p * p; i <= n; i += p)
                    prime[i] = false;
            }
        }
        break;
    }

    // Sakupljanje prostih brojeva
    for (int p = 2; p <= n; p++)
        if (prime[p] == true)
        {
            res.push_back(p);
        }

    s = omp_get_wtime() - s;

    printf("\nVreme izvrsenja je : %.6lf sekundi\n", s);

    // Cuvanje rezultata u fajl prosti_brojevi.txt
    FILE *fp = fopen("prosti_brojevi.txt", "w");
    if (fp == NULL)
    {
        printf("Greska prilikom otvaranja fajla.\n");
        return 1;
    }

    for (int i = 0; i < (int)res.size(); i++)
    {
        fprintf(fp, "%d ", res[i]);
    }
    fclose(fp);

    printf("Rezultati su sacuvani u fajlu: prosti_brojevi.txt.\n");
    return 0;
}

