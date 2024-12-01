#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>

// Struktura cvora binarnog stabla
struct tree {
    int data;
    struct tree* left;
    struct tree* right;
};

// Funkcija za kreiranje novog cvora
struct tree* create_node(int data) {
    struct tree* node = (struct tree*)malloc(sizeof(struct tree));
    node->data = data;
    node->left = NULL;
    node->right = NULL;
    return node;
}

// Funkcija za generisanje binarnog stabla sa zadatom dubinom
struct tree* generate_tree(int depth) {
    if (depth <= 0) return NULL;

    struct tree* node = create_node(depth % 10);
    node->left = generate_tree(depth - 1);
    node->right = generate_tree(depth - 1);
    return node;
}

// Sekvencijalna pretraga
void search_tree_sequential(struct tree* root, int target, int* count) {
    if (root == NULL) return;

    *count += (root->data == target) ? 1 : 0;

    search_tree_sequential(root->left, target, count);
    search_tree_sequential(root->right, target, count);
}

// Paralelna pretraga
void search_tree_parallel(struct tree* root, int target, int depth, int threshold, int* count) {
    if (root == NULL) return;

    int local_count = (root->data == target) ? 1 : 0;
    int left_count = 0, right_count = 0;

    if (depth > threshold) {
        #pragma omp task shared(left_count)
        search_tree_parallel(root->left, target, depth - 1, threshold, &left_count);

        #pragma omp task shared(right_count)
        search_tree_parallel(root->right, target, depth - 1, threshold, &right_count);

        #pragma omp taskwait
    } else {
        search_tree_sequential(root->left, target, &left_count);
        search_tree_sequential(root->right, target, &right_count);
    }

    #pragma omp atomic
    *count += local_count + left_count + right_count;
}

// Sekvencijalno sumiranje
int calculate_sum_sequential(struct tree* root) {
    if (root == NULL) return 0;
    return root->data + calculate_sum_sequential(root->left) + calculate_sum_sequential(root->right);
}

// Paralelno sumiranje
int calculate_sum_parallel(struct tree* root, int depth, int threshold) {
    if (root == NULL) return 0;

    int local_sum = root->data;
    int left_sum = 0, right_sum = 0;

    if (depth > threshold) {
        #pragma omp task shared(left_sum)
        left_sum = calculate_sum_parallel(root->left, depth - 1, threshold);

        #pragma omp task shared(right_sum)
        right_sum = calculate_sum_parallel(root->right, depth - 1, threshold);

        #pragma omp taskwait
    } else {
        left_sum = calculate_sum_sequential(root->left);
        right_sum = calculate_sum_sequential(root->right);
    }

    return local_sum + left_sum + right_sum;
}

// Sekvencijalno brojanje parnih i neparnih vrednosti
void count_even_odd_sequential(struct tree* root, int* even_count, int* odd_count) {
    if (root == NULL) return;

    if (root->data % 2 == 0) {
        (*even_count)++;
    } else {
        (*odd_count)++;
    }

    count_even_odd_sequential(root->left, even_count, odd_count);
    count_even_odd_sequential(root->right, even_count, odd_count);
}

// Paralelno brojanje parnih i neparnih vrednosti
void count_even_odd_parallel(struct tree* root, int* even_count, int* odd_count, int depth, int threshold) {
    if (root == NULL) return;

    int local_even = 0, local_odd = 0;

    if (root->data % 2 == 0)
        local_even++;
    else
        local_odd++;

    int left_even = 0, left_odd = 0;
    int right_even = 0, right_odd = 0;

    if (depth > threshold) {
        #pragma omp task shared(left_even, left_odd)
        count_even_odd_parallel(root->left, &left_even, &left_odd, depth - 1, threshold);

        #pragma omp task shared(right_even, right_odd)
        count_even_odd_parallel(root->right, &right_even, &right_odd, depth - 1, threshold);

        #pragma omp taskwait
    } else {
        count_even_odd_sequential(root->left, &left_even, &left_odd);
        count_even_odd_sequential(root->right, &right_even, &right_odd);
    }

    #pragma omp atomic
    *even_count += local_even + left_even + right_even;

    #pragma omp atomic
    *odd_count += local_odd + left_odd + right_odd;
}

// Sekvencijalno pronalazenje minimuma i maksimuma
void find_min_max_sequential(struct tree* root, int* min_val, int* max_val) {
    if (root == NULL) return;

    if (root->data < *min_val) *min_val = root->data;
    if (root->data > *max_val) *max_val = root->data;

    find_min_max_sequential(root->left, min_val, max_val);
    find_min_max_sequential(root->right, min_val, max_val);
}

// Paralelno pronalazenje minimuma i maksimuma
void find_min_max_parallel(struct tree* root, int* min_val, int* max_val, int depth, int threshold) {
    if (root == NULL) return;

    int local_min = root->data;
    int local_max = root->data;

    int left_min = INT_MAX, right_min = INT_MAX;
    int left_max = INT_MIN, right_max = INT_MIN;

    if (depth > threshold) {
        #pragma omp task shared(left_min, left_max)
        find_min_max_parallel(root->left, &left_min, &left_max, depth - 1, threshold);

        #pragma omp task shared(right_min, right_max)
        find_min_max_parallel(root->right, &right_min, &right_max, depth - 1, threshold);

        #pragma omp taskwait
    } else {
        find_min_max_sequential(root->left, &left_min, &left_max);
        find_min_max_sequential(root->right, &right_min, &right_max);
    }

    local_min = local_min < left_min ? (local_min < right_min ? local_min : right_min) : (left_min < right_min ? left_min : right_min);
    local_max = local_max > left_max ? (local_max > right_max ? local_max : right_max) : (left_max > right_max ? left_max : right_max);

    #pragma omp critical
    {
        if (local_min < *min_val) *min_val = local_min;
        if (local_max > *max_val) *max_val = local_max;
    }
}

// Funkcija za testiranje sa odredjenim brojem niti
void test_with_threads(struct tree* root, int target, int depth, int threshold, int num_threads) {
    omp_set_num_threads(num_threads);
    printf("__________ Testiranje sa %d niti __________\n\n", num_threads);

    // Testiranje pretrage
    int seq_count = 0;
    double start_time = omp_get_wtime();
    search_tree_sequential(root, target, &seq_count);
    double seq_search_time = omp_get_wtime() - start_time;

    int par_count = 0;
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        search_tree_parallel(root, target, depth, threshold, &par_count);
    }
    double par_search_time = omp_get_wtime() - start_time;

    printf("Pretraga: Serijska = %d, Paralelna = %d\n", seq_count, par_count);
    printf("Ubrzanje pretrage: %.2f\n", seq_search_time / par_search_time);

    // Testiranje sumiranja
    start_time = omp_get_wtime();
    int seq_sum = calculate_sum_sequential(root);
    double seq_sum_time = omp_get_wtime() - start_time;

    start_time = omp_get_wtime();
    int par_sum = 0;
    #pragma omp parallel
    {
        #pragma omp single
        par_sum = calculate_sum_parallel(root, depth, threshold);
    }
    double par_sum_time = omp_get_wtime() - start_time;

    printf("Sumiranje: Serijska = %d, Paralelna = %d\n", seq_sum, par_sum);
    printf("Ubrzanje sumiranja: %.2f\n", seq_sum_time / par_sum_time);

    // Testiranje brojanja parnih i neparnih vrednosti
    int seq_even = 0, seq_odd = 0;
    start_time = omp_get_wtime();
    count_even_odd_sequential(root, &seq_even, &seq_odd);
    double seq_count_time = omp_get_wtime() - start_time;

    int par_even = 0, par_odd = 0;
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        count_even_odd_parallel(root, &par_even, &par_odd, depth, threshold);
    }
    double par_count_time = omp_get_wtime() - start_time;

    printf("Brojanje: Serijski (Parni = %d, Neparni = %d), Paralelni (Parni = %d, Neparni = %d)\n",
           seq_even, seq_odd, par_even, par_odd);
    printf("Ubrzanje brojanja: %.2f\n", seq_count_time / par_count_time);

    // Testiranje pronalazenja minimuma i maksimuma
    int seq_min = INT_MAX, seq_max = INT_MIN;
    start_time = omp_get_wtime();
    find_min_max_sequential(root, &seq_min, &seq_max);
    double seq_min_max_time = omp_get_wtime() - start_time;

    int par_min = INT_MAX, par_max = INT_MIN;
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        find_min_max_parallel(root, &par_min, &par_max, depth, threshold);
    }
    double par_min_max_time = omp_get_wtime() - start_time;

    printf("Min/Max: Serijski (Min = %d, Max = %d), Paralelni (Min = %d, Max = %d)\n",
           seq_min, seq_max, par_min, par_max);
    printf("Ubrzanje Min/Max: %.2f\n", seq_min_max_time / par_min_max_time);

    printf("___________________________________________\n\n");
}

int main() {
    int depth = 24;         // Dubina stabla
    int threshold = 12;     // Prag za paralelizaciju
    int target = 3;         // Trazeni podatak

    // Generisanje stabla
    struct tree* root = generate_tree(depth);

    // Testiranje sa 2, 4 i 6 niti
    test_with_threads(root, target, depth, threshold, 2);
    test_with_threads(root, target, depth, threshold, 4);
    test_with_threads(root, target, depth, threshold, 6);

    return 0;
}

