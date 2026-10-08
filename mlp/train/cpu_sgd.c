/*
 * cpu_sgd.c : optimised CPU baseline for the on-chip trainer.
 *
 * Same algorithm as the hardware: 64 -> 16 (ReLU) -> 10, MSE loss,
 * per-sample SGD, same init, same visiting order, lr = 2^-7. float32,
 * plain loops that gcc -O3 -march=native auto-vectorises.
 *
 * usage: cpu_sgd <dir> <nsamp> <nsteps> <reps>
 *   reads  <dir>/x.f32 (nsamp*64) y.i32 order.i32 w1.f32 b1.f32 w2.f32 b2.f32
 *   writes <dir>/w1_c.f32 b1_c.f32 w2_c.f32 b2_c.f32, prints best time (s)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N0 64
#define N1 16
#define N2 10

static void *load(const char *dir, const char *name, size_t bytes) {
    char path[512]; snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    void *p = malloc(bytes);
    if (fread(p, 1, bytes, f) != bytes) { fprintf(stderr, "short %s\n", path); exit(1); }
    fclose(f); return p;
}
static void save(const char *dir, const char *name, const void *p, size_t bytes) {
    char path[512]; snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb"); fwrite(p, 1, bytes, f); fclose(f);
}

int main(int argc, char **argv) {
    const char *dir = argv[1];
    int ns = atoi(argv[2]), nsteps = atoi(argv[3]), reps = atoi(argv[4]);
    const float lr = 1.0f / 128.0f;
    float *X  = load(dir, "x.f32",  sizeof(float) * ns * N0);
    int   *Y  = load(dir, "y.i32",  sizeof(int) * ns);
    int   *O  = load(dir, "order.i32", sizeof(int) * nsteps);
    float *W1i = load(dir, "w1.f32", sizeof(float) * N1 * N0);
    float *B1i = load(dir, "b1.f32", sizeof(float) * N1);
    float *W2i = load(dir, "w2.f32", sizeof(float) * N2 * N1);
    float *B2i = load(dir, "b2.f32", sizeof(float) * N2);

    static float W1[N1][N0], B1[N1], W2[N2][N1], B2[N2];
    double best = 1e30;
    for (int rep = 0; rep < reps; rep++) {
        memcpy(W1, W1i, sizeof W1); memcpy(B1, B1i, sizeof B1);
        memcpy(W2, W2i, sizeof W2); memcpy(B2, B2i, sizeof B2);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int s = 0; s < nsteps; s++) {
            const float *x = &X[(size_t)O[s] * N0];
            float a[N1], h[N1], z[N2], e2[N2], e1[N1];
            for (int j = 0; j < N1; j++) {                 /* forward L1 */
                float acc = B1[j];
                for (int k = 0; k < N0; k++) acc += W1[j][k] * x[k];
                a[j] = acc; h[j] = acc > 0 ? acc : 0;
            }
            for (int i = 0; i < N2; i++) {                 /* forward L2 */
                float acc = B2[i];
                for (int j = 0; j < N1; j++) acc += W2[i][j] * h[j];
                z[i] = acc;
                e2[i] = z[i] - (i == Y[O[s]] ? 1.0f : 0.0f);   /* MSE grad */
            }
            for (int j = 0; j < N1; j++) {                 /* backprop */
                float acc = 0;
                for (int i = 0; i < N2; i++) acc += W2[i][j] * e2[i];
                e1[j] = a[j] > 0 ? acc : 0;
            }
            for (int i = 0; i < N2; i++) {                 /* update L2 */
                for (int j = 0; j < N1; j++) W2[i][j] -= lr * e2[i] * h[j];
                B2[i] -= lr * e2[i];
            }
            for (int j = 0; j < N1; j++) {                 /* update L1 */
                for (int k = 0; k < N0; k++) W1[j][k] -= lr * e1[j] * x[k];
                B1[j] -= lr * e1[j];
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        if (dt < best) best = dt;
    }
    save(dir, "w1_c.f32", W1, sizeof W1); save(dir, "b1_c.f32", B1, sizeof B1);
    save(dir, "w2_c.f32", W2, sizeof W2); save(dir, "b2_c.f32", B2, sizeof B2);
    printf("%.9f\n", best);
    return 0;
}
