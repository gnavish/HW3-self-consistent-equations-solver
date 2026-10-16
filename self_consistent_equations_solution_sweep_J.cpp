// Sweep J1 = J2 = J for the pi-flux Z2 Schwinger-boson mean-field theory (B = 0) and write a CSV:
//     J1=J2,A1,A2,mu,gap
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_errno.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Params {
    double J1, J2;
    std::vector<double> s2, c2;   // sin^2kx + sin^2ky and cos^2kx cos^2ky on the k-grid
};

static double d2_at(const Params* P, size_t i, double A1, double A2) {
    return P->J1 * P->J1 * A1 * A1 * P->s2[i] + 4 * P->J2 * P->J2 * A2 * A2 * P->c2[i];
}

static double mu_of(const Params* P, double A1, double A2, double t) {
    double M = 0;
    for (size_t i = 0; i < P->s2.size(); ++i) M = std::max(M, d2_at(P, i, A1, A2));
    return std::sqrt(M) * (1 + std::exp(t));
}

static int residuals(const gsl_vector* x, void* p, gsl_vector* F) {
    auto* P = static_cast<Params*>(p);
    double A1 = gsl_vector_get(x, 0), A2 = gsl_vector_get(x, 1), t = gsl_vector_get(x, 2);
    double mu = mu_of(P, A1, A2, t);
    double S0 = 0, S1 = 0, S2 = 0;
    size_t n = P->s2.size();
    for (size_t i = 0; i < n; ++i) {
        double w = std::sqrt(mu * mu - d2_at(P, i, A1, A2));
        S0 += P->s2[i] / w; S1 += P->c2[i] / w; S2 += 1 / w;
    }
    gsl_vector_set(F, 0, P->J1 / 4 * S0 / n - 1);
    gsl_vector_set(F, 1, P->J2 * S1 / n - 1);
    gsl_vector_set(F, 2, mu / 2 * S2 / n - 1);
    return GSL_SUCCESS;
}

struct Result { double A1, A2, mu, gap, t; int status; int iters; };

// Solve at one coupling; (A1, A2, t) is the starting guess.
static Result solve(Params& P, double A1, double A2, double t) {
    gsl_multiroot_function FN = {&residuals, 3, &P};
    gsl_vector* x = gsl_vector_alloc(3);
    gsl_vector_set(x, 0, A1); gsl_vector_set(x, 1, A2); gsl_vector_set(x, 2, t);
    gsl_multiroot_fsolver* s = gsl_multiroot_fsolver_alloc(gsl_multiroot_fsolver_hybrids, 3);
    gsl_multiroot_fsolver_set(s, &FN, x);

    int status, iter = 0;
    do {
        ++iter;
        status = gsl_multiroot_fsolver_iterate(s);
        if (status) break;
        status = gsl_multiroot_test_residual(s->f, 1e-12);
    } while (status == GSL_CONTINUE && iter < 500);

    Result r;
    r.A1 = gsl_vector_get(s->x, 0); r.A2 = gsl_vector_get(s->x, 1); r.t = gsl_vector_get(s->x, 2);
    r.mu = mu_of(&P, r.A1, r.A2, r.t);
    r.gap = 1e300;
    for (size_t i = 0; i < P.s2.size(); ++i)
        r.gap = std::min(r.gap, std::sqrt(r.mu * r.mu - d2_at(&P, i, r.A1, r.A2)));
    r.status = status; r.iters = iter;
    gsl_multiroot_fsolver_free(s);
    gsl_vector_free(x);
    return r;
}

int main(int argc, char** argv) {
    double Jmin = argc > 1 ? std::atof(argv[1]) : 0.1;
    double Jmax = argc > 2 ? std::atof(argv[2]) : 2.0;
    int    n    = argc > 3 ? std::atoi(argv[3]) : 20;
    int    L    = argc > 4 ? std::atoi(argv[4]) : 64;
    const char* outname = argc > 5 ? argv[5] : "sweep_J.csv";
    if (n < 1 || L < 2 || L % 2) { std::fprintf(stderr, "need n >= 1 and even L >= 2\n"); return 1; }

    gsl_set_error_handler_off();   // report failures ourselves instead of aborting

    Params P{1.0, 1.0, {}, {}};
    for (int i = 0; i < L; ++i)              // kx in [-pi, pi)
        for (int j = 0; j < L / 2; ++j) {    // ky in [-pi/2, pi/2)
            double kx = -M_PI + 2 * M_PI * i / L, ky = -M_PI / 2 + 2 * M_PI * j / L;
            P.s2.push_back(std::sin(kx) * std::sin(kx) + std::sin(ky) * std::sin(ky));
            P.c2.push_back(std::pow(std::cos(kx) * std::cos(ky), 2));
        }

    FILE* out = std::fopen(outname, "w");
    if (!out) { std::perror(outname); return 1; }
    std::fprintf(out, "J1=J2,A1,A2,mu,gap\n");

    // Warm start: each J begins from the previous converged solution (A1, A2, t are J-independent
    // when J1 = J2, so later points converge almost immediately). Fall back to a cold start on failure.
    double gA1 = 0.3, gA2 = 0.3, gt = -3.0;
    int failures = 0;
    for (int k = 0; k < n; ++k) {
        double J = (n == 1) ? Jmin : Jmin + (Jmax - Jmin) * k / (n - 1);
        P.J1 = J; P.J2 = J;

        Result r = solve(P, gA1, gA2, gt);
        if (r.status != GSL_SUCCESS) r = solve(P, 0.3, 0.3, -3.0);
        if (r.status != GSL_SUCCESS) {
            std::printf("J = %.6f  FAILED (%s) - not written to CSV\n", J, gsl_strerror(r.status));
            ++failures;
            continue;
        }
        gA1 = r.A1; gA2 = r.A2; gt = r.t;
        std::fprintf(out, "%.8f,%.8f,%.8f,%.8f,%.8f\n", J, std::fabs(r.A1), std::fabs(r.A2), r.mu, r.gap);
        std::printf("J = %.6f  A1 = %.6f  A2 = %.6f  mu = %.6f  gap = %.6f  (%d iters)\n",
                    J, std::fabs(r.A1), std::fabs(r.A2), r.mu, r.gap, r.iters);
    }
    std::fclose(out);
    std::printf("wrote %s (%d of %d points, L = %d)\n", outname, n - failures, n, L);
    return failures ? 2 : 0;
}