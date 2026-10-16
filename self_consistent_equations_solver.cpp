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
struct Params { double J1, J2; std::vector<double> s2, c2; };
static double mu_of(const Params* P, double A1, double A2, double t) {
    double M = 0;
    for (size_t i = 0; i < P->s2.size(); ++i)
        M = std::max(M, P->J1*P->J1*A1*A1*P->s2[i] + 4*P->J2*P->J2*A2*A2*P->c2[i]);
    return std::sqrt(M) * (1 + std::exp(t));
}
static int f(const gsl_vector* x, void* p, gsl_vector* F) {
    auto* P = static_cast<Params*>(p);
    double A1 = gsl_vector_get(x,0), A2 = gsl_vector_get(x,1), t = gsl_vector_get(x,2);
    double mu = mu_of(P, A1, A2, t), S0=0, S1=0, S2=0; size_t n = P->s2.size();
    for (size_t i = 0; i < n; ++i) {
        double d2 = P->J1*P->J1*A1*A1*P->s2[i] + 4*P->J2*P->J2*A2*A2*P->c2[i];
        double w = std::sqrt(mu*mu - d2);
        S0 += P->s2[i]/w; S1 += P->c2[i]/w; S2 += 1/w;
    }
    gsl_vector_set(F,0, P->J1/4*S0/n - 1);
    gsl_vector_set(F,1, P->J2*S1/n - 1);
    gsl_vector_set(F,2, mu/2*S2/n - 1);
    return GSL_SUCCESS;
}
int main(int argc, char** argv) {
    int L = argc > 1 ? std::atoi(argv[1]) : 64;
    double J1 = argc > 2 ? std::atof(argv[2]) : 1;
    double J2 = argc > 3 ? std::atof(argv[3]) : 1;
    Params P{J1, J2, {}, {}};
    for (int i = 0; i < L; ++i) for (int j = 0; j < L/2; ++j) {
        double kx = -M_PI + 2*M_PI*i/L, ky = -M_PI/2 + 2*M_PI*j/L;
        P.s2.push_back(sin(kx)*sin(kx) + sin(ky)*sin(ky));
        P.c2.push_back(pow(cos(kx)*cos(ky), 2));
    }
    gsl_multiroot_function FN = {&f, 3, &P};
    gsl_vector* x = gsl_vector_alloc(3);
    gsl_vector_set(x,0,0.3); gsl_vector_set(x,1,0.3); gsl_vector_set(x,2,-3.0);
    auto* s = gsl_multiroot_fsolver_alloc(gsl_multiroot_fsolver_hybrids, 3);
    gsl_multiroot_fsolver_set(s, &FN, x);
    int status, iter = 0;
    do {
        status = gsl_multiroot_fsolver_iterate(s); ++iter;
        if (status) break;
        status = gsl_multiroot_test_residual(s->f, 1e-12);
    } while (status == GSL_CONTINUE && iter < 500);
    double A1 = gsl_vector_get(s->x,0), A2 = gsl_vector_get(s->x,1);
    double mu = mu_of(&P, A1, A2, gsl_vector_get(s->x,2));
    double gap = 1e9;
    for (size_t i = 0; i < P.s2.size(); ++i)
        gap = std::min(gap, std::sqrt(mu*mu - (J1*J1*A1*A1*P.s2[i] + 4*J2*J2*A2*A2*P.c2[i])));
    printf("J1=%g J2=%g L=%d iters=%d status=%s  A1=%.6f A2=%.6f mu=%.6f gap=%.5f\n",
           J1, J2, L, iter, gsl_strerror(status), fabs(A1), fabs(A2), mu, gap);
    gsl_multiroot_fsolver_free(s); gsl_vector_free(x);
}