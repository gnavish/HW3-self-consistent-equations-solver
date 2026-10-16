// pi-flux Z2 spin liquid: Schwinger-boson mean-field theory (B = 0) on the J1-J2 square lattice.
//
// Solves the self-consistency equations for fixed J1, J2 on an L x L lattice and writes the
// spinon dispersion for plotting.
//
// Equations (averages <.> over the reduced Brillouin zone, kx in [-pi,pi), ky in [-pi/2,pi/2)):
//     (J1/4) <s2/w> = 1                 (A1 equation, divided by A1)
//      J2    <c2/w> = 1                 (A2 equation, divided by A2)
//     (mu/2) <1/w>  = 1                 (constraint <n_a + n_b> = 1)
// with  w_k = sqrt(mu^2 - |d_k|^2),  |d_k|^2 = J1^2 A1^2 s2 + 4 J2^2 A2^2 c2,
//       s2 = sin^2 kx + sin^2 ky,     c2 = cos^2 kx cos^2 ky.
//
// Unknowns are (A1, A2, t) with mu = max_k |d_k| * (1 + e^t). This keeps w_k real for every trial
// point, which is essential: the solution sits close to mu = max|d|. Solved with GSL's hybrids
// (Powell hybrid method, finite-difference Jacobian).
//
// Build (Ubuntu):  sudo apt install libgsl-dev
//                  g++ -O2 -std=c++17 solve_dispersion.cpp -o solve_dispersion -lgsl -lgslcblas -lm
// Build (macOS):   brew install gsl
//                  clang++ -O2 -std=c++17 solve_dispersion.cpp -o solve_dispersion $(pkg-config --cflags --libs gsl)
// Run:             ./solve_dispersion J1 J2 [L] [prefix]        defaults: L = 64, prefix = "out"
//
// Output files:
//   <prefix>_params.txt   J1, J2, L, A1, A2, mu, gap, k of gap minimum, checks
//   <prefix>_path.csv     omega_k along Gamma -> X -> M -> Y -> Gamma   (columns: s,kx,ky,omega,label)
//   <prefix>_map.csv      omega_k on a 201 x 101 grid over the reduced zone (columns: kx,ky,omega)
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
    std::vector<double> s2, c2;   // sin^2 kx + sin^2 ky and cos^2 kx cos^2 ky on the k-grid
};

static double d2_of(double J1, double J2, double A1, double A2, double kx, double ky) {
    double s2 = std::sin(kx) * std::sin(kx) + std::sin(ky) * std::sin(ky);
    double c2 = std::pow(std::cos(kx) * std::cos(ky), 2);
    return J1 * J1 * A1 * A1 * s2 + 4 * J2 * J2 * A2 * A2 * c2;
}

static double d2_grid(const Params* P, size_t i, double A1, double A2) {
    return P->J1 * P->J1 * A1 * A1 * P->s2[i] + 4 * P->J2 * P->J2 * A2 * A2 * P->c2[i];
}

static double mu_of(const Params* P, double A1, double A2, double t) {
    double M = 0;
    for (size_t i = 0; i < P->s2.size(); ++i) M = std::max(M, d2_grid(P, i, A1, A2));
    return std::sqrt(M) * (1 + std::exp(t));
}

static int residuals(const gsl_vector* x, void* p, gsl_vector* F) {
    auto* P = static_cast<Params*>(p);
    double A1 = gsl_vector_get(x, 0), A2 = gsl_vector_get(x, 1), t = gsl_vector_get(x, 2);
    double mu = mu_of(P, A1, A2, t);
    double S0 = 0, S1 = 0, S2 = 0;
    size_t n = P->s2.size();
    for (size_t i = 0; i < n; ++i) {
        double w = std::sqrt(mu * mu - d2_grid(P, i, A1, A2));
        S0 += P->s2[i] / w; S1 += P->c2[i] / w; S2 += 1 / w;
    }
    gsl_vector_set(F, 0, P->J1 / 4 * S0 / n - 1);
    gsl_vector_set(F, 1, P->J2 * S1 / n - 1);
    gsl_vector_set(F, 2, mu / 2 * S2 / n - 1);
    return GSL_SUCCESS;
}

struct Result { double A1, A2, t, mu; int status, iters; double res[3]; };

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
        status = gsl_multiroot_test_residual(s->f, 1e-11);
    } while (status == GSL_CONTINUE && iter < 1000);
    // Near a gap closing the solver can stall at round-off level; accept that as converged.
    if (status != GSL_SUCCESS && gsl_multiroot_test_residual(s->f, 1e-9) == GSL_SUCCESS) status = GSL_SUCCESS;
    Result r;
    r.A1 = std::fabs(gsl_vector_get(s->x, 0));   // signs of A1, A2 are a gauge choice
    r.A2 = std::fabs(gsl_vector_get(s->x, 1));
    r.t = gsl_vector_get(s->x, 2);
    r.mu = mu_of(&P, r.A1, r.A2, r.t);
    for (int k = 0; k < 3; ++k) r.res[k] = gsl_vector_get(s->f, k);
    r.status = status; r.iters = iter;
    gsl_multiroot_fsolver_free(s);
    gsl_vector_free(x);
    return r;
}

// k-grid of the reduced Brillouin zone (two-site unit cell): L x L/2 points
static void build_grid(Params& P, int L) {
    P.s2.clear(); P.c2.clear();
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L / 2; ++j) {
            double kx = -M_PI + 2 * M_PI * i / L, ky = -M_PI / 2 + 2 * M_PI * j / L;
            P.s2.push_back(std::sin(kx) * std::sin(kx) + std::sin(ky) * std::sin(ky));
            P.c2.push_back(std::pow(std::cos(kx) * std::cos(ky), 2));
        }
}

// Try a preferred start (if given) and then several fixed starting points; true on convergence.
static bool solve_any(Params& P, Result& r, double A1 = -1, double A2 = -1, double t = 0) {
    if (A1 > 0) { r = solve(P, A1, A2, t); if (r.status == GSL_SUCCESS) return true; }
    const double guesses[][3] = {{0.3, 0.3, -3}, {0.4, 0.2, -5}, {0.2, 0.4, -5}, {0.5, 0.1, -2},
                                 {0.1, 0.5, -2}, {0.3, 0.1, -6}, {0.1, 0.3, -6}};
    for (auto& g : guesses) {
        r = solve(P, g[0], g[1], g[2]);
        if (r.status == GSL_SUCCESS) return true;
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s J1 J2 [L=64] [prefix=out]\n", argv[0]);
        return 1;
    }
    double J1 = std::atof(argv[1]), J2 = std::atof(argv[2]);
    int L = argc > 3 ? std::atoi(argv[3]) : 64;
    const char* prefix = argc > 4 ? argv[4] : "out";
    if (J1 <= 0 || J2 <= 0 || L < 4 || L % 2) {
        std::fprintf(stderr, "need J1 > 0, J2 > 0 and even L >= 4\n");
        return 1;
    }
    gsl_set_error_handler_off();   // handle solver failures ourselves

    // Main solve on L x L, plus a cheaper solve on (L/2) x (L/2) to check finite-size effects.
    Params P{J1, J2, {}, {}};
    build_grid(P, L);
    Result r{};
    if (!solve_any(P, r)) {
        std::fprintf(stderr, "solver did not converge for J1 = %g, J2 = %g, L = %d (%s).\n"
                     "Residuals: %.2e %.2e %.2e. A gapped pi-flux solution with A1, A2 != 0 may not exist here.\n",
                     J1, J2, L, gsl_strerror(r.status), r.res[0], r.res[1], r.res[2]);
        return 2;
    }
    double gap_half = -1;
    if ((L / 2) % 2 == 0 && L / 2 >= 4) {
        Params Ph{J1, J2, {}, {}};
        build_grid(Ph, L / 2);
        Result rh{};
        if (solve_any(Ph, rh, r.A1, r.A2, r.t)) {
            gap_half = 1e300;
            for (size_t i = 0; i < Ph.s2.size(); ++i)
                gap_half = std::min(gap_half, std::sqrt(rh.mu * rh.mu - d2_grid(&Ph, i, rh.A1, rh.A2)));
        }
    }

    // gap on the solver grid, and on a fine grid for the plot
    double gap_grid = 1e300;
    for (size_t i = 0; i < P.s2.size(); ++i)
        gap_grid = std::min(gap_grid, std::sqrt(r.mu * r.mu - d2_grid(&P, i, r.A1, r.A2)));

    // constraint check via Bogoliubov coefficients: (4/N) sum_k |v_k|^2 = 1, |v_k|^2 = (mu/w - 1)/2
    double v2 = 0;
    for (size_t i = 0; i < P.s2.size(); ++i) {
        double w = std::sqrt(r.mu * r.mu - d2_grid(&P, i, r.A1, r.A2));
        v2 += 0.5 * (r.mu / w - 1);
    }
    double nsite = 4.0 * v2 / (double)(L * L);

    // ---- fine map over the reduced zone ----
    const int NX = 201, NY = 101;
    char fname[512];
    std::snprintf(fname, sizeof fname, "%s_map.csv", prefix);
    FILE* fm = std::fopen(fname, "w");
    if (!fm) { std::perror(fname); return 1; }
    std::fprintf(fm, "kx,ky,omega\n");
    double gap = 1e300, kgx = 0, kgy = 0, wmax = 0;
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j) {
            double kx = -M_PI + 2 * M_PI * i / (NX - 1), ky = -M_PI / 2 + M_PI * j / (NY - 1);
            double w = std::sqrt(r.mu * r.mu - d2_of(J1, J2, r.A1, r.A2, kx, ky));
            std::fprintf(fm, "%.8f,%.8f,%.10f\n", kx, ky, w);
            if (w < gap) { gap = w; kgx = kx; kgy = ky; }
            wmax = std::max(wmax, w);
        }
    std::fclose(fm);

    // ---- high-symmetry path Gamma(0,0) -> X(pi,0) -> M(pi,pi/2) -> Y(0,pi/2) -> Gamma ----
    std::snprintf(fname, sizeof fname, "%s_path.csv", prefix);
    FILE* fp = std::fopen(fname, "w");
    if (!fp) { std::perror(fname); return 1; }
    std::fprintf(fp, "s,kx,ky,omega,label\n");
    const double cx[5] = {0, M_PI, M_PI, 0, 0}, cy[5] = {0, 0, M_PI / 2, M_PI / 2, 0};
    const char* lab[5] = {"G", "X", "M", "Y", "G"};
    const int NSEG = 400;
    double s0 = 0;
    for (int seg = 0; seg < 4; ++seg) {
        double len = std::hypot(cx[seg + 1] - cx[seg], cy[seg + 1] - cy[seg]);
        for (int m = 0; m < NSEG; ++m) {
            double u = (double)m / NSEG;
            double kx = cx[seg] + u * (cx[seg + 1] - cx[seg]), ky = cy[seg] + u * (cy[seg + 1] - cy[seg]);
            double w = std::sqrt(r.mu * r.mu - d2_of(J1, J2, r.A1, r.A2, kx, ky));
            std::fprintf(fp, "%.8f,%.8f,%.8f,%.10f,%s\n", s0 + u * len, kx, ky, w, m == 0 ? lab[seg] : "");
        }
        s0 += len;
    }
    std::fprintf(fp, "%.8f,%.8f,%.8f,%.10f,%s\n", s0, 0.0, 0.0,
                 std::sqrt(r.mu * r.mu - d2_of(J1, J2, r.A1, r.A2, 0, 0)), lab[4]);
    std::fclose(fp);

    // ---- parameters ----
    std::snprintf(fname, sizeof fname, "%s_params.txt", prefix);
    FILE* fq = std::fopen(fname, "w");
    if (!fq) { std::perror(fname); return 1; }
    // finite-size verdict from ratio = gap(L/2) / gap(L):
    //   ~1      gap converged -> genuinely gapped Z2 spin liquid
    //   1-1.5   gap still converging -> increase L
    //   >= 1.5  gap keeps dropping (roughly ~1/L^2) -> gap closes as L -> inf: spinon condensation
    double rel = gap_half > 0 ? std::fabs(gap_half - gap_grid) / gap_grid : -1;
    double ratio = gap_half > 0 ? gap_half / gap_grid : -1;
    const char* verdict = ratio < 0    ? "unknown (L/2 check not available)"
                        : rel < 0.01   ? "converged: gapped Z2 spin liquid"
                        : ratio < 1.5  ? "still converging: rerun with larger L"
                        : "gap shrinks rapidly with L: likely gapless (spinon condensation) as L -> inf";

    std::fprintf(fq, "J1=%.10g\nJ2=%.10g\nL=%d\nA1=%.12f\nA2=%.12f\nmu=%.12f\ngap=%.12f\n"
                     "gap_kx_over_pi=%.6f\ngap_ky_over_pi=%.6f\nbandwidth=%.12f\n"
                     "gap_on_solver_grid=%.12f\ngap_on_half_grid=%.12f\nn_per_site=%.12f\niterations=%d\n"
                     "finite_size=%s\n",
                 J1, J2, L, r.A1, r.A2, r.mu, gap, kgx / M_PI, kgy / M_PI, wmax - gap, gap_grid, gap_half,
                 nsite, r.iters, verdict);
    std::fclose(fq);

    std::printf("J1 = %g, J2 = %g, L = %d   (converged in %d iterations)\n", J1, J2, L, r.iters);
    std::printf("  A1  = %.10f\n  A2  = %.10f\n  mu  = %.10f\n", r.A1, r.A2, r.mu);
    std::printf("  gap = %.10f at k = (%+.3f, %+.3f) pi\n", gap, kgx / M_PI, kgy / M_PI);
    std::printf("  check: <n_a + n_b> per site = %.12f (should be 1)\n", nsite);
    if (rel >= 0)
        std::printf("  finite-size check: gap(L) = %.6f, gap(L/2) = %.6f, change %.2f%% -> %s\n",
                    gap_grid, gap_half, 100 * rel, verdict);
    else
        std::printf("  finite-size check: %s\n", verdict);
    std::printf("wrote %s_params.txt, %s_path.csv, %s_map.csv\n", prefix, prefix, prefix);
    return 0;
}