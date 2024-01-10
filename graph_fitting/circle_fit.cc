// This fits circles to a collection of points, where the error is related to
// the distance of a point from the circle. This uses auto-differentiation to
// take the derivatives.
//
// The input format is simple text. Feed on standard in:
//
//   x_initial y_initial r_initial
//   x1 y1
//   x2 y2
//   y3 y3
//   ...
//
// And the result after solving will be printed to stdout:
//
//   x y r
//
// There are closed form solutions [1] to this problem which you may want to
// consider instead of using this one. If you already have a decent guess, Ceres
// can squeeze down the last bit of error.
//
//   [1] http://www.mathworks.com/matlabcentral/fileexchange/5557-circle-fit/content/circfit.m  // NOLINT

#include <cstdio>
#include <vector>

#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_double(robust_threshold,
              0.0,
              "Robust loss parameter. Set to 0 for normal squared error (no "
              "robustification).");

// The cost for a single sample. The returned residual is related to the
// distance of the point from the circle (passed in as x, y, m parameters).
//
// Note that the radius is parameterized as r = m^2 to constrain the radius to
// positive values.
class DistanceFromCircleCost {
 public:
  DistanceFromCircleCost(double xx, double yy) : xx_(xx), yy_(yy) {}
  template <typename T>
  bool operator()(const T* const x,
                  const T* const y,
                  const T* const m,  // r = m^2
                  T* residual) const {
    // Since the radius is parameterized as m^2, unpack m to get r.
    T r = *m * *m;

    // Get the position of the sample in the circle's coordinate system.
    T xp = xx_ - *x;
    T yp = yy_ - *y;

    // It is tempting to use the following cost:
    //
    //   residual[0] = r - sqrt(xp*xp + yp*yp);
    //
    // which is the distance of the sample from the circle. This works
    // reasonably well, but the sqrt() adds strong nonlinearities to the cost
    // function. Instead, a different cost is used, which while not strictly a
    // distance in the metric sense (it has units distance^2) it produces more
    // robust fits when there are outliers. This is because the cost surface is
    // more convex.
    residual[0] = r * r - xp * xp - yp * yp;
    return true;
  }

 private:
  // The measured x,y coordinate that should be on the circle.
  double xx_, yy_;
};

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  double x, y, r;
  if (scanf("%lg %lg %lg", &x, &y, &r) != 3) {
    fprintf(stderr, "Couldn't read first line.\n");
    return 1;
  }
  fprintf(stderr, "Got x, y, r %lg, %lg, %lg\n", x, y, r);

  // Save initial values for comparison.
  double initial_x = x;
  double initial_y = y;
  double initial_r = r;

  // Parameterize r as m^2 so that it can't be negative.
  double m = sqrt(r);

  ceres::Problem problem;

  // Configure the loss function.
  ceres::LossFunction* loss = nullptr;
  if (CERES_GET_FLAG(FLAGS_robust_threshold)) {
    loss = new ceres::CauchyLoss(CERES_GET_FLAG(FLAGS_robust_threshold));
  }

  // Add the residuals.
  double xx, yy;
  int num_points = 0;
  while (scanf("%lf %lf\n", &xx, &yy) == 2) {
    ceres::CostFunction* cost =
        new ceres::AutoDiffCostFunction<DistanceFromCircleCost, 1, 1, 1, 1>(
            new DistanceFromCircleCost(xx, yy));
    problem.AddResidualBlock(cost, loss, &x, &y, &m);
    num_points++;
  }

  std::cout << "Got " << num_points << " points.\n";

  // Build and solve the problem.
  ceres::Solver::Options options;
  options.max_num_iterations = 500;
  options.linear_solver_type = ceres::DENSE_QR;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Recover r from m.
  r = m * m;

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x << " -> " << x << "\n";
  std::cout << "y : " << initial_y << " -> " << y << "\n";
  std::cout << "r : " << initial_r << " -> " << r << "\n";
  return 0;
}
