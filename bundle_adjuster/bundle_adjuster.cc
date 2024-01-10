// NOTE: This example will not compile without gflags and SuiteSparse.
//
// The problem being solved here is known as a Bundle Adjustment
// problem in computer vision. Given a set of 3d points X_1, ..., X_n,
// a set of cameras P_1, ..., P_m. If the point X_i is visible in
// image j, then there is a 2D observation u_ij that is the expected
// projection of X_i using P_j. The aim of this optimization is to
// find values of X_i and P_j such that the reprojection error
//
//    E(X,P) =  sum_ij  |u_ij - P_j X_i|^2
//
// is minimized.
//
// The problem used here comes from a collection of bundle adjustment
// problems published at University of Washington.
// http://grail.cs.washington.edu/projects/bal

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "bal_problem.h"
#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "snavely_reprojection_error.h"

// clang-format makes the gflags definitions too verbose
// clang-format off

DEFINE_string(input, "", "Input File name");
DEFINE_string(trust_region_strategy, "levenberg_marquardt",
              "Options are: levenberg_marquardt, dogleg.");
DEFINE_string(dogleg, "traditional_dogleg", "Options are: traditional_dogleg,"
              "subspace_dogleg.");

DEFINE_bool(inner_iterations, false, "Use inner iterations to non-linearly "
            "refine each successful trust region step.");

DEFINE_string(blocks_for_inner_iterations, "automatic", "Options are: "
              "automatic, cameras, points, cameras,points, points,cameras");

DEFINE_string(linear_solver, "sparse_schur", "Options are: "
              "sparse_schur, dense_schur, iterative_schur, "
              "sparse_normal_cholesky, dense_qr, dense_normal_cholesky, "
              "and cgnr.");
DEFINE_bool(explicit_schur_complement, false, "If using ITERATIVE_SCHUR "
            "then explicitly compute the Schur complement.");
DEFINE_string(preconditioner, "jacobi", "Options are: "
              "identity, jacobi, schur_jacobi, schur_power_series_expansion, cluster_jacobi, "
              "cluster_tridiagonal.");
DEFINE_string(visibility_clustering, "canonical_views",
              "single_linkage, canonical_views");
DEFINE_bool(use_spse_initialization, false,
            "Use power series expansion to initialize the solution in ITERATIVE_SCHUR linear solver.");

DEFINE_string(sparse_linear_algebra_library, "suite_sparse",
              "Options are: suite_sparse, accelerate_sparse, eigen_sparse, and "
              "cuda_sparse.");
DEFINE_string(dense_linear_algebra_library, "eigen",
              "Options are: eigen, lapack, and cuda");
DEFINE_string(ordering_type, "amd", "Options are: amd, nesdis");
DEFINE_string(linear_solver_ordering, "user",
              "Options are: automatic and user");

DEFINE_bool(use_quaternions, false, "If true, uses quaternions to represent "
            "rotations. If false, angle axis is used.");
DEFINE_bool(use_manifolds, false, "For quaternions, use a manifold.");
DEFINE_bool(robustify, false, "Use a robust loss function.");

DEFINE_double(eta, 1e-2, "Default value for eta. Eta determines the "
              "accuracy of each linear solve of the truncated newton step. "
              "Changing this parameter can affect solve performance.");

DEFINE_int32(num_threads, -1, "Number of threads. -1 = std::thread::hardware_concurrency.");
DEFINE_int32(num_iterations, 5, "Number of iterations.");
DEFINE_int32(max_linear_solver_iterations, 500, "Maximum number of iterations"
            " for solution of linear system.");
DEFINE_double(spse_tolerance, 0.1,
             "Tolerance to reach during the iterations of power series expansion initialization or preconditioning.");
DEFINE_int32(max_num_spse_iterations, 5,
             "Maximum number of iterations for power series expansion initialization or preconditioning.");
DEFINE_double(max_solver_time, 1e32, "Maximum solve time in seconds.");
DEFINE_bool(nonmonotonic_steps, false, "Trust region algorithm can use"
            " nonmonotic steps.");

DEFINE_double(rotation_sigma, 0.0, "Standard deviation of camera rotation "
              "perturbation.");
DEFINE_double(translation_sigma, 0.0, "Standard deviation of the camera "
              "translation perturbation.");
DEFINE_double(point_sigma, 0.0, "Standard deviation of the point "
              "perturbation.");
DEFINE_int32(random_seed, 38401, "Random seed used to set the state "
             "of the pseudo random number generator used to generate "
             "the perturbations.");
DEFINE_bool(line_search, false, "Use a line search instead of trust region "
            "algorithm.");
DEFINE_bool(mixed_precision_solves, false, "Use mixed precision solves.");
DEFINE_int32(max_num_refinement_iterations, 0, "Iterative refinement iterations");
DEFINE_string(initial_ply, "/home/SENSETIME/yangfan5/code/1-2/learn-ceres/bundle_adjuster/data", "Export the BAL file data as a PLY file.");
DEFINE_string(final_ply, "/home/SENSETIME/yangfan5/code/1-2/learn-ceres/bundle_adjuster/output", "Export the refined BAL file data as a PLY "
              "file.");
// clang-format on

namespace ceres::examples {
namespace {
//设置线性优化器
void SetLinearSolver(Solver::Options* options) {
  //设置线性优化器
  CHECK(StringToLinearSolverType(CERES_GET_FLAG(FLAGS_linear_solver),
                                 &options->linear_solver_type));
                                 //线性优化器类型
  CHECK(StringToPreconditionerType(CERES_GET_FLAG(FLAGS_preconditioner),
                                   &options->preconditioner_type));
                                 //预设条件子
   CHECK(StringToVisibilityClusteringType( 
       CERES_GET_FLAG(FLAGS_visibility_clustering),  // 输入字符串
       &options->visibility_clustering_type  // 输出可见性聚类类型
   ));
   CHECK(StringToSparseLinearAlgebraLibraryType( 
       CERES_GET_FLAG(FLAGS_sparse_linear_algebra_library),  // 输入字符串
       &options->sparse_linear_algebra_library_type  // 输出稀疏线性代数库类型
   ));
  CHECK(StringToDenseLinearAlgebraLibraryType(
      CERES_GET_FLAG(FLAGS_dense_linear_algebra_library),  // 输入字符串
      &options->dense_linear_algebra_library_type)); // 稠密线性代数库类型
  CHECK(
      StringToLinearSolverOrderingType(CERES_GET_FLAG(FLAGS_ordering_type),  // 输入字符串
                                       &options->linear_solver_ordering_type)); // 线性求解器排序类型
  options->use_explicit_schur_complement =
      CERES_GET_FLAG(FLAGS_explicit_schur_complement);
  options->use_mixed_precision_solves =
      CERES_GET_FLAG(FLAGS_mixed_precision_solves);
  options->max_num_refinement_iterations =
      CERES_GET_FLAG(FLAGS_max_num_refinement_iterations);
  options->max_linear_solver_iterations =
      CERES_GET_FLAG(FLAGS_max_linear_solver_iterations);
  options->use_spse_initialization =
      CERES_GET_FLAG(FLAGS_use_spse_initialization);
  options->spse_tolerance = CERES_GET_FLAG(FLAGS_spse_tolerance);
  options->max_num_spse_iterations =
      CERES_GET_FLAG(FLAGS_max_num_spse_iterations);
}

void SetOrdering(BALProblem* bal_problem, Solver::Options* options) {
  const int num_points = bal_problem->num_points();
  const int point_block_size = bal_problem->point_block_size();
  double* points = bal_problem->mutable_points();

  const int num_cameras = bal_problem->num_cameras();
  const int camera_block_size = bal_problem->camera_block_size();
  double* cameras = bal_problem->mutable_cameras();

  
  if (options->use_inner_iterations) {
    if (CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations) == "cameras") {
      LOG(INFO) << "Camera blocks for inner iterations";
      options->inner_iteration_ordering =
          std::make_shared<ParameterBlockOrdering>();
      for (int i = 0; i < num_cameras; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            cameras + camera_block_size * i, 0);
      }
    } else if (CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations) == "points") {
      LOG(INFO) << "Point blocks for inner iterations";
      options->inner_iteration_ordering =
          std::make_shared<ParameterBlockOrdering>();
      for (int i = 0; i < num_points; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            points + point_block_size * i, 0);
      }
    } else if (CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations) ==
               "cameras,points") {
      LOG(INFO) << "Camera followed by point blocks for inner iterations";
      options->inner_iteration_ordering =
          std::make_shared<ParameterBlockOrdering>();
      for (int i = 0; i < num_cameras; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            cameras + camera_block_size * i, 0);
      }
      for (int i = 0; i < num_points; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            points + point_block_size * i, 1);
      }
    } else if (CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations) ==
               "points,cameras") {
      LOG(INFO) << "Point followed by camera blocks for inner iterations";
      options->inner_iteration_ordering =
          std::make_shared<ParameterBlockOrdering>();
      for (int i = 0; i < num_cameras; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            cameras + camera_block_size * i, 1);
      }
      for (int i = 0; i < num_points; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            points + point_block_size * i, 0);
      }
    } else if (CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations) ==
               "automatic") {
      LOG(INFO) << "Choosing automatic blocks for inner iterations";
    } else {
      LOG(FATAL) << "Unknown block type for inner iterations: "
                 << CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations);
    }
  }

  // Bundle adjustment problems have a sparsity structure that makes
  // them amenable to more specialized and much more efficient
  // solution strategies. The SPARSE_SCHUR, DENSE_SCHUR and
  // ITERATIVE_SCHUR solvers make use of this specialized
  // structure.
  //
  // This can either be done by specifying a
  // Options::linear_solver_ordering or having Ceres figure it out
  // automatically using a greedy maximum independent set algorithm.

  // 束缚调整问题具有稀疏结构，这使得它们适合使用更专门和更高效的解法策略。SPARSE_SCHUR、DENSE_SCHUR和ITERATIVE_SCHUR求解器利用了这种专门的结构。
  // 这可以通过指定Options::linear_solver_ordering来完成，也可以让Ceres使用贪婪的最大独立集算法自动确定。

  if (CERES_GET_FLAG(FLAGS_linear_solver_ordering) == "user") {
    auto* ordering = new ceres::ParameterBlockOrdering;

    // The points come before the cameras.
    for (int i = 0; i < num_points; ++i) {
      ordering->AddElementToGroup(points + point_block_size * i, 0);
    }

    for (int i = 0; i < num_cameras; ++i) {
      // When using axis-angle, there is a single parameter block for
      // the entire camera.
      ordering->AddElementToGroup(cameras + camera_block_size * i, 1);
    }

    options->linear_solver_ordering.reset(ordering);
  }
}

//设置最小化终止条件
void SetMinimizerOptions(Solver::Options* options) {
  options->max_num_iterations = CERES_GET_FLAG(FLAGS_num_iterations);
  options->minimizer_progress_to_stdout = true;
  if (CERES_GET_FLAG(FLAGS_num_threads) == -1) {
    const int num_available_threads =
        static_cast<int>(std::thread::hardware_concurrency());
    if (num_available_threads > 0) {
      options->num_threads = num_available_threads;
    }
  } else {
    options->num_threads = CERES_GET_FLAG(FLAGS_num_threads);
  }
  CHECK_GE(options->num_threads, 1);

  options->eta = CERES_GET_FLAG(FLAGS_eta);
  options->max_solver_time_in_seconds = CERES_GET_FLAG(FLAGS_max_solver_time);
  options->use_nonmonotonic_steps = CERES_GET_FLAG(FLAGS_nonmonotonic_steps);
  if (CERES_GET_FLAG(FLAGS_line_search)) {
    options->minimizer_type = ceres::LINE_SEARCH;
  }

  CHECK(StringToTrustRegionStrategyType(
      CERES_GET_FLAG(FLAGS_trust_region_strategy),
      &options->trust_region_strategy_type));
  CHECK(
      StringToDoglegType(CERES_GET_FLAG(FLAGS_dogleg), &options->dogleg_type));
  options->use_inner_iterations = CERES_GET_FLAG(FLAGS_inner_iterations);
}

//配置优化选项
void SetSolverOptionsFromFlags(BALProblem* bal_problem,
                               Solver::Options* options) {
  SetMinimizerOptions(options);
  SetLinearSolver(options);
  SetOrdering(bal_problem, options);
}

//构建问题,将残差依次添加loss函数中
void BuildProblem(BALProblem* bal_problem, Problem* problem) {
  const int point_block_size = bal_problem->point_block_size();
  const int camera_block_size = bal_problem->camera_block_size();
  double* points = bal_problem->mutable_points();
  double* cameras = bal_problem->mutable_cameras();

  // Observations is 2*num_observations long array observations =
  // [u_1, u_2, ... , u_n], where each u_i is two dimensional, the x
  // and y positions of the observation.

  // Observations是一个2*num_observations长度的数组observations，
  // observations = [u_1, u_2, ... , u_n]，
  // 其中每个u_i是二维的，表示观测的位置的x和y坐标。

  const double* observations = bal_problem->observations();
  for (int i = 0; i < bal_problem->num_observations(); ++i) {
    CostFunction* cost_function;
    // Each Residual block takes a point and a camera as input and
    // outputs a 2 dimensional residual.
    cost_function = (CERES_GET_FLAG(FLAGS_use_quaternions))
                        ? SnavelyReprojectionErrorWithQuaternions::Create(
                              observations[2 * i + 0], observations[2 * i + 1])
                        : SnavelyReprojectionError::Create(
                              observations[2 * i + 0], observations[2 * i + 1]);

    // If enabled use Huber's loss function.
    LossFunction* loss_function =
        CERES_GET_FLAG(FLAGS_robustify) ? new HuberLoss(1.0) : nullptr;

    // Each observation corresponds to a pair of a camera and a point
    // which are identified by camera_index()[i] and point_index()[i]
    // respectively.
    double* camera =
        cameras + camera_block_size * bal_problem->camera_index()[i];
    double* point = points + point_block_size * bal_problem->point_index()[i];
    problem->AddResidualBlock(cost_function, loss_function, camera, point);
  }

  if (CERES_GET_FLAG(FLAGS_use_quaternions) &&
      CERES_GET_FLAG(FLAGS_use_manifolds)) {
    Manifold* camera_manifold =
        new ProductManifold<QuaternionManifold, EuclideanManifold<6>>{};
    for (int i = 0; i < bal_problem->num_cameras(); ++i) {
      problem->SetManifold(cameras + camera_block_size * i, camera_manifold);
    }
  }
}

void SolveProblem(const char* filename) {
  //读取命令行参数和输入的文件
  BALProblem bal_problem(filename, CERES_GET_FLAG(FLAGS_use_quaternions));
  
  //文件为空就返回
  if (!CERES_GET_FLAG(FLAGS_initial_ply).empty()) {
    bal_problem.WriteToPLYFile(CERES_GET_FLAG(FLAGS_initial_ply));
  }

  Problem problem;
  //设置随机参数
  srand(CERES_GET_FLAG(FLAGS_random_seed));
  //正则化数据
  bal_problem.Normalize();
  bal_problem.Perturb(CERES_GET_FLAG(FLAGS_rotation_sigma),
                      CERES_GET_FLAG(FLAGS_translation_sigma),
                      CERES_GET_FLAG(FLAGS_point_sigma));

  BuildProblem(&bal_problem, &problem);
  Solver::Options options;
  SetSolverOptionsFromFlags(&bal_problem, &options);
  options.gradient_tolerance = 1e-16;
  options.function_tolerance = 1e-16;
  options.parameter_tolerance = 1e-16;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  if (!CERES_GET_FLAG(FLAGS_final_ply).empty()) {
    bal_problem.WriteToPLYFile(CERES_GET_FLAG(FLAGS_final_ply));
  }
}

}  // namespace
}  // namespace ceres::examples

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  if (CERES_GET_FLAG(FLAGS_input).empty()) {
    LOG(ERROR) << "Usage: bundle_adjuster --input=bal_problem";
    return 1;
  }

  CHECK(CERES_GET_FLAG(FLAGS_use_quaternions) ||
        !CERES_GET_FLAG(FLAGS_use_manifolds))
      << "--use_manifolds can only be used with --use_quaternions.";
  ceres::examples::SolveProblem(CERES_GET_FLAG(FLAGS_input).c_str());
  return 0;
}
