#include "ceres/ceres.h"
#include "glog/logging.h"

// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.


//构建代价函数,cost function
struct CostFunctor {
  //用模板来兼容各种数据类型
  template <typename T>
  //输入用const修饰,函数定义为const防止修改内部其他变量
  //使用仿函数重载(),目的是为了保存函数调用过程中的一些状态(做拟合会用到)
  bool operator()(const T* const x, T* residual) const {
    //输入的x可以是一个向量,因此x是一个数组的地址,用下标可以取出向量对应的值
    //残差计算步骤
    residual[0] = 10.0 - x[0];
    return true;
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double x = 0.5;
  const double initial_x = x;

  // Build the problem. 
  // 构建优化问题
  ceres::Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  // 实例化代价函数,代价函数的选择 有自动求导 和 数值求导
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
  // 运用模板,传入已经实现的代价函数,以及对应的输入与输出的维度,即x与y的维度
  problem.AddResidualBlock(cost_function, nullptr, &x);
  //                       定义的残差模块, 使用的损失函数,传入的因变量

  // Run the solver!
  // 设置求解器,其中有很多可选值,来控制优化的过程,具体见后续文档
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  //输出中间结果
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x << " -> " << x << "\n";
  return 0;
}
