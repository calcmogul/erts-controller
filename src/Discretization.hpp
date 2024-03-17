// Copyright (c) Tyler Veness

#pragma once

#include <chrono>

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

/**
 * Discretizes the given continuous A matrix.
 *
 * @tparam States Number of states.
 * @param contA Continuous system matrix.
 * @param dt Discretization timestep.
 * @param discA Storage for discrete system matrix.
 */
template <int States>
void DiscretizeA(const Eigen::Matrix<double, States, States>& contA,
                 std::chrono::duration<double> dt,
                 Eigen::Matrix<double, States, States>* discA) {
  // A_d = eᴬᵀ
  *discA = (contA * dt.count()).exp();
}

/**
 * Discretizes the given continuous A and B matrices.
 *
 * @tparam States Number of states.
 * @tparam Inputs Number of inputs.
 * @param contA Continuous system matrix.
 * @param contB Continuous input matrix.
 * @param dt Discretization timestep.
 * @param discA Storage for discrete system matrix.
 * @param discB Storage for discrete input matrix.
 */
template <int States, int Inputs>
void DiscretizeAB(const Eigen::Matrix<double, States, States>& contA,
                  const Eigen::Matrix<double, States, Inputs>& contB,
                  std::chrono::duration<double> dt,
                  Eigen::Matrix<double, States, States>* discA,
                  Eigen::Matrix<double, States, Inputs>* discB) {
  // M = [A  B]
  //     [0  0]
  Eigen::Matrix<double, States + Inputs, States + Inputs> M;
  M.template block<States, States>(0, 0) = contA;
  M.template block<States, Inputs>(0, States) = contB;
  M.template block<Inputs, States + Inputs>(States, 0).setZero();

  // ϕ = eᴹᵀ = [A_d  B_d]
  //           [ 0    I ]
  Eigen::Matrix<double, States + Inputs, States + Inputs> phi =
      (M * dt.count()).exp();

  *discA = phi.template block<States, States>(0, 0);
  *discB = phi.template block<States, Inputs>(0, States);
}
