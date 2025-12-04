// Copyright (c) Tyler Veness

#pragma once

#include <chrono>

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

/// Discretizes the given continuous A matrix.
///
/// @tparam States Number of states.
/// @param cont_A Continuous system matrix.
/// @param dt Discretization timestep.
/// @param disc_A Storage for discrete system matrix.
template <int States>
void discretize_a(const Eigen::Matrix<double, States, States>& cont_A,
                  std::chrono::duration<double> dt,
                  Eigen::Matrix<double, States, States>* disc_A) {
  // A_d = eᴬᵀ
  *disc_A = (cont_A * dt.count()).exp();
}

/// Discretizes the given continuous A and B matrices.
///
/// @tparam States Number of states.
/// @tparam Inputs Number of inputs.
/// @param cont_A Continuous system matrix.
/// @param cont_B Continuous input matrix.
/// @param dt Discretization timestep.
/// @param disc_A Storage for discrete system matrix.
/// @param disc_B Storage for discrete input matrix.
template <int States, int Inputs>
void discretize_ab(const Eigen::Matrix<double, States, States>& cont_A,
                   const Eigen::Matrix<double, States, Inputs>& cont_B,
                   std::chrono::duration<double> dt,
                   Eigen::Matrix<double, States, States>* disc_A,
                   Eigen::Matrix<double, States, Inputs>* disc_B) {
  // M = [A  B]
  //     [0  0]
  Eigen::Matrix<double, States + Inputs, States + Inputs> M;
  M.template block<States, States>(0, 0) = cont_A;
  M.template block<States, Inputs>(0, States) = cont_B;
  M.template block<Inputs, States + Inputs>(States, 0).setZero();

  // ϕ = eᴹᵀ = [A_d  B_d]
  //           [ 0    I ]
  Eigen::Matrix<double, States + Inputs, States + Inputs> phi =
      (M * dt.count()).exp();

  *disc_A = phi.template block<States, States>(0, 0);
  *disc_B = phi.template block<States, Inputs>(0, States);
}
