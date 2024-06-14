// Copyright (c) Tyler Veness

// Runs Extended Rauch-Tung-Striebel controller on differential drive.
// https://file.tavsys.net/control/papers/Extended%20Rauch-Tung-Striebel%20Controller%2C%20ZAGSS.pdf

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numbers>
#include <print>
#include <string_view>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>

#include "DCMotor.hpp"
#include "Discretization.hpp"
#include "RK4.hpp"
#include "WriteVectorCSV.hpp"

using namespace std::chrono_literals;
using namespace std::string_view_literals;

constexpr std::chrono::duration<double> DT = 20ms;
constexpr int HORIZON = 100;

/**
 * Linear interpolation between a and b.
 *
 * @param a Left value.
 * @param b Right value.
 * @param t Interpolant [0, 1].
 */
template <typename T>
T lerp(const T& a, const T& b, double t) {
  return a + t * (b - a);
}

/**
 * Generate square path reference.
 */
std::vector<Eigen::Vector<double, 5>> GetSquareRefs() {
  std::vector<Eigen::Vector<double, 5>> refs;

  double heading = 0.0;
  constexpr double v = 2.0;
  std::vector<Eigen::Vector<double, 2>> point_vectors{
      {0, 0}, {2, 0}, {2, 2}, {-2, 2}, {-2, 4}, {4, 4}, {4, -2}, {0, -2}};

  for (size_t i = 0; i < point_vectors.size() - 1; ++i) {
    const auto& pt0 = point_vectors[i];
    const auto& pt1 = point_vectors[i + 1];

    Eigen::Vector<double, 2> diff = pt1 - pt0;
    double t = std::sqrt((diff.transpose() * diff)(0)) / v;
    double new_heading = std::atan2(diff(1), diff(0));

    double heading_diff = new_heading - heading;
    if (heading_diff > std::numbers::pi) {
      if (heading_diff > 0) {
        heading += heading_diff - 2 * std::numbers::pi;
      } else {
        heading += heading_diff + 2 * std::numbers::pi;
      }
    } else {
      heading += heading_diff;
    }

    int num_pts = t / DT.count();
    for (int j = 0; j < num_pts; ++j) {
      Eigen::Vector<double, 2> mid =
          lerp(pt0, pt1, static_cast<double>(j) / num_pts);
      refs.emplace_back(
          Eigen::Vector<double, 5>{{mid(0), mid(1), heading, v, v}});
    }
  }

  return refs;
}

/**
 * Differential drive with ERTS controller.
 */
class DifferentialDrive {
 public:
  // Number of motors per side
  static constexpr int num_motors = 3;

  // Gear ratio
  static constexpr double G = 60.0 / 11.0;

  // Drivetrain mass in kg
  static constexpr double m = 52.0;

  // Radius of wheels in meters
  static constexpr double r = 0.08255 / 2.0;

  // Radius of robot in meters
  static constexpr double rb = 0.59055 / 2.0;

  // Moment of inertia of the differential drive in kg-m²
  static constexpr double J = 6.0;

  static constexpr DCMotor motor = DCMotor::CIM(num_motors);

  static constexpr double C1 =
      -(G * G) * motor.Kt / (motor.Kv * motor.R * r * r);
  static constexpr double C2 = G * motor.Kt / (motor.R * r);
  static constexpr double C3 =
      -(G * G) * motor.Kt / (motor.Kv * motor.R * r * r);
  static constexpr double C4 = G * motor.Kt / (motor.R * r);

  static constexpr Eigen::Matrix<double, 2, 2> velocity_A{{
      {(1 / m + rb * rb / J) * C1, (1 / m - rb * rb / J) * C3},
      {(1 / m - rb * rb / J) * C1, (1 / m + rb * rb / J) * C3},
  }};
  static constexpr Eigen::Matrix<double, 2, 2> velocity_B{{
      {(1 / m + rb * rb / J) * C2, (1 / m - rb * rb / J) * C4},
      {(1 / m - rb * rb / J) * C2, (1 / m + rb * rb / J) * C4},
  }};

  // States: x (m), y (m), heading (rad), left velocity (m/s),
  //         right velocity (m/s)
  // Q = diag(1/q²)
  // Q⁻¹ = diag(q²)
  static constexpr Eigen::Matrix<double, 5, 5> Qinv{{0.125 * 0.125, 0, 0, 0, 0},
                                                    {0, 0.125 * 0.125, 0, 0, 0},
                                                    {0, 0, 10.0 * 10.0, 0, 0},
                                                    {0, 0, 0, 0.95 * 0.95, 0},
                                                    {0, 0, 0, 0, 0.95 * 0.95}};

  // Inputs: Left voltage (V), right voltage (V)
  // R = diag(1/r²)
  // R⁻¹ = diag(r²)
  static constexpr Eigen::Matrix<double, 2, 2> Rinv{{12.0 * 12.0, 0},
                                                    {0, 12.0 * 12.0}};

  std::chrono::duration<double> m_dt = 0s;

  size_t t = 0;

  // Sim variables
  Eigen::Vector<double, 5> x = Eigen::Vector<double, 5>::Zero();
  Eigen::Vector<double, 2> u = Eigen::Vector<double, 2>::Zero();

  std::vector<Eigen::Vector<double, 5>> m_refs;

  // Kalman smoother storage
  std::vector<Eigen::Vector<double, 5>> x_hat_pre;
  std::vector<Eigen::Vector<double, 5>> x_hat_post;
  std::vector<Eigen::Matrix<double, 5, 5>> A;
  std::vector<Eigen::Matrix<double, 5, 5>> P_pre;
  std::vector<Eigen::Matrix<double, 5, 5>> P_post;
  std::vector<Eigen::Vector<double, 5>> x_hat_smooth;

  static constexpr Eigen::Vector<double, 2> u_min{{-12.0}, {-12.0}};
  static constexpr Eigen::Vector<double, 2> u_max{{12.0}, {12.0}};

  /**
   * Drivetrain subsystem.
   *
   * @param dt Time between model/controller updates.
   */
  explicit DifferentialDrive(std::chrono::duration<double> dt) : m_dt{dt} {
    // Get reference trajectory
    m_refs = GetSquareRefs();

    // Kalman smoother storage
    x_hat_pre.reserve(m_refs.size());
    x_hat_post.reserve(m_refs.size());
    A.reserve(m_refs.size());
    P_pre.reserve(m_refs.size());
    P_post.reserve(m_refs.size());
    x_hat_smooth.reserve(m_refs.size());
    for (size_t i = 0; i < m_refs.size(); ++i) {
      x_hat_pre.emplace_back(Eigen::Vector<double, 5>::Zero());
      x_hat_post.emplace_back(Eigen::Vector<double, 5>::Zero());
      A.emplace_back(Eigen::Matrix<double, 5, 5>::Zero());
      P_pre.emplace_back(Eigen::Matrix<double, 5, 5>::Zero());
      P_post.emplace_back(Eigen::Matrix<double, 5, 5>::Zero());
      x_hat_smooth.emplace_back(Eigen::Vector<double, 5>::Zero());
    }
  }

  /**
   * Nonlinear differential drive dynamics.
   *
   * States: [[x], [y], [heading], [left velocity], [right velocity]]
   * Inputs: [[left voltage], [right voltage]]
   *
   * @param x State vector.
   * @param u Input vector.
   */
  Eigen::Vector<double, 5> f(const Eigen::Vector<double, 5>& x,
                             const Eigen::Vector<double, 2>& u) {
    Eigen::Vector<double, 5> dxdt;

    double v = (x(3) + x(4)) / 2.0;
    dxdt(0) = v * std::cos(x(2));
    dxdt(1) = v * std::sin(x(2));
    dxdt(2) = (x(4) - x(3)) / (2.0 * rb);
    dxdt.segment<2>(3) = velocity_A * x.segment<2>(3) + velocity_B * u;

    return dxdt;
  }

  /**
   * Nonlinear differential drive measurement model.
   *
   * Outputs: [[x], [y], [heading]]
   *
   * @param x State vector.
   */
  Eigen::Vector<double, 3> h(const Eigen::Vector<double, 5>& x) const {
    return x.segment<3>(0);
  }

  /**
   * Returns the Jacobian of f with respect to the state.
   *
   * @param x The current state.
   * @param u The current input.
   */
  Eigen::Matrix<double, 5, 5> df_dx(
      const Eigen::Vector<double, 5>& x,
      [[maybe_unused]] const Eigen::Vector<double, 2>& u) {
    double v = (x(3) + x(4)) / 2.0;
    double c = std::cos(x(2));
    double s = std::sin(x(2));

    return Eigen::Matrix<double, 5, 5>{
        {0.0, 0.0, -v * s, 0.5 * c, 0.5 * c},
        {0.0, 0.0, v * c, 0.5 * s, 0.5 * s},
        {0.0, 0.0, 0.0, -0.5 / rb, 0.5 / rb},
        {0.0, 0.0, 0.0, velocity_A(0, 0), velocity_A(0, 1)},
        {0.0, 0.0, 0.0, velocity_A(1, 0), velocity_A(1, 1)}};
  }

  /**
   * Returns the Jacobian of f with respect to the input.
   *
   * @param x The current state.
   * @param u The current input.
   */
  Eigen::Matrix<double, 5, 2> df_du(
      [[maybe_unused]] const Eigen::Vector<double, 5>& x,
      [[maybe_unused]] const Eigen::Vector<double, 2>& u) {
    return Eigen::Matrix<double, 5, 2>{
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {(1 / m + rb * rb / J) * C2, (1 / m - rb * rb / J) * C4},
        {(1 / m - rb * rb / J) * C2, (1 / m + rb * rb / J) * C4}};
  }

  /**
   * Returns the Jacobian of h with respect to the state.
   *
   * @param x The current state.
   * @param u The current input.
   */
  Eigen::Matrix<double, 3, 5> dh_dx(
      [[maybe_unused]] const Eigen::Vector<double, 5>& x,
      [[maybe_unused]] const Eigen::Vector<double, 2>& u) {
    return Eigen::Matrix<double, 3, 5>{
        {1, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}};
  }

  /**
   * Advance the model by one timestep.
   *
   * @param r The current reference.
   * @param nextR The next reference.
   */
  void Update([[maybe_unused]] const Eigen::Vector<double, 5>& r,
              [[maybe_unused]] const Eigen::Vector<double, 5>& nextR) {
    x = RK4([this](const auto& x, const auto& u) { return f(x, u); }, x, u,
            m_dt);

    // Since this is the last reference, there are no
    // reference dynamics to follow
    if (t == m_refs.size() - 1) {
      u = Eigen::Vector<double, 2>::Zero();
      return;
    }

    // Linearize model
    Eigen::Matrix<double, 5, 5> Ac = df_dx(x, Eigen::Vector<double, 2>::Zero());
    Eigen::Matrix<double, 5, 2> Bc = df_du(x, Eigen::Vector<double, 2>::Zero());
    Eigen::Matrix<double, 5, 5> Ad;
    Eigen::Matrix<double, 5, 2> Bd;
    DiscretizeAB<5, 2>(Ac, Bc, m_dt, &Ad, &Bd);

    x_hat_pre[t] = x;
    P_pre[t] = Eigen::Matrix<double, 5, 5>::Zero();
    x_hat_post[t] = x;
    P_post[t] = Eigen::Matrix<double, 5, 5>::Zero();

    // Prediction
    int N = std::min<double>(m_refs.size() - 1, t + HORIZON);
    for (int τ = t + 1; τ < N + 1; ++τ) {
      x_hat_pre[τ] =
          RK4([this](const auto& x, const auto& u) { return f(x, u); },
              x_hat_post[τ - 1], Eigen::Vector<double, 2>::Zero(), m_dt);

      // Linearization
      DiscretizeA<5>(df_dx(x_hat_post[τ - 1], Eigen::Vector<double, 2>::Zero()),
                     m_dt, &A[τ - 1]);

      Eigen::Matrix<double, 3, 5> C =
          dh_dx(x_hat_post[τ - 1], Eigen::Vector<double, 2>::Zero());
      Eigen::Matrix<double, 3, 1> s_τ = C * m_refs[τ];

      P_pre[τ] = A[τ - 1] * P_post[τ - 1] * A[τ - 1].transpose() +
                 Bd * Rinv * Bd.transpose();

      // S = CPCᵀ + CQ⁻¹Cᵀ
      Eigen::Matrix<double, 3, 3> S =
          C * P_pre[τ] * C.transpose() + C * Qinv * C.transpose();

      // We want to put K = PCᵀS⁻¹ into Ax = b form so we can solve it more
      // efficiently.
      //
      // K = PCᵀS⁻¹
      // KS = PCᵀ
      // (KS)ᵀ = (PCᵀ)ᵀ
      // SᵀKᵀ = CPᵀ
      //
      // The solution of Ax = b can be found via x = A.solve(b).
      //
      // Kᵀ = Sᵀ.solve(CPᵀ)
      // K = (Sᵀ.solve(CPᵀ))ᵀ
      Eigen::Matrix<double, 5, 3> K =
          S.transpose().llt().solve(C * P_pre[τ].transpose()).transpose();

      x_hat_post[τ] = x_hat_pre[τ] + K * (s_τ - h(x_hat_pre[τ]));

      constexpr Eigen::Matrix<double, 5, 5> I{{1, 0, 0, 0, 0},
                                              {0, 1, 0, 0, 0},
                                              {0, 0, 1, 0, 0},
                                              {0, 0, 0, 1, 0},
                                              {0, 0, 0, 0, 1}};
      P_post[τ] = (I - K * C) * P_pre[τ] * (I - K * C).transpose() +
                  K * C * Qinv * C.transpose() * K.transpose();
    }

    // Last filtered estimate is already optimal smoothed estimate
    x_hat_smooth[N] = x_hat_post[N];

    // Smoothing
    for (size_t τ = N - 1; τ > (t + 1) - 1; --τ) {
      // L = P⁺[τ] A[τ]ᵀ P⁻[τ + 1]⁻¹
      // L P⁻[τ + 1] = P⁺[τ] A[τ]ᵀ
      // P⁻[τ + 1]ᵀ Lᵀ = A[τ] P⁺[τ]ᵀ
      // Lᵀ = P⁻[τ + 1]ᵀ.solve(A[τ] P⁺[τ]ᵀ)
      // L = P⁻[τ + 1]ᵀ.solve(A[τ] P⁺[τ]ᵀ)ᵀ
      Eigen::Matrix<double, 5, 5> L = P_pre[τ + 1]
                                          .transpose()
                                          .llt()
                                          .solve(A[τ] * P_post[τ].transpose())
                                          .transpose();

      x_hat_smooth[τ] =
          x_hat_post[τ] + L * (x_hat_smooth[τ + 1] - x_hat_pre[τ + 1]);
    }

    // x̂ₖ₊₁ = f(x̂ₖ) + Buₖ
    // Buₖ = x̂ₖ₊₁ − f(x̂ₖ)
    // uₖ = B⁺(x̂ₖ₊₁ − f(x̂ₖ))
    u = Bd.householderQr().solve(
        x_hat_smooth[t + 1] -
        RK4([this](const auto& x, const auto& u) { return f(x, u); },
            x_hat_post[t], Eigen::Vector<double, 2>::Zero(), m_dt));

    double u_cap = u.lpNorm<Eigen::Infinity>();
    if (u_cap > 12.0) {
      u = u / u_cap * 12.0;
    }
    ++t;
  }
};

int main() {
  auto refs = GetSquareRefs();

  std::vector<double> t;
  t.emplace_back(0.0);
  for (size_t i = 0; i < refs.size() - 1; ++i) {
    t.emplace_back(t.back() + DT.count());
  }

  Eigen::Vector<double, 5> x = Eigen::Vector<double, 5>::Zero();
  auto diff_drive = DifferentialDrive(DT);
  diff_drive.x = x;

  auto start = std::chrono::system_clock::now();

  std::vector<Eigen::Vector<double, 5>> r_rec;
  r_rec.reserve(refs.size());
  std::vector<Eigen::Vector<double, 5>> x_rec;
  x_rec.reserve(refs.size());
  std::vector<Eigen::Vector<double, 2>> u_rec;
  u_rec.reserve(refs.size());

  // Run simulation
  for (size_t i = 0; i < refs.size(); ++i) {
    if (i < refs.size() - 1) {
      diff_drive.Update(refs[i], refs[i + 1]);
    } else {
      diff_drive.Update(refs[i], refs[i]);
    }

    // Log states for plotting
    r_rec.emplace_back(refs[i]);
    x_rec.emplace_back(diff_drive.x);
    u_rec.emplace_back(diff_drive.u);
  }

  auto end = std::chrono::system_clock::now();

  std::print(stderr, "Total time = {} ms\n",
             std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                     .count() /
                 1e3);

  WriteVectorCSV("ERTS references.csv"sv, t, r_rec,
                 std::array{"Time (s)"sv, "X position (m)"sv,
                            "Y position (m)"sv, "Heading (rad)"sv,
                            "Left velocity (m/s)"sv, "Right velocity (m/s)"sv});

  WriteVectorCSV("ERTS states.csv"sv, t, x_rec,
                 std::array{"Time (s)"sv, "X position (m)"sv,
                            "Y position (m)"sv, "Heading (rad)"sv,
                            "Left velocity (m/s)"sv, "Right velocity (m/s)"sv});

  WriteVectorCSV(
      "ERTS inputs.csv"sv, t, u_rec,
      std::array{"Time (s)"sv, "Left voltage (V)"sv, "Right voltage (V)"sv});
}
