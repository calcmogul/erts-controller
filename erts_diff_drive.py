#!/usr/bin/env python3

"""
Runs Extended Rauch-Tung-Striebel controller on differential drive.
https://file.tavsys.net/control/papers/Extended%20Rauch-Tung-Striebel%20Controller%2C%20ZAGSS.pdf
"""

import math
import time
import sys

import frccontrol as fct
import matplotlib.pyplot as plt
import numpy as np

DT = 0.02
HORIZON = 100


def lerp(a, b, t):
    """
    Linear interpolation between a and b.

    Keyword arguments:
    a -- left value
    b -- right value
    t -- interpolant [0, 1]
    """
    return a + t * (b - a)


def get_square_refs():
    """
    Generate square path reference.
    """
    refs = []
    heading = 0.0
    points = [
        (0, 0),
        (2, 0),
        (2, 2),
        (-2, 2),
        (-2, 4),
        (4, 4),
        (4, -2),
        (0, -2),
    ]
    v = 2.0
    point_vectors = [np.array([[x], [y]]) for x, y in points]
    for pt0, pt1 in zip(point_vectors, point_vectors[1:]):
        diff = pt1 - pt0
        t = math.sqrt((diff.T @ diff)[0, 0]) / v
        new_heading = math.atan2(diff[1, 0], diff[0, 0])

        heading_diff = new_heading - heading
        if heading_diff > math.pi:
            if heading_diff > 0:
                heading += heading_diff - 2 * math.pi
            else:
                heading += heading_diff + 2 * math.pi
        else:
            heading += heading_diff

        num_pts = int(t / DT)
        for j in range(num_pts):
            mid = lerp(pt0, pt1, j / num_pts)
            ref = np.array(
                [
                    [mid[0, 0]],
                    [mid[1, 0]],
                    [heading],
                    [v],
                    [v],
                ]
            )
            refs.append(ref)
    return refs


class DifferentialDrive:
    """
    Differential drive with ERTS controller.
    """

    def __init__(self, dt):
        """Drivetrain subsystem.

        Keyword arguments:
        dt -- time between model/controller updates
        """
        self.dt = dt

        # Number of motors per side
        num_motors = 3
        # Gear ratio
        G = 60.0 / 11.0
        # Drivetrain mass in kg
        m = 52.0
        # Radius of wheels in meters
        r = 0.08255 / 2.0
        # Radius of robot in meters
        self.rb = 0.59055 / 2.0
        # Moment of inertia of the differential drive in kg-m²
        J = 6.0

        motor = fct.models.gearbox(fct.models.MOTOR_CIM, num_motors)

        C1 = -(G**2) * motor.Kt / (motor.Kv * motor.R * r**2)
        C2 = G * motor.Kt / (motor.R * r)
        C3 = -(G**2) * motor.Kt / (motor.Kv * motor.R * r**2)
        C4 = G * motor.Kt / (motor.R * r)
        self.velocity_A = np.array(
            [
                [(1 / m + self.rb**2 / J) * C1, (1 / m - self.rb**2 / J) * C3],
                [(1 / m - self.rb**2 / J) * C1, (1 / m + self.rb**2 / J) * C3],
            ]
        )
        self.velocity_B = np.array(
            [
                [(1 / m + self.rb**2 / J) * C2, (1 / m - self.rb**2 / J) * C4],
                [(1 / m - self.rb**2 / J) * C2, (1 / m + self.rb**2 / J) * C4],
            ]
        )

        # Linearized dynamics
        self.Ac = np.zeros((5, 5))
        self.Ac[2, 3] = -0.5 / self.rb
        self.Ac[2, 4] = 0.5 / self.rb
        self.Ac[3:5, 3:5] = self.velocity_A
        self.Bc = np.block(
            [
                [np.zeros((3, 2))],
                [self.velocity_B],
            ]
        )

        # Sim variables
        self.x = np.zeros((5, 1))
        self.u = np.zeros((2, 1))
        self.y = np.zeros((3, 1))

        self.u_min = np.array([[-12.0], [-12.0]])
        self.u_max = np.array([[12.0], [12.0]])

        # States: x (m), y (m), heading (rad), left velocity (m/s),
        #         right velocity (m/s)
        # Q = diag(1/q²)
        # Q⁻¹ = diag(q²)
        self.Qinv = np.diag(np.square([0.125, 0.125, 10.0, 0.95, 0.95]))

        # Inputs: Left voltage (V), right voltage (V)
        # R = diag(1/r²)
        # R⁻¹ = diag(r²)
        self.Rinv = np.diag(np.square([12.0, 12.0]))

        self.t = 0

        # Get reference trajectory
        self.refs = get_square_refs()

        # Kalman smoother storage
        self.x_hat_pre = [np.zeros((5, 1)) for _ in range(len(self.refs))]
        self.x_hat_post = [np.zeros((5, 1)) for _ in range(len(self.refs))]
        self.A = [np.zeros((5, 5)) for _ in range(len(self.refs))]
        self.P_pre = [np.zeros((5, 5)) for _ in range(len(self.refs))]
        self.P_post = [np.zeros((5, 5)) for _ in range(len(self.refs))]
        self.x_hat_smooth = [np.zeros((5, 1)) for _ in range(len(self.refs))]

    def f(self, x, u):
        """
        Nonlinear differential drive dynamics.

        States: [[x], [y], [heading], [left velocity], [right velocity]]
        Inputs: [[left voltage], [right voltage]]

        Keyword arguments:
        x -- state vector
        u -- input vector

        Returns:
        dx/dt -- state derivative
        """
        return np.block(
            [
                [(x[3, 0] + x[4, 0]) / 2.0 * math.cos(x[2, 0])],
                [(x[3, 0] + x[4, 0]) / 2.0 * math.sin(x[2, 0])],
                [(x[4, 0] - x[3, 0]) / (2.0 * self.rb)],
                [self.velocity_A @ x[3:5, :] + self.velocity_B @ u],
            ]
        )

    def h(self, x):
        """
        Nonlinear differential drive measurement model.

        Outputs: [[x], [y], [heading]]

        Keyword arguments:
        x -- state vector

        Returns:
        y -- measurement vector
        """
        return x[:3, :]

    # pragma pylint: disable=unused-argument
    def df_dx(self, x, u):
        """
        Returns the Jacobian of f with respect to the state.

        Keyword arguments:
        x -- the current state
        u -- the current input
        """
        v = (x[3, 0] + x[4, 0]) / 2.0
        c = math.cos(x[2, 0])
        s = math.sin(x[2, 0])
        self.Ac[0, 2] = -v * s
        self.Ac[0, 3] = 0.5 * c
        self.Ac[0, 4] = 0.5 * c
        self.Ac[1, 2] = v * c
        self.Ac[1, 3] = 0.5 * s
        self.Ac[1, 4] = 0.5 * s
        return self.Ac

    # pragma pylint: disable=unused-argument
    def dh_dx(self, x, u):
        """
        Returns the Jacobian of h with respect to the state.

        Keyword arguments:
        x -- the current state
        u -- the current input
        """
        return np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])

    # pragma pylint: disable=unused-argument
    def update(self, r, next_r):
        """
        Advance the model by one timestep.

        Keyword arguments:
        r -- the current reference
        next_r -- the next reference
        """
        self.x = fct.rk4(self.f, self.x, self.u, self.dt)

        start = time.time()
        # Since this is the last reference, there are no reference dynamics to
        # follow
        if self.t == len(self.refs) - 1:
            self.u = np.zeros((2, 1))
            return

        states = self.x.shape[0]

        # Linearize model
        Ac = self.df_dx(self.x, np.zeros((2, 1)))
        _, B = fct.discretize_ab(Ac, self.Bc, self.dt)
        Binv = np.linalg.pinv(B)

        self.x_hat_pre[self.t] = self.x.copy()
        self.P_pre[self.t] = np.zeros((states, states))
        self.x_hat_post[self.t] = self.x.copy()
        self.P_post[self.t] = np.zeros((states, states))

        # Prediction
        N = min(len(self.refs) - 1, self.t + HORIZON)
        for τ in range(self.t + 1, N + 1):
            self.x_hat_pre[τ] = fct.rk4(
                self.f, self.x_hat_post[τ - 1], np.zeros((2, 1)), self.dt
            )

            # Linearization
            self.A[τ - 1] = fct.discretize_a(
                self.df_dx(self.x_hat_post[τ - 1], np.zeros((2, 1))), self.dt
            )
            C = self.dh_dx(self.x_hat_pre[τ], np.zeros((2, 1)))

            s_τ = C @ self.refs[τ]

            self.P_pre[τ] = (
                self.A[τ - 1] @ self.P_post[τ - 1] @ self.A[τ - 1].T
                + B @ self.Rinv @ B.T
            )

            # S = CPCᵀ + CQ⁻¹Cᵀ
            S = C @ self.P_pre[τ] @ C.T + C @ self.Qinv @ C.T

            # We want to put K = PCᵀS⁻¹ into Ax = b form so we can solve it more
            # efficiently.
            #
            # K = PCᵀS⁻¹
            # KS = PCᵀ
            # (KS)ᵀ = (PCᵀ)ᵀ
            # SᵀKᵀ = CPᵀ
            #
            # The solution of Ax = b can be found via x = A.solve(b).
            #
            # Kᵀ = Sᵀ.solve(CPᵀ)
            # K = Sᵀ.solve(CPᵀ)ᵀ
            K = np.linalg.solve(S.T, C @ self.P_pre[τ].T).T

            self.x_hat_post[τ] = self.x_hat_pre[τ] + K @ (
                s_τ - self.h(self.x_hat_pre[τ])
            )
            self.P_post[τ] = (np.eye(5) - K @ C) @ self.P_pre[τ] @ (
                np.eye(5) - K @ C
            ).T + K @ C @ self.Qinv @ C.T @ K.T

        # Last filtered estimate is already optimal smoothed estimate
        self.x_hat_smooth[N] = self.x_hat_post[N]

        # Smoothing
        for τ in range(N - 1, (self.t + 1) - 1, -1):
            # L = P⁺[τ] A[τ]ᵀ P⁻[τ + 1]⁻¹
            # L P⁻[τ + 1] = P⁺[τ] A[τ]ᵀ
            # P⁻[τ + 1]ᵀ Lᵀ = A[τ] P⁺[τ]ᵀ
            # Lᵀ = P⁻[τ + 1]ᵀ.solve(A[τ] P⁺[τ]ᵀ)
            # L = P⁻[τ + 1]ᵀ.solve(A[τ] P⁺[τ]ᵀ)ᵀ
            try:
                L = np.linalg.solve(self.P_pre[τ + 1].T, self.A[τ] @ self.P_post[τ].T).T
            except np.linalg.LinAlgError:
                L = self.P_post[τ] @ self.A[τ].T @ np.linalg.pinv(self.P_pre[τ + 1])
            self.x_hat_smooth[τ] = self.x_hat_post[τ] + L @ (
                self.x_hat_smooth[τ + 1] - self.x_hat_pre[τ + 1]
            )

        # x̂ₖ₊₁ = f(x̂ₖ) + Buₖ
        # Buₖ = x̂ₖ₊₁ − f(x̂ₖ)
        # uₖ = B⁺(x̂ₖ₊₁ − f(x̂ₖ))
        self.u = Binv @ (
            self.x_hat_smooth[self.t + 1]
            - fct.rk4(self.f, self.x_hat_post[self.t], np.zeros((2, 1)), self.dt)
        )

        u_cap = np.max(np.abs(self.u))
        if u_cap > 12.0:
            self.u = self.u / u_cap * 12.0
        self.t += 1

        end = time.time()
        print(
            f"\riteration: {self.t}/{len(self.refs) - 1}, dt = {round((end - start) * 1e3)} ms ",
            end="",
        )


def main():
    """Entry point."""
    refs = get_square_refs()

    t = [0.0]
    for _ in range(len(refs) - 1):
        t.append(t[-1] + DT)

    x = np.zeros((5, 1))
    diff_drive = DifferentialDrive(DT)
    diff_drive.x = x.copy()

    start = time.time()
    ref_rec, state_rec, u_rec, _ = fct.generate_time_responses(diff_drive, refs)
    end = time.time()
    print(f"\nTotal time = {round(end - start, 3)} s")

    plt.figure(1)
    plt.plot(ref_rec[0, :], ref_rec[1, :], label="Reference trajectory")
    plt.plot(state_rec[0, :], state_rec[1, :], label="ERTS controller")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()

    plt.gca().set_aspect(1.0)
    plt.gca().set_box_aspect(1.0)

    if "--noninteractive" in sys.argv:
        plt.savefig("erts_diff_drive_xy.png")

    fct.plot_time_responses(
        [
            "x position (m)",
            "y position (m)",
            "Heading (rad)",
            "Left velocity (m/s)",
            "Right velocity (m/s)",
        ],
        ["Left voltage (V)", "Right voltage (V)"],
        t,
        ref_rec,
        state_rec,
        u_rec,
    )

    if "--noninteractive" in sys.argv:
        plt.savefig("erts_diff_drive_response.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
