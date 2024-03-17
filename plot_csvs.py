#!/usr/bin/env python3

"""Plots ERTS results."""

import sys

import frccontrol as fct
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Entry point."""

    t_rec = np.genfromtxt("ERTS references.csv", delimiter=",", skip_header=1)[:, 0]
    ref_rec = np.genfromtxt("ERTS references.csv", delimiter=",", skip_header=1)[
        :, 1:
    ].T
    state_rec = np.genfromtxt("ERTS states.csv", delimiter=",", skip_header=1)[:, 1:].T
    u_rec = np.genfromtxt("ERTS inputs.csv", delimiter=",", skip_header=1)[:, 1:].T

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
        t_rec,
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
