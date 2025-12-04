// Copyright (c) Tyler Veness

#pragma once

#include <numbers>

/// Holds the constants for a DC motor.
class DCMotor {
 public:
  /// Voltage at which the motor constants were measured.
  double nominal_voltage;

  /// Torque when stalled.
  double stall_torque;

  /// Current draw when stalled.
  double stall_current;

  /// Current draw under no load.
  double free_current;

  /// Angular velocity under no load.
  double free_speed;

  /// Motor internal resistance.
  double R;

  /// Motor velocity constant.
  double Kv;

  /// Motor torque constant.
  double Kt;

  /// Constructs a DC motor.
  ///
  /// @param nominal_voltage Voltage at which the motor constants were measured.
  /// @param stall_torque Torque when stalled.
  /// @param stall_current Current draw when stalled.
  /// @param free_current Current draw under no load.
  /// @param free_speed Angular velocity under no load.
  /// @param num_motors Number of motors in a gearbox.
  constexpr DCMotor(double nominal_voltage, double stall_torque,
                    double stall_current, double free_current,
                    double free_speed, int num_motors = 1)
      : nominal_voltage(nominal_voltage),
        stall_torque(stall_torque * num_motors),
        stall_current(stall_current * num_motors),
        free_current(free_current * num_motors),
        free_speed(free_speed),
        R(nominal_voltage / this->stall_current),
        Kv(free_speed / (nominal_voltage - R * this->free_current)),
        Kt(this->stall_torque / this->stall_current) {}

  /// Returns a gearbox of CIM motors.
  static constexpr DCMotor cim(int num_motors = 1) {
    return DCMotor(12.0, 2.42, 133.0, 2.7, rpm2radpsec(5310.0), num_motors);
  }

 private:
  static constexpr double rpm2radpsec(double rpm) {
    return rpm * 2.0 * std::numbers::pi / 60.0;
  }
};
