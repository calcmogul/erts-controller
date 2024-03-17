// Copyright (c) Tyler Veness

#pragma once

#include <numbers>

/**
 * Holds the constants for a DC motor.
 */
class DCMotor {
 public:
  /// Voltage at which the motor constants were measured.
  double nominalVoltage;

  /// Torque when stalled.
  double stallTorque;

  /// Current draw when stalled.
  double stallCurrent;

  /// Current draw under no load.
  double freeCurrent;

  /// Angular velocity under no load.
  double freeSpeed;

  /// Motor internal resistance.
  double R;

  /// Motor velocity constant.
  double Kv;

  /// Motor torque constant.
  double Kt;

  /**
   * Constructs a DC motor.
   *
   * @param nominalVoltage Voltage at which the motor constants were measured.
   * @param stallTorque Torque when stalled.
   * @param stallCurrent Current draw when stalled.
   * @param freeCurrent Current draw under no load.
   * @param freeSpeed Angular velocity under no load.
   * @param numMotors Number of motors in a gearbox.
   */
  constexpr DCMotor(double nominalVoltage, double stallTorque,
                    double stallCurrent, double freeCurrent, double freeSpeed,
                    int numMotors = 1)
      : nominalVoltage(nominalVoltage),
        stallTorque(stallTorque * numMotors),
        stallCurrent(stallCurrent * numMotors),
        freeCurrent(freeCurrent * numMotors),
        freeSpeed(freeSpeed),
        R(nominalVoltage / this->stallCurrent),
        Kv(freeSpeed / (nominalVoltage - R * this->freeCurrent)),
        Kt(this->stallTorque / this->stallCurrent) {}

  /**
   * Returns a gearbox of CIM motors.
   */
  static constexpr DCMotor CIM(int numMotors = 1) {
    return DCMotor(12.0, 2.42, 133.0, 2.7, rpm2radpsec(5310.0), numMotors);
  }

 private:
  static constexpr double rpm2radpsec(double rpm) {
    return rpm * 2.0 * std::numbers::pi / 60.0;
  }
};
