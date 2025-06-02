// Copyright (c) Tyler Veness

#pragma once

#include <array>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

template <typename T, size_t N>
void write_vector_csv(std::string_view filename,
                      const std::vector<double>& times,
                      const std::vector<T>& data,
                      std::array<std::string_view, N> labels) {
  std::ofstream csv{std::string{filename}};
  if (!csv.is_open()) {
    return;
  }

  // Write labels
  for (size_t i = 0; i < labels.size(); ++i) {
    csv << labels[i];
    if (i < labels.size() - 1) {
      csv << ',';
    }
  }
  csv << '\n';

  // Write data
  for (size_t i = 0; i < data.size(); ++i) {
    csv << times[i] << ',';

    const auto& vec = data[i];
    for (int j = 0; j < vec.rows(); ++j) {
      csv << vec(j);
      if (j < vec.rows() - 1) {
        csv << ',';
      }
    }
    csv << '\n';
  }
}
