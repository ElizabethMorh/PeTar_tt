#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <cmath>

/*
 * External tidal tensor manager
 *
 * Reads NBODY6tt-style tt.dat files:
 *   time  Txx Txy Txz  Tyx Tyy Tyz  Tzx Tzy Tzz
 *
 * Tensor definition:
 *   T_ij = d^2 Phi_ext / (dx_i dx_j)
 *
 * No force application here — this class ONLY:
 *   - loads
 *   - interpolates
 *   - stores the tensor
 */

class TidalTensorManager {
public:
    struct TensorSnapshot {
        double time;
        double T[3][3];
    };

private:
    std::vector<TensorSnapshot> snapshots_;
    bool loaded_ = false;

    // interpolation cache
    size_t last_index_ = 0;
    double last_query_time_ = -1.0;

    double T_interp_[3][3];

public:
    TidalTensorManager() = default;

    bool isLoaded() const { return loaded_; }

    size_t size() const { return snapshots_.size(); }

    // --------------------------------------------------
    // Load tt.dat
    // --------------------------------------------------
    bool loadFromFile(const std::string& filename) {
        std::ifstream fin(filename);
        if (!fin) {
            std::cerr << "[TidalTensorManager] Cannot open file: "
                      << filename << std::endl;
            return false;
        }

        snapshots_.clear();

        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            TensorSnapshot snap;

            if (!(iss >> snap.time)) continue;

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    iss >> snap.T[i][j];
                }
            }

            snapshots_.push_back(snap);
        }

        if (snapshots_.size() < 2) {
            std::cerr << "[TidalTensorManager] Need >= 2 snapshots\n";
            loaded_ = false;
            return false;
        }

        loaded_ = true;
        last_index_ = 0;
        last_query_time_ = -1.0;

        std::cerr << "[TidalTensorManager] Loaded "
                  << snapshots_.size()
                  << " tensor snapshots\n";

        return true;
    }

    // --------------------------------------------------
    // Interpolate tensor at time t
    // --------------------------------------------------
    void update(double t) {
        assert(loaded_);

        if (t == last_query_time_) return;
        last_query_time_ = t;

        // clamp to bounds
        if (t <= snapshots_.front().time) {
            copyTensor(snapshots_.front().T);
            return;
        }

        if (t >= snapshots_.back().time) {
            copyTensor(snapshots_.back().T);
            return;
        }

        // advance cached index
        while (last_index_ + 1 < snapshots_.size() &&
               snapshots_[last_index_ + 1].time < t) {
            last_index_++;
        }

        const auto& s0 = snapshots_[last_index_];
        const auto& s1 = snapshots_[last_index_ + 1];

        double dt = s1.time - s0.time;
        assert(dt > 0.0);

        double w = (t - s0.time) / dt;

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                T_interp_[i][j] =
                    (1.0 - w) * s0.T[i][j] + w * s1.T[i][j];
            }
        }
    }

    // --------------------------------------------------
    // Access interpolated tensor
    // --------------------------------------------------
    double get(int i, int j) const {
        assert(i >= 0 && i < 3);
        assert(j >= 0 && j < 3);
        return T_interp_[i][j];
    }

    const double* getRow(int i) const {
        assert(i >= 0 && i < 3);
        return T_interp_[i];
    }

private:
    void copyTensor(const double T[3][3]) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                T_interp_[i][j] = T[i][j];
            }
        }
    }
};