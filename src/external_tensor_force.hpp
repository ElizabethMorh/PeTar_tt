#ifndef EXTERNAL_TENSOR_FORCE_HPP
#define EXTERNAL_TENSOR_FORCE_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <cmath>

/****************************************************
 * IO parameters
 ****************************************************/
class IOParamsTT {
public:
    bool print_flag;
    std::string tt_filename;

    IOParamsTT()
        : print_flag(false),
          tt_filename("tt.dat") {}

    template<class Tinput>
    void read(Tinput& input, int argc, char** argv) {
        input(tt_filename, "tt-file", "External tidal tensor file");
    }
};

/****************************************************
 * Tidal tensor reader & interpolator
 ****************************************************/
class TidalTensorManager {
public:
    struct Snapshot {
        double time;
        double T[3][3];
    };

private:
    std::vector<Snapshot> snaps_;
    bool loaded_ = false;

    size_t last_idx_ = 0;
    double last_time_ = -1.0;

    double T_[3][3];

public:
    bool load(const std::string& fname) {
        std::ifstream fin(fname);
        if (!fin) {
            std::cerr << "[TT] Cannot open " << fname << "\n";
            return false;
        }

        snaps_.clear();

        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty() || line[0]=='#') continue;

            Snapshot s;
            std::istringstream iss(line);
            iss >> s.time;

            for (int i=0;i<3;i++)
                for (int j=0;j<3;j++)
                    iss >> s.T[i][j];

            snaps_.push_back(s);
        }

        if (snaps_.size() < 2) {
            std::cerr << "[TT] Need at least 2 snapshots\n";
            return false;
        }

        loaded_ = true;
        last_idx_ = 0;
        last_time_ = -1.0;

        return true;
    }

    void update(double t) {
        assert(loaded_);

        if (t == last_time_) return;
        last_time_ = t;

        if (t <= snaps_.front().time) {
            copy(snaps_.front().T);
            return;
        }

        if (t >= snaps_.back().time) {
            copy(snaps_.back().T);
            return;
        }

        while (last_idx_+1 < snaps_.size() &&
               snaps_[last_idx_+1].time < t)
            last_idx_++;

        const auto& s0 = snaps_[last_idx_];
        const auto& s1 = snaps_[last_idx_+1];

        double w = (t - s0.time) / (s1.time - s0.time);

        for (int i=0;i<3;i++)
            for (int j=0;j<3;j++)
                T_[i][j] = (1.0-w)*s0.T[i][j] + w*s1.T[i][j];
    }

    const double* row(int i) const { return T_[i]; }

private:
    void copy(const double Tin[3][3]) {
        for (int i=0;i<3;i++)
            for (int j=0;j<3;j++)
                T_[i][j] = Tin[i][j];
    }
};

/****************************************************
 * PeTar-facing manager (Galpy-compatible interface)
 ****************************************************/
class TTManager {
private:
    TidalTensorManager tt_;
    bool print_flag_ = false;

public:
    TTManager() = default;

    void initial(const IOParamsTT& params,
                 const double time,
                 const std::string&,
                 const bool restart_flag,
                 const bool print_flag) {

        print_flag_ = print_flag;

        if (!restart_flag) {
            if (!tt_.load(params.tt_filename)) {
                std::cerr << "[TTManager] Failed to load tt.dat\n";
                std::abort();
            }
        }

        tt_.update(time);

        if (print_flag_)
            std::cerr << "[TTManager] Initialized\n";
    }

    /************************************************
     * Acceleration + potential
     *
     * acc_i = sum_j T_ij x_j
     * Phi   = 0.5 sum_ij T_ij x_i x_j
     ************************************************/
    void calcAccPot(double acc[3],
                    double& pot,
                    const double time,
                    const double,
                    const double* pos,
                    const double* pos_ref) {

        tt_.update(time);

        double x[3] = {
            pos[0] - pos_ref[0],
            pos[1] - pos_ref[1],
            pos[2] - pos_ref[2]
        };

        acc[0] = acc[1] = acc[2] = 0.0;
        pot = 0.0;

        for (int i=0;i<3;i++) {
            for (int j=0;j<3;j++) {
                double Tij = tt_.row(i)[j];
                acc[i] += Tij * x[j];
                pot    += 0.5 * Tij * x[i] * x[j];
            }
        }
    }

    void printData(std::ostream& out) const {
        out << "External potential: Tidal Tensor (tt.dat)\n";
    }

    void writePotentialPars(const std::string& fname,
                            const double time) const {
        std::ofstream fout(fname.c_str());
        fout << "# Tidal tensor external potential\n";
        fout << "# time = " << time << "\n";
    }
};

#endif // EXTERNAL_TENSOR_FORCE_HPP
