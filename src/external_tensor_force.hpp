#pragma once
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cctype>
#include <cassert>
#include "particle_simulator.hpp"

class IOParamsContainer;
template <class Type> class IOParams;

class TidalTensorManager {
public:
    struct TensorSnapshot {
        double time;    // time in PeTar units (after scaling)
        double T[3][3]; // full 3x3 tensor already scaled to PeTar units
    };

private:
    std::vector<TensorSnapshot> snapshots_;
    double last_query_time_ = 0.0;
    TensorSnapshot interp_{};
    bool loaded_ = false;
    PS::F64vec ref_point_{0.0, 0.0, 0.0};

    // Header as in NBODY6tt style
    int    NBTT_     = 0;    // number of rows expected
    double TTUNIT_   = 1.0;  // time unit of the file's time column, in "TT time units"
    double TTOFFSET_ = 0.0;  // time offset in "TT time units"

    // Scaling between IN (PeTar) and TT systems provided by the user
    double rscale_ = 1.0;
    double vscale_ = 1.0;

public:
    TidalTensorManager() {
        interp_.time = 0.0;
        for (int i=0;i<3;i++) for(int j=0;j<3;j++) interp_.T[i][j]=0.0;
    }

    void setScales(double rscale, double vscale) {
        rscale_ = rscale;
        vscale_ = vscale;
    }

    bool isLoaded() const { return loaded_; }

    // loadFromFile: read nbtt, ttunit, ttoffset and NBODY6-style table
    bool loadFromFile(const std::string &filename) {
        snapshots_.clear();
        FILE *fp = std::fopen(filename.c_str(), "r");
        if (!fp) {
            std::cerr << "[TT] ERROR: cannot open '" << filename << "'\n";
            return false;
        }

        if (std::fscanf(fp, "%d %lf %lf", &NBTT_, &TTUNIT_, &TTOFFSET_) != 3) {
            std::cerr << "[TT] ERROR: failed to read header (NBTT TTUNIT TTOFFSET)\n";
            std::fclose(fp);
            return false;
        }
        if (NBTT_ <= 0) {
            std::cerr << "[TT] ERROR: NBTT <= 0 in tt.dat\n";
            std::fclose(fp);
            return false;
        }

        const double tt_to_in_time = (vscale_ / rscale_) * TTUNIT_;
        const double tens_scale    = (rscale_ / vscale_) * (rscale_ / vscale_);

        snapshots_.reserve(std::max(1, NBTT_));

        for (int k=0; k<NBTT_; ++k) {
            char linebuf[1024];
            if (!std::fgets(linebuf, sizeof(linebuf), fp)) break;

            // skip empty/comment lines
            char *ptr = linebuf;
            while (*ptr && std::isspace(*ptr)) ++ptr;
            if (*ptr=='\0' || *ptr=='#') { --k; continue; }

            double t; double v[9];
            int n = std::sscanf(linebuf,
                                "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                                &t, &v[0],&v[1],&v[2],&v[3],&v[4],&v[5],&v[6],&v[7],&v[8]);

            TensorSnapshot s{};
            if (n == 10) {
                s.time = t * tt_to_in_time + TTOFFSET_ * (vscale_ / rscale_);
                s.T[0][0]=v[0]*tens_scale; s.T[0][1]=v[1]*tens_scale; s.T[0][2]=v[2]*tens_scale;
                s.T[1][0]=v[3]*tens_scale; s.T[1][1]=v[4]*tens_scale; s.T[1][2]=v[5]*tens_scale;
                s.T[2][0]=v[6]*tens_scale; s.T[2][1]=v[7]*tens_scale; s.T[2][2]=v[8]*tens_scale;
                snapshots_.push_back(s);
            } else {
                double t2,a,b,c,d,e,f;
                int n7 = std::sscanf(linebuf, "%lf %lf %lf %lf %lf %lf %lf",
                                     &t2,&a,&b,&c,&d,&e,&f);
                if (n7 == 7) {
                    s.time = t2 * tt_to_in_time + TTOFFSET_ * (vscale_ / rscale_);
                    s.T[0][0]=a*tens_scale; s.T[0][1]=b*tens_scale; s.T[0][2]=c*tens_scale;
                    s.T[1][0]=b*tens_scale; s.T[1][1]=d*tens_scale; s.T[1][2]=e*tens_scale;
                    s.T[2][0]=c*tens_scale; s.T[2][1]=e*tens_scale; s.T[2][2]=f*tens_scale;
                    snapshots_.push_back(s);
                } else {
                    std::cerr << "[TT] WARNING: unrecognized tt.dat line, skipping:\n  " << linebuf;
                    --k; continue;
                }
            }
        }
        std::fclose(fp);

        if (snapshots_.empty()) {
            std::cerr << "[TT] ERROR: no valid tensor snapshots loaded\n";
            return false;
        }

        std::sort(snapshots_.begin(), snapshots_.end(),
                  [](const TensorSnapshot &a, const TensorSnapshot &b){ return a.time < b.time; });

        interp_ = snapshots_.front();
        last_query_time_ = interp_.time;
        loaded_ = true;

        std::cerr << "[TT] Loaded " << snapshots_.size() << " snapshots; time range: "
                  << snapshots_.front().time << " -> " << snapshots_.back().time << " (PeTar units)\n"
                  << "      NBTT="<<NBTT_<<", TTUNIT="<<TTUNIT_<<", TTOFFSET="<<TTOFFSET_<<"\n"
                  << "      scales: rscale="<<rscale_<<", vscale="<<vscale_<<"\n";
        return true;
    }

    // Set the reference point r0(t). Default is origin.
    void setReference(const PS::F64vec &r0) { ref_point_ = r0; }

    void update(double time_now) {
        if (!loaded_) return;
        if (time_now == last_query_time_) return;

        if (time_now <= snapshots_.front().time) {
            interp_ = snapshots_.front(); interp_.time = time_now; last_query_time_ = time_now; return;
        }
        if (time_now >= snapshots_.back().time) {
            interp_ = snapshots_.back();  interp_.time = time_now; last_query_time_ = time_now; return;
        }

        auto it_hi = std::upper_bound(snapshots_.begin(), snapshots_.end(), time_now,
                                      [](double t, const TensorSnapshot &s){ return t < s.time; });
        auto it_lo = it_hi - 1;
        const double t0 = it_lo->time, t1 = it_hi->time;
        const double alpha = (time_now - t0) / (t1 - t0);

        interp_.time = time_now;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++)
            interp_.T[i][j] = (1.0-alpha)*it_lo->T[i][j] + alpha*it_hi->T[i][j];

        last_query_time_ = time_now;
    }

    // Apply interpolated tensor to a position (pos - ref_point)
    PS::F64vec applyTensor(const PS::F64vec &pos) const {
        PS::F64vec rel = pos - ref_point_;
        PS::F64vec a;
        a.x = interp_.T[0][0]*rel.x + interp_.T[0][1]*rel.y + interp_.T[0][2]*rel.z;
        a.y = interp_.T[1][0]*rel.x + interp_.T[1][1]*rel.y + interp_.T[1][2]*rel.z;
        a.z = interp_.T[2][0]*rel.x + interp_.T[2][1]*rel.y + interp_.T[2][2]*rel.z;
        return a;
    }

    bool getTimeRange(double &tmin, double &tmax) const {
        if (!loaded_) return false;
        tmin = snapshots_.front().time;
        tmax = snapshots_.back().time;
        return true;
    }
};

// IO layer (mirrors Galpy IO conventions)
class IOParamsExternalTensor {
public:
    IOParams<std::string> fname_tt;
    IOParams<double>      rscale;
    IOParams<double>      vscale;

    IOParamsExternalTensor(IOParamsContainer &store)
    : fname_tt(store, "tt.dat", "tidal-tensor-file", "Filename for tidal tensor data"),
      rscale  (store, 1.0,     "tt-rscale",         "Length scale from IN to TT"),
      vscale  (store, 1.0,     "tt-vscale",         "Velocity scale from IN to TT")
    {}
};

// High-level manager used by PeTar (like GalpyManager)
class ExternalTensorManager {
    TidalTensorManager tt_;
public:
    bool initial(const IOParamsExternalTensor &io, bool print_flag=true) {
        tt_.setScales(io.rscale.value, io.vscale.value);
        bool ok = tt_.loadFromFile(io.fname_tt.value);
        if (!ok) return false;
        if (print_flag) {
            double tmin, tmax;
            if (tt_.getTimeRange(tmin, tmax)) {
                std::cerr << "[TT] Active. File='"<<io.fname_tt.value<<"', time range "<<tmin<<" to "<<tmax<<"\n";
            }
        }
        return true;
    }

    void setReference(const PS::F64vec &r0) { tt_.setReference(r0); }
    void update(double t_now)               { tt_.update(t_now);    }
    PS::F64vec accel(const PS::F64vec &pos) const { return tt_.applyTensor(pos); }
    bool isLoaded() const { return tt_.isLoaded(); }
};
