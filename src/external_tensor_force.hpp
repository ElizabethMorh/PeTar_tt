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

#ifndef PS_F64VEC_DEFINED
#define PS_F64VEC_DEFINED
namespace PS {
    struct F64vec {
        double x, y, z;
        F64vec() : x(0.0), y(0.0), z(0.0) {}
        F64vec(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
        F64vec operator-(const F64vec &o) const { return F64vec(x-o.x, y-o.y, z-o.z); }
        F64vec operator+(const F64vec &o) const { return F64vec(x+o.x, y+o.y, z+o.z); }
        PS::F64vec& operator+=(const PS::F64vec &o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
    };
}
#endif

template<typename T> class IOParams; // Declare PeTar's IOParams store type

class TidalTensorManager {
public:
    struct TensorSnapshot {
        double time;    // time in PeTar units (after scaling)
        double T[3][3]; // full 3x3 tensor already scaled to PeTar units
    };

private:
    std::vector<TensorSnapshot> snapshots;
    double last_query_time_ = 0.0;
    TensorSnapshot interp_; // last interpolated tensor
    bool loaded_ = false;
    PS::F64vec ref_point_{0.0,0.0,0.0};

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

    bool isLoaded() const { return loaded_; }
    void setScales(double rscale, double vscale) {
        rscale_ = rscale;
        vscale_ = vscale;
    }

    // loadFromFile: read nbtt, ttunit, ttoffset and NBODY6-style table
    bool loadFromFile(const std::string &filename) {
        snapshots.clear();
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
        snapshots.reserve(std::max(1, NBTT_));

        for (int k=0; k<NBTT_; ++k) {
            char linebuf[1024];
            if (!std::fgets(linebuf, sizeof(linebuf), fp)) break; // EOF or error

            // skip empty or comment
            char *ptr = linebuf;
            while (*ptr && std::isspace(*ptr)) ++ptr;
            if (*ptr=='\0' || *ptr=='#') { --k; continue; } // ignore and retry same k

            double t; double vals[9];
            int nread = std::sscanf(linebuf,
                                    "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                                    &t,
                                    &vals[0], &vals[1], &vals[2],
                                    &vals[3], &vals[4], &vals[5],
                                    &vals[6], &vals[7], &vals[8]);

            TensorSnapshot s{};
            if (nread == 10) {
                s.time = t * tt_to_in_time + TTOFFSET_ * (vscale_ / rscale_);
                s.T[0][0] = vals[0]*tens_scale; s.T[0][1] = vals[1]*tens_scale; s.T[0][2] = vals[2]*tens_scale;
                s.T[1][0] = vals[3]*tens_scale; s.T[1][1] = vals[4]*tens_scale; s.T[1][2] = vals[5]*tens_scale;
                s.T[2][0] = vals[6]*tens_scale; s.T[2][1] = vals[7]*tens_scale; s.T[2][2] = vals[8]*tens_scale;
                snapshots.push_back(s);
            } else {
                double t2,a,b,c,d,e,f;
                int n7 = std::sscanf(linebuf, "%lf %lf %lf %lf %lf %lf %lf",
                                     &t2,&a,&b,&c,&d,&e,&f);
                if (n7==7) {
                    s.time = t2 * tt_to_in_time + TTOFFSET_ * (vscale_ / rscale_);
                    s.T[0][0] = a*tens_scale; s.T[0][1] = b*tens_scale; s.T[0][2] = c*tens_scale;
                    s.T[1][0] = b*tens_scale; s.T[1][1] = d*tens_scale; s.T[1][2] = e*tens_scale;
                    s.T[2][0] = c*tens_scale; s.T[2][1] = e*tens_scale; s.T[2][2] = f*tens_scale;
                    snapshots.push_back(s);
                } else {
                    std::cerr << "[TT] WARNING: unrecognized tt.dat line, skipping:\n  " << linebuf;
                    --k; continue;
                }
            }
        }
        std::fclose(fp);

        if (snapshots.empty()) {
            std::cerr << "[TT] ERROR: no valid tensor snapshots loaded\n";
            return false;
        }

        std::sort(snapshots.begin(), snapshots.end(),
                  [](const TensorSnapshot &a, const TensorSnapshot &b){ return a.time < b.time; });

        interp_ = snapshots.front();
        last_query_time_ = interp_.time;
        loaded_ = true;

        std::cerr << "[TT] Loaded " << snapshots.size() << " snapshots; time range: "
                  << snapshots.front().time << " -> " << snapshots.back().time << " (PeTar units)\n"
                  << "      NBTT="<<NBTT_<<", TTUNIT="<<TTUNIT_<<", TTOFFSET="<<TTOFFSET_<<"\n"
                  << "      scales: rscale="<<rscale_<<", vscale="<<vscale_<<"\n";
        return true;
    }

    // Set the reference point r0(t). Default is origin.
    void setReference(const PS::F64vec &r0) { ref_point_ = r0; }

    void update(double time) {
        if (!loaded_) return;
        if (time == last_query_time_) return;

        if (time <= snapshots.front().time) { interp_ = snapshots.front(); interp_.time = time; last_query_time_=time; return; }
        if (time >= snapshots.back().time)  { interp_ = snapshots.back();  interp_.time = time; last_query_time_=time; return; }

        auto it_hi = std::upper_bound(snapshots.begin(), snapshots.end(), time,
                                      [](double t, const TensorSnapshot &s){ return t < s.time; });
        auto it_lo = it_hi - 1;
        double t_lo = it_lo->time, t_hi2 = it_hi->time;
        double alpha = (time - t_lo) / (t_hi2 - t_lo);

        interp_.time = time;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++)
            interp_.T[i][j] = (1.0 - alpha)*it_lo->T[i][j] + alpha*it_hi->T[i][j];

        last_query_time_ = time;
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

    // Convenience: multiply an arbitrary tensor by a vector
    static PS::F64vec multiplyTensor(const double Tmat[3][3], const PS::F64vec &v) {
        PS::F64vec r;
        r.x = Tmat[0][0]*v.x + Tmat[0][1]*v.y + Tmat[0][2]*v.z;
        r.y = Tmat[1][0]*v.x + Tmat[1][1]*v.y + Tmat[1][2]*v.z;
        r.z = Tmat[2][0]*v.x + Tmat[2][1]*v.y + Tmat[2][2]*v.z;
        return r;
    }

    bool getTimeRange(double &tmin, double &tmax) const {
        if (!loaded_) return false;
        tmin = snapshots.front().time; tmax = snapshots.back().time; return true;
    }
    size_t size() const { return snapshots.size(); }
};

// IO layer (mirrors Galpy IO conventions)
class IOParamsExternalTensor {
public:
    IOParams<std::string> mode;      // "none" | "galpy" | "tidal_tensor" (shared key)
    IOParams<std::string> fname_tt;  // tt.dat filename
    IOParams<double>      rscale;    // IN->TT length scale (1.0 default)
    IOParams<double>      vscale;    // IN->TT velocity scale (1.0 default)

    IOParamsExternalTensor(std::vector<void*> &input_par_store)
    : mode     (input_par_store, "none", "external-force-mode", "Type of external force: none, galpy, tidal_tensor"),
      fname_tt (input_par_store, "tt.dat", "tidal-tensor-file", "Filename for tidal tensor data"),
      rscale   (input_par_store, 1.0, "tt-rscale", "Length scale from IN to TT unit"),
      vscale   (input_par_store, 1.0, "tt-vscale", "Velocity scale from IN to TT unit")
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
    void update(double t_now)               { tt_.update(t_now); }
    PS::F64vec accel(const PS::F64vec &pos) const { return tt_.applyTensor(pos); }
    bool isLoaded() const { return tt_.isLoaded(); }
};