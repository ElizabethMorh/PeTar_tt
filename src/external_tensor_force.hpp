#pragma once
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>

// PeTar uses PS::F64vec for vectors
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

class TidalTensorManager {
public:
    struct TensorSnapshot {
        double time;    // time in PeTar units (after scaling)
        double T[3][3]; // full 3x3 tensor already scaled to PeTar units
    };

private:
    std::vector<TensorSnapshot> snapshots;
    double last_query_time = 0.0;
    TensorSnapshot interp; // last interpolated tensor
    bool loaded = false;
    PS::F64vec ref_point{0.0,0.0,0.0};

public:
    TidalTensorManager() {
        interp.time = 0.0;
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) interp.T[i][j]=0.0;
    }

    bool isLoaded() const { return loaded; }

    // loadFromFile: read nbtt, ttunit, ttoffset and NBODY6-style table
    // TSTAR: PeTar time unit (same units as PeTar's internal time).
    bool loadFromFile(const std::string &filename, double TSTAR) {
        snapshots.clear();
        FILE *fp = std::fopen(filename.c_str(), "r");
        if (!fp) {
            std::cerr << "[TT] ERROR: cannot open '" << filename << "'\n";
            return false;
        }

        // Read header: NBTT TTUNIT TTOFFSET
        int NBTT = 0;
        double TTUNIT = 0.0, TTOFFSET = 0.0;
        if (std::fscanf(fp, "%d %lf %lf", &NBTT, &TTUNIT, &TTOFFSET) != 3) {
            std::cerr << "[TT] ERROR: failed to read header (NBTT TTUNIT TTOFFSET)\n";
            std::fclose(fp);
            return false;
        }

        if (NBTT <= 0) {
            std::cerr << "[TT] ERROR: NBTT <= 0 in tt.dat\n";
            std::fclose(fp);
            return false;
        }

        // pre-reserve
        snapshots.reserve(std::max(1, NBTT));

        for (int k=0; k<NBTT; ++k) {
            // We'll attempt to read a line with either 10 columns (time + 9) or 7 columns (time + 6 symmetric)
            // read as a whole line and parse with sscanf to be robust against different whitespace
            char linebuf[1024];
            if (!std::fgets(linebuf, sizeof(linebuf), fp)) break; // EOF or error

            // skip empty or comment lines
            char *ptr = linebuf;
            while(*ptr && isspace(*ptr)) ++ptr;
            if (*ptr == '\0' || *ptr == '#') { --k; continue; } // ignore and retry same k

            double t;
            double vals[9];
            int nread = std::sscanf(linebuf,
                                    "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                                    &t,
                                    &vals[0], &vals[1], &vals[2],
                                    &vals[3], &vals[4], &vals[5],
                                    &vals[6], &vals[7], &vals[8]);

            if (nread == 10) {
                TensorSnapshot s;
                // scale time: TTTIME = t * (TTUNIT/TSTAR) + TTOFFSET/TSTAR  (NBODY6 logic)
                s.time = t * (TTUNIT / TSTAR) + (TTOFFSET / TSTAR);
                // scale tensor: TTENS *= (TSTAR / TTUNIT)^2
                double fac = (TSTAR / TTUNIT) * (TSTAR / TTUNIT);
                s.T[0][0] = vals[0] * fac; s.T[0][1] = vals[1] * fac; s.T[0][2] = vals[2] * fac;
                s.T[1][0] = vals[3] * fac; s.T[1][1] = vals[4] * fac; s.T[1][2] = vals[5] * fac;
                s.T[2][0] = vals[6] * fac; s.T[2][1] = vals[7] * fac; s.T[2][2] = vals[8] * fac;
                snapshots.push_back(s);
            } else {
                // try 7-value symmetric format: time Txx Txy Txz Tyy Tyz Tzz
                double t2, a,b,c,d,e,f;
                int n7 = std::sscanf(linebuf, "%lf %lf %lf %lf %lf %lf %lf", &t2, &a,&b,&c,&d,&e,&f);
                if (n7==7) {
                    TensorSnapshot s;
                    s.time = t2 * (TTUNIT / TSTAR) + (TTOFFSET / TSTAR);
                    double fac = (TSTAR / TTUNIT) * (TSTAR / TTUNIT);
                    // interpret as: t, Txx, Txy, Txz, Tyy, Tyz, Tzz (common symmetric 6 values)
                    s.T[0][0] = a * fac; s.T[0][1] = b * fac; s.T[0][2] = c * fac;
                    s.T[1][0] = b * fac; s.T[1][1] = d * fac; s.T[1][2] = e * fac;
                    s.T[2][0] = c * fac; s.T[2][1] = e * fac; s.T[2][2] = f * fac;
                    snapshots.push_back(s);
                } else {
                    std::cerr << "[TT] WARNING: unrecognized line format in tt.dat, skipping line:\n  '"
                              << linebuf << "'\n";
                    --k; // don't count this line
                    continue;
                }
            }
        }

        std::fclose(fp);

        if (snapshots.empty()) {
            std::cerr << "[TT] ERROR: no valid tensor snapshots loaded\n";
            return false;
        }

        std::sort(snapshots.begin(), snapshots.end(), [](const TensorSnapshot &a, const TensorSnapshot &b){ return a.time < b.time; });

        // initialise interp to first snapshot
        interp = snapshots.front();
        last_query_time = interp.time;
        loaded = true;

        std::cerr << "[TT] Loaded " << snapshots.size() << " snapshots; time range: "
                  << snapshots.front().time << " -> " << snapshots.back().time << " (PeTar time units)\n";
        return true;
    }

    // Set the reference point r0(t). Default is origin.
    void setReference(const PS::F64vec &r0) {
        ref_point = r0;
    }

    // Update internal interpolated tensor for the requested time.
    // This does linear interpolation and clamps outside the range.
    void update(double time) {
        if (!loaded) return;
        // quick path: same time
        if (time == last_query_time) return;

        // before first
        if (time <= snapshots.front().time) {
            interp = snapshots.front(); interp.time = time; last_query_time = time; return;
        }
        // after last
        if (time >= snapshots.back().time) {
            interp = snapshots.back(); interp.time = time; last_query_time = time; return;
        }

        // find interval
        auto it = std::upper_bound(snapshots.begin(), snapshots.end(), time,
                                   [](double t, const TensorSnapshot &s){ return t < s.time; });
        // it points to first snapshot with time > t, so prev is <= t
        auto it_hi = it;
        auto it_lo = it - 1;
        double t_lo = it_lo->time;
        double t_hi = it_hi->time;
        double alpha = (time - t_lo) / (t_hi - t_lo);

        interp.time = time;
        for (int i=0;i<3;i++){
            for (int j=0;j<3;j++){
                interp.T[i][j] = (1.0 - alpha) * it_lo->T[i][j] + alpha * it_hi->T[i][j];
            }
        }
        last_query_time = time;
    }

    // Apply interpolated tensor to a position (pos - ref_point)
    PS::F64vec applyTensor(const PS::F64vec &pos) const {
        PS::F64vec rel = pos - ref_point;
        PS::F64vec a;
        a.x = interp.T[0][0]*rel.x + interp.T[0][1]*rel.y + interp.T[0][2]*rel.z;
        a.y = interp.T[1][0]*rel.x + interp.T[1][1]*rel.y + interp.T[1][2]*rel.z;
        a.z = interp.T[2][0]*rel.x + interp.T[2][1]*rel.y + interp.T[2][2]*rel.z;
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

    // Utility: get time range
    bool getTimeRange(double &tmin, double &tmax) const {
        if (!loaded) return false;
        tmin = snapshots.front().time; tmax = snapshots.back().time; return true;
    }

    // number of snapshots
    size_t size() const { return snapshots.size(); }
};

