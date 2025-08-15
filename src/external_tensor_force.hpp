#pragma once
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include "particle_simulator.hpp"

class TidalTensorManager {
public:
    struct TensorSnapshot {
        double time;   // time in PeTar units
        double T[3][3]; // tidal tensor components
    };

private:
    std::vector<TensorSnapshot> snapshots;
    double current_time;
    double TSTAR; // PeTar's internal time unit

public:
    TidalTensorManager() : current_time(0.0), TSTAR(1.0) {}

    void loadFromFile(const std::string &filename, double tstar) {
        TSTAR = tstar;
        snapshots.clear();

        FILE *fp = fopen(filename.c_str(), "r");
        if (!fp) {
            fprintf(stderr, "Error: Cannot open tidal tensor file %s\n", filename.c_str());
            exit(1);
        }

        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            // Skip comments and blank lines
            if (line[0] == '#' || std::isspace(line[0])) continue;

            double t, Txx, Txy, Txz, Tyy, Tyz, Tzz;
            if (sscanf(line, "%lf %lf %lf %lf %lf %lf %lf",
                       &t, &Txx, &Txy, &Txz, &Tyy, &Tyz, &Tzz) == 7) {
                TensorSnapshot snap;
                snap.time = t / TSTAR; // convert to PeTar units
                snap.T[0][0] = Txx; snap.T[0][1] = Txy; snap.T[0][2] = Txz;
                snap.T[1][0] = Txy; snap.T[1][1] = Tyy; snap.T[1][2] = Tyz;
                snap.T[2][0] = Txz; snap.T[2][1] = Tyz; snap.T[2][2] = Tzz;
                snapshots.push_back(snap);
            }
        }
        fclose(fp);

        if (snapshots.empty()) {
            fprintf(stderr, "Error: tidal tensor file is empty!\n");
            exit(1);
        }

        fprintf(stdout, "[TidalTensorManager] Loaded %zu snapshots from %s\n",
                snapshots.size(), filename.c_str());
    }

    void update(double sim_time) {
        current_time = sim_time;
    }

    PS::F64vec applyTensor(const PS::F64vec &pos) const {
        if (snapshots.empty()) return PS::F64vec(0.0);

        // If before first snapshot
        if (current_time <= snapshots.front().time)
            return multiplyTensor(snapshots.front(), pos);

        // If after last snapshot
        if (current_time >= snapshots.back().time)
            return multiplyTensor(snapshots.back(), pos);

        // Find surrounding snapshots
        for (size_t i = 0; i < snapshots.size() - 1; ++i) {
            if (current_time >= snapshots[i].time &&
                current_time <= snapshots[i+1].time) {

                const TensorSnapshot &s1 = snapshots[i];
                const TensorSnapshot &s2 = snapshots[i+1];

                double dt = s2.time - s1.time;
                double alpha = (current_time - s1.time) / dt;

                // Interpolate each tensor component
                TensorSnapshot interp;
                for (int r = 0; r < 3; r++) {
                    for (int c = 0; c < 3; c++) {
                        interp.T[r][c] = (1.0 - alpha) * s1.T[r][c] + alpha * s2.T[r][c];
                    }
                }

                return multiplyTensor(interp, pos);
            }
        }

        // Should not reach here
        return PS::F64vec(0.0);
    }

private:
    PS::F64vec multiplyTensor(const TensorSnapshot &snap, const PS::F64vec &pos) const {
        double ax = snap.T[0][0]*pos.x + snap.T[0][1]*pos.y + snap.T[0][2]*pos.z;
        double ay = snap.T[1][0]*pos.x + snap.T[1][1]*pos.y + snap.T[1][2]*pos.z;
        double az = snap.T[2][0]*pos.x + snap.T[2][1]*pos.y + snap.T[2][2]*pos.z;
        return PS::F64vec(ax, ay, az);
    }
};

