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
        for (int i=0; i<3; i++) 
            for (int j=0; j<3; j++) 
                interp_.T[i][j] = 0.0;
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
          rscale(store, 1.0, "tt-rscale", "Length scale from IN to TT"),
          vscale(store, 1.0, "tt-vscale", "Velocity scale from IN to TT") {}
};

// High-level manager used by PeTar (like GalpyManager)
class ExternalTensorManager {
private:
    // Similar to Galpy's labelCheck
    template <class Tstream>
    void labelCheck(Tstream& fconf, const char* match) {
        std::string label;
        fconf>>label;
        if (label!=match) {
            std::cerr << "ExternalTensor config: reading label error, should be " 
                      << match << " given " << label << std::endl;
            abort();
        }
    }

    void eofCheck(std::ifstream& fconf, const char* message) {
        if (fconf.eof()) {
            std::cerr << "ExternalTensor config: reading " << message 
                     << " fails! File reaches EOF." << std::endl;
            abort();
        }
    }

    //! resize array by insert or erase elements for given offset index
    /*!
      @param[in] n_diff: size difference after update
      @param[in] index: index of array offset for change
      @param[in,out] array: array of data
      @param[in,out] array_offset: offset of differert data groups 
     */
    template <class ttype>
    void resizeArray(const int n_diff, const int index, std::vector<ttype>& array, std::vector<int>& array_offset) {
        int offset = array_offset[index];
        if (n_diff!=0) {
            if (n_diff>0) {
                std::vector<ttype> data(n_diff,ttype());
                array.insert(array.begin()+offset, data.begin(), data.end());
            }
            else 
                array.erase(array.begin()+offset, array.begin()+offset-n_diff);
            for (size_t i=index+1; i<array_offset.size(); i++) 
                array_offset[i] += n_diff;
        }
    }

    //! erase array for one offset index
    /*!
      @param[in] index: index of array offset for remove
      @param[in,out] array: array of data
      @param[in,out] array_offset: offset of differert data groups 
    */
    template <class ttype>
    void eraseArray(const int index, std::vector<ttype>& array, std::vector<int>& array_offset) {
        int n = array_offset[index+1] - array_offset[index];
        int offset = array_offset[index];
        if (n>0) {
            array.erase(array.begin()+offset, array.begin()+offset+n);
        }
        for (size_t i=index+1; i<array_offset.size(); i++) 
            array_offset[i] = array_offset[i+1]-n;
        array_offset.pop_back();
        assert(array_offset.back()==int(array.size()));
    }
public:
    // for MPI communication, data IO and initialization
    std::vector<int> pot_type_offset; //  set offset
    std::vector<int> pot_type;    // types for each set
    std::vector<int> pot_args_offset; // arguments of pot for each set
    std::vector<double> pot_args;    // set offset
    double time;   // current time, update in evolveChangingArguments
    std::vector<ChangeArgument> change_args;  // changing argument index to evolve
    std::vector<int> change_args_offset; // set offset
    // tt potential arguments
    std::vector<PotentialSetPar> pot_set_pars; // potential parameters for each set
    std::vector<PotentialSet> pot_sets; // potential arguments for each set
    double update_time;
    // unit scaling
    double rscale;
    double vscale;
    double tscale;
    double fscale;
    double pscale;
    double gmscale;

    ExternalTensorManager(): pot_type_offset(), pot_type(), 
                    pot_args_offset(), pot_args(), change_args(), change_args_offset(),
                    pot_set_pars(), pot_sets(), update_time(0.0), rscale(1.0), vscale(1.0), tscale(1.0), fscale(1.0), pscale(1.0), gmscale(1.0), fconf(), set_name(), set_parfile() {}

    //! print current potential data
    void printData(std::ostream& fout) {
        fout<<"tt parameters, time: "<<time;
        fout<<" Next update time: "<<update_time;
        fout<<std::endl;
        int nset = pot_set_pars.size();
        for (int k=0; k<nset; k++) {
            auto& pot_set_par_k = pot_set_pars[k];
            fout<<"Potential set "<<k+1<<" Mode: "<<pot_set_par_k.mode
                <<" GM: "<<pot_set_par_k.gm
                <<" Pos: "<<pot_set_par_k.pos[0]<<" "<<pot_set_par_k.pos[1]<<" "<<pot_set_par_k.pos[2]
                <<" Vel: "<<pot_set_par_k.vel[0]<<" "<<pot_set_par_k.vel[1]<<" "<<pot_set_par_k.vel[2]
                <<" Acc: "<<pot_set_par_k.acc[0]<<" "<<pot_set_par_k.acc[1]<<" "<<pot_set_par_k.acc[2]
                <<"\nPotential type indice: ";
            for (int i=pot_type_offset[k]; i<pot_type_offset[k+1]; i++) 
                fout<<pot_type[i]<<" ";
            fout<<"\nPotential arguments: ";
            for (int i=pot_args_offset[k]; i<pot_args_offset[k+1]; i++) 
                fout<<pot_args[i]<<" ";
            fout<<"\nChange argument [index mode rate]:";
            for (int i=change_args_offset[k]; i<change_args_offset[k+1]; i++) {
                fout<<"["<<change_args[i].index
                    <<" "<<change_args[i].mode
                    <<" "<<change_args[i].rate<<"] ";
            }
            fout<<std::endl;
        }
    }        

#ifdef PARTICLE_SIMULATOR_MPI_PARALLEL        
    void broadcastDataMPI() {

        int nset;
        int my_rank = PS::Comm::getRank();
        if (my_rank==0) nset = pot_set_pars.size();
        PS::Comm::broadcast(&nset, 1, 0);

        // update potentials
        if (nset>=0) {

            PS::Comm::broadcast(&update_time, 1, 0);
            if (my_rank==0) {
                assert((int)pot_type_offset.size()==nset+1);
                assert((int)pot_args_offset.size()==nset+1);
                assert((int)change_args_offset.size()==nset+1);
            }
            else {
                pot_set_pars.resize(nset);
                pot_type_offset.resize(nset+1);
                pot_args_offset.resize(nset+1);
                change_args_offset.resize(nset+1);
            }
            PS::Comm::broadcast(pot_set_pars.data(), pot_set_pars.size(), 0);
            PS::Comm::broadcast(pot_type_offset.data(), pot_type_offset.size(), 0);
            PS::Comm::broadcast(pot_args_offset.data(), pot_args_offset.size(), 0);
            PS::Comm::broadcast(change_args_offset.data(), change_args_offset.size(), 0);
            
            if (my_rank==0) {
                assert((int)pot_type.size()==pot_type_offset.back());
                assert((int)pot_args.size()==pot_args_offset.back());
                assert((int)change_args.size()==change_args_offset.back());
            }
            else {
                pot_type.resize(pot_type_offset.back());
                pot_args.resize(pot_args_offset.back());
                change_args.resize(change_args_offset.back());
            }
            PS::Comm::broadcast(pot_type.data(), pot_type.size(), 0);
            if (pot_args.size()>0) PS::Comm::broadcast(pot_args.data(), pot_args.size(), 0);
            if (change_args.size()>0) PS::Comm::broadcast(change_args.data(), change_args.size(), 0);
        }
    }

    //! calculate acceleration and potential at give position
    /*!
      @param[out] acc: [3] acceleration to return
      @param[out] pot: potential to return 
      @param[in] _time: time in input unit
      @param[in] gm: G*mass of particles [input unit]
      @param[in] pos_g: position of particles in the galactic frame [input unit]
      @param[in] pos_l: position of particles in the particle system frame [input unit]
     */
    void calcAccPot(double* acc, double& pot, const double _time, const double gm, const double* pos_g, const double* pos_l) {
        assert(pot_sets.size()==pot_set_pars.size());
        int nset = pot_sets.size();
        if (nset>0) {
            double t = _time*tscale;

            // galactic frame and rest frame of particle system
            double x[2] = {pos_g[0]*rscale, pos_l[0]*rscale};
            double y[2] = {pos_g[1]*rscale, pos_l[1]*rscale};
            double z[2] = {pos_g[2]*rscale, pos_l[2]*rscale};

            pot = 0;
            acc[0] = acc[1] = acc[2] = 0.0;


            for (int k=0; k<nset; k++) {
                int mode_k = pot_set_pars[k].mode;
                assert(mode_k>=0||mode_k<=2);
                int npot = pot_sets[k].npot;
                double* pos_k = pot_set_pars[k].pos;
                int i = (mode_k & 1); // get first bit to select frame (0: galactic; 1: rest)
                // frame is consistent 
                double dx = x[i]-pos_k[0];
                double dy = y[i]-pos_k[1];
                double dz = z[i]-pos_k[2];
                double rxy= std::sqrt(dx*dx+dy*dy);
                double phi= std::atan2(dy, dx);
                double sinphi = dy/rxy;
                double cosphi = dx/rxy;

                auto& pot_args = pot_sets[k].arguments;
                double acc_rxy = calcRforce(rxy, dz, phi, t, npot, pot_args);
                double acc_z   = calczforce(rxy, dz, phi, t, npot, pot_args);
                double pot_i = evaluatePotentials(rxy, dz, npot, pot_args);
                double gm_pot = pot_set_pars[k].gm;
                if (rxy>0.0) {
                    assert(!std::isinf(acc_rxy));
                    assert(!std::isnan(acc_rxy));
                    assert(!std::isinf(acc_phi));
                    assert(!std::isnan(acc_phi));
                    assert(!std::isinf(pot));
                    assert(!std::isnan(pot));
                    pot += pot_i;
                    double acc_x = (cosphi*acc_rxy - sinphi*acc_phi/rxy);
                    double acc_y = (sinphi*acc_rxy + cosphi*acc_phi/rxy);
                    acc[0] += acc_x;
                    acc[1] += acc_y;
                    acc[2] += acc_z;
                    if (mode_k==2) {
                        double* acc_pot = pot_set_pars[k].acc;
                        acc_pot[0] -= gm*acc_x/gm_pot; // anti-acceleration to potential set origin
                        acc_pot[1] -= gm*acc_y/gm_pot;
                        acc_pot[2] -= gm*acc_z/gm_pot;
                    }
                }
            }
            pot /= pscale;
            acc[0] /= fscale;
            acc[1] /= fscale;
            acc[2] /= fscale;

            //pot = acc[0]*x+acc[1]*y+acc[2]*z;
        }
        else {
            acc[0] = acc[1] = acc[2] = pot = 0.0;
        }
    }
};