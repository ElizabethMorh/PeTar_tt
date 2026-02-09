#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <algorithm>

/****************************************************
 * IO parameters
 ****************************************************/
class IOParamsTT {
public:
    IOParamsContainer input_par_store;
    IOParams<std::string> tt_filename;
    IOParams<double> rscale;
    IOParams<double> vscale;

    bool print_flag;

    IOParamsTT():
        input_par_store(),
        tt_filename(input_par_store, "__NONE__", "tt-file",
                    "External tidal tensor file (tt.dat)"),
        rscale(input_par_store, 1.0, "tt-rscale",
               "Length scale: IN → TT"),
        vscale(input_par_store, 1.0, "tt-vscale",
               "Velocity scale: IN → TT"),
        print_flag(false) {}

    int read(int argc, char *argv[], const int opt_used_pre=0) {
        static int tt_flag = -1;
        const struct option long_options[] = {
            {tt_filename.key, required_argument, &tt_flag, 1},
            {rscale.key,      required_argument, &tt_flag, 2},
            {vscale.key,      required_argument, &tt_flag, 3},
            {"help", no_argument, 0, 'h'},
            {0,0,0,0}
        };

        int opt_used = opt_used_pre;
        int copt, option_index;
        optind = 0;

        while ((copt = getopt_long(argc, argv, "-p:h",
                                   long_options, &option_index)) != -1) {
            switch (copt) {
            case 0:
                if (tt_flag == 1) {
                    tt_filename.value = optarg;
                    opt_used += 2;
                }
                else if (tt_flag == 2) {
                    rscale.value = atof(optarg);
                    opt_used += 2;
                }
                else if (tt_flag == 3) {
                    vscale.value = atof(optarg);
                    opt_used += 2;
                }
                break;

            case 'p': {
                std::string fname = optarg;
                std::string ftt = fname + ".tt";
                FILE* fpar = fopen(ftt.c_str(), "r");
                if (!fpar) {
                    fprintf(stderr, "Error: cannot open %s\n", ftt.c_str());
                    abort();
                }
                input_par_store.readAscii(fpar);
                fclose(fpar);
                opt_used += 2;
#ifdef PARTICLE_SIMULATOR_MPI_PARALLEL
                input_par_store.mpi_broadcast();
                PS::Comm::barrier();
#endif
                break;
            }

            case 'h':
                if (print_flag) {
                    std::cout << "TT options:\n";
                    input_par_store.printHelp(std::cout, 2, 10, 23);
                }
                return -1;
            default:
                break;
            }
        }

        return opt_used;
    }
};

/****************************************************
 * Tidal tensor external potential
 ****************************************************/
class TidalTensorManager {
public:
    struct Snapshot {
        double time;
        double T[3][3];
    };

private:
    std::vector<Snapshot> snaps_;
    double Tcur_[3][3];

    bool enabled_;
    bool print_flag_;

    size_t last_idx_;
    double last_time_;

    // unit scaling
    double rscale_, vscale_;
    double tscale_, fscale_, pscale_;

public:
    TidalTensorManager():
        enabled_(false),
        print_flag_(false),
        last_idx_(0),
        last_time_(-1.0),
        rscale_(1.0),
        vscale_(1.0),
        tscale_(1.0) {}

    /************************************************
     * Initialization
     ************************************************/
    void initial(const IOParamsTT& params,
                 const double time,
                 const std::string&,
                 const bool restart_flag,
                 const bool print_flag) {

        print_flag_ = print_flag;

        rscale_ = params.rscale.value;
        vscale_ = params.vscale.value;

        if (vscale_ <= 0.0 || rscale_ <= 0.0) {
            std::cerr << "[TT] ERROR: Invalid scale parameters - rscale=" << rscale_ 
                     << " vscale=" << vscale_ << "\n";
            enabled_ = false;
            return;
        }

        tscale_ = rscale_/vscale_;
        fscale_ = vscale_*vscale_/rscale_;
        pscale_ = vscale_*vscale_;

        if (print_flag_) {
            std::cerr << "[TT] Scale factors: rscale=" << rscale_ 
                     << " vscale=" << vscale_ << " tscale=" << tscale_
                     << " fscale=" << fscale_ << " pscale=" << pscale_ << "\n";
        }

        if (params.tt_filename.value == "__NONE__") {
            enabled_ = false;
            return;
        }

        if (!restart_flag) {
            if (!load(params.tt_filename.value, params)) {
                enabled_ = false;
                if (print_flag_)
                    std::cerr << "[TT] Disabled (tt.dat file not found)\n";
                return;
            }
        }

        enabled_ = true;
        update(time);

        if (print_flag_)
            std::cerr << "[TT] External tidal tensor enabled\n";
    }

    /************************************************
     * Read tt.dat
     ************************************************/
    bool load(const std::string& fname, const IOParamsTT& params) {
        std::ifstream fin(fname);
        if (!fin) return false;

        snaps_.clear();

        // Read header first to get proper time units
        std::ifstream fin_temp(fname);
        if (!fin_temp) return false;
        
        std::string header_line;
        while (std::getline(fin_temp, header_line)) {
            if (!header_line.empty() && header_line[0] != '#') break;
        }
        
        std::istringstream header_iss(header_line);
        std::vector<double> header_vals;
        double val;
        while (header_iss >> val) header_vals.push_back(val);
        
        // tt.dat files are in physical units (Myr for time, physical units for tensor)
        // PeTar should use these directly without N-body conversion
        double effective_tt_unit = header_vals[1];  // tt_unit in Myr (time conversion factor)
        double effective_tt_offset = header_vals[2]; // tt_offset in Myr
        
        // Time conversion: tt.dat time (dimensionless) to PeTar internal time units (Myr)
        // tt.dat times are in units where t=1 corresponds to TTUNIT Myr
        const double time_scale = effective_tt_unit;  // Convert t=1 to TTUNIT Myr
        
        // Tensor scaling: use tensor values directly for maximum flexibility
        // tt.dat values should be in appropriate units for PeTar physical system
        // No scaling applied to handle complex orbital cases (elliptical, non-circular, etc.)
        const double tensor_scale = 1.0;  // Use tensor values as provided

        if (print_flag_) {
            std::cerr << "[TT] Loading tt.dat with physical units\n";
            std::cerr << "[TT] tt_unit=" << effective_tt_unit << " Myr\n";
            std::cerr << "[TT] tt_offset=" << effective_tt_offset << " Myr\n";
            std::cerr << "[TT] Using tensor values directly (no scaling)\n";
            std::cerr << "[TT] Time scale: " << time_scale << "\n";
            std::cerr << "[TT] Tensor scale: " << tensor_scale << "\n";
        }

        std::string line;
        int line_count = 0;
        int expected_snapshots = -1;

        while (std::getline(fin, line)) {
            line_count++;
            if (line.empty() || line[0]=='#') continue;
            
            // First line should be header: n_snapshots time_unit time_offset
            if (expected_snapshots == -1) {
                std::istringstream iss(line);
                std::vector<double> values;
                double val;
                while (iss >> val) values.push_back(val);
                
                if (values.size() == 3) {
                    expected_snapshots = static_cast<int>(values[0]);
                    if (print_flag_) {
                        std::cerr << "[TT] Header: " << expected_snapshots << " snapshots, "
                                 << "time_unit=" << values[1] << " Myr, "
                                 << "time_offset=" << values[2] << " Myr\n";
                    }
                    continue;
                }
            }
            
            // Data lines should have exactly 10 values (time + 9 tensor elements)
            std::istringstream iss(line);
            std::vector<double> values;
            double val;
            while (iss >> val) values.push_back(val);
            
            if (values.size() != 10) {
                if (print_flag_) {
                    std::cerr << "[TT] Skipping line " << line_count 
                             << " (incorrect number of values: " << values.size() << ")\n";
                }
                continue;
            }

            Snapshot s;
            // Reuse the existing stringstream
            iss.clear();
            iss.seekg(0, std::ios::beg);

            double t_input;
            iss >> t_input;

            // Convert input time (dimensionless) to PeTar internal time units (Myr)
            // tt.dat times are in units where t=1 corresponds to TTUNIT Myr
            s.time = (t_input + effective_tt_offset) * time_scale;

            // Check for invalid time conversion
            if (std::isnan(s.time) || std::isinf(s.time)) {
                std::cerr << "[TT] ERROR: Invalid time conversion - t_input=" << t_input 
                         << " tscale=" << tscale_ << "\n";
                return false;
            }

            for (int i=0;i<3;i++)
                for (int j=0;j<3;j++)
                    iss >> s.T[i][j];
            
            // Apply tensor scaling to convert from normalized to physical units
            // This scaling is applied to ALL tensor components in ALL snapshots
            for (int i=0;i<3;i++)
                for (int j=0;j<3;j++)
                    s.T[i][j] *= tensor_scale;

            snaps_.push_back(s);
        }

        if (snaps_.size() < 2) return false;

        last_idx_ = 0;
        last_time_ = -1.0;
        return true;
    }

    /************************************************
     * Time interpolation
     ************************************************/
    void update(double time_in) {
        if (!enabled_) return;

        double t = time_in;
        if (t == last_time_) return;
        last_time_ = t;

        static bool first_call = true;
        if (first_call) {
            first_call = false;
            std::string logfile = "tidal_tensor_log.dat";
            writeLogHeader(logfile);
            
            // Log initial snapshot times for debugging
            std::ofstream fout("tidal_tensor_snapshots.dat");
            if (fout) {
                fout << "# Snapshot times and first tensor element for reference\n";
                for (const auto& snap : snaps_) {
                    fout << snap.time << " " << snap.T[0][0] << "\n";
                }
            }
        }

        // Handle edge cases
        if (t <= snaps_.front().time) {
            copy(snaps_.front().T);
            if (print_flag_) {
                std::cerr << "[TT] WARNING: Simulation time " << t 
                         << " is before first tidal tensor snapshot at " 
                         << snaps_.front().time << "\n";
                std::cerr << "[TT] Using first snapshot - this may cause integrator issues!\n";
            }
            logTidalTensor("tidal_tensor_log.dat", t, Tcur_);
            return;
        }
        if (t >= snaps_.back().time) {
            copy(snaps_.back().T);
            if (print_flag_) {
                std::cerr << "[TT] Using last snapshot at t=" << t << " (after last snapshot)\n";
            }
            logTidalTensor("tidal_tensor_log.dat", t, Tcur_);
            return;
        }

        // Find right interval for quadratic interpolation (need 3 points)
        while (last_idx_+1 < snaps_.size() &&
               snaps_[last_idx_+1].time < t) {
            last_idx_++;
        }
        while (last_idx_ > 0 && snaps_[last_idx_].time > t) {
            last_idx_--;
        }

        // Ensure we have 3 points for quadratic interpolation
        if (last_idx_ == 0) {
            copy(snaps_[0].T);
            logTidalTensor("tidal_tensor_log.dat", t, Tcur_);
            return;
        }
        if (last_idx_+1 >= snaps_.size()) {
            copy(snaps_.back().T);
            logTidalTensor("tidal_tensor_log.dat", t, Tcur_);
            return;
        }

        // Get three consecutive snapshots for quadratic interpolation
        const auto& s0 = snaps_[last_idx_-1];   // previous
        const auto& s1 = snaps_[last_idx_];     // current  
        const auto& s2 = snaps_[last_idx_+1];   // next
        
        // Time differences (NBODY6tt style)
        double dt21 = s1.time - s0.time;  // TTTIME(INDTT) - TTTIME(INDTT-1)
        double dt32 = s2.time - s1.time;  // TTTIME(INDTT+1) - TTTIME(INDTT)
        double dt31 = s2.time - s0.time;  // TTTIME(INDTT+1) - TTTIME(INDTT-1)
        
        // Quadratic interpolation for each tensor component
        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++) {
                double t1 = s0.T[i][j];
                double t2 = s1.T[i][j];
                double t3 = s2.T[i][j];
                
                // NBODY6tt quadratic interpolation coefficients
                double tta = (t2 - t1) / dt21;
                double ttb = ((t3 - t1) * dt21 - (t2 - t1) * dt31) / (dt31 * dt32 * dt21);
                
                // Quadratic interpolation formula
                Tcur_[i][j] = t1 + tta * (t - s0.time) + ttb * (t - s0.time) * (t - s1.time);
            }
        }
        
        // Log the interpolated tensor
        logTidalTensor("tidal_tensor_log.dat", t, Tcur_);
    }

    /************************************************
     * Acceleration + potential
     ************************************************/
    void calcAccPot(double acc[3],
                    double& pot,
                    const double time,
                    const double,
                    const double* pos,
                    const double* pos_ref) {

        if (!enabled_) {
            acc[0]=acc[1]=acc[2]=0.0;
            pot = 0.0;
            return;
        }

        update(time);

        double x[3] = {
            (pos[0] - pos_ref[0]) / rscale_,
            (pos[1] - pos_ref[1]) / rscale_,
            (pos[2] - pos_ref[2]) / rscale_
        };

        acc[0]=acc[1]=acc[2]=0.0;
        pot = 0.0;

        for (int i=0;i<3;i++)
            for (int j=0;j<3;j++) {
                acc[i] += Tcur_[i][j] * x[j];
                pot    += 0.5 * Tcur_[i][j] * x[i] * x[j];
            }

        acc[0] *= fscale_;
        acc[1] *= fscale_;
        acc[2] *= fscale_;
        pot    *= pscale_;
        
        // Safety checks to prevent numerical instability
        for (int i=0; i<3; i++) {
            if (std::isnan(acc[i]) || std::isinf(acc[i])) {
                std::cerr << "[TT] ERROR: Invalid acceleration detected - resetting to zero\n";
                acc[i] = 0.0;
            }
        }
        if (std::isnan(pot) || std::isinf(pot)) {
            std::cerr << "[TT] ERROR: Invalid potential detected - resetting to zero\n";
            pot = 0.0;
        }
    }

    void printData(std::ostream& out) const {
        if (enabled_) {
            out << "External potential: Tidal Tensor (tt.dat)\n";
            out << "  Logging tidal tensor data to: tidal_tensor_log.dat\n";
        }
    }

    //! Write tidal tensor data for restart
    /*! 
      @param[in] fname: file to save data
      @param[in] time: current simulation time
    */
    void writePotentialPars(const std::string& fname,
                          const double time) const {
        if (!enabled_) return;
        
        std::ofstream fout(fname.c_str());
        if (!fout) {
            std::cerr << "Error: Cannot open file " << fname << " for writing tidal tensor data\n";
            return;
        }
        
        fout << std::scientific << std::setprecision(15);
        fout << "# External tidal tensor restart data\n";
        fout << "# time = " << time << "\n";
        fout << "# last_time last_idx\n";
        fout << last_time_ << " " << last_idx_ << "\n";
        
        // Save current tensor state
        fout << "# Current tidal tensor (3x3 matrix):\n";
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                fout << Tcur_[i][j] << " ";
            }
            fout << "\n";
        }
        
        // Save all snapshots
        fout << "# Number of snapshots: " << snaps_.size() << "\n";
        fout << "# time T11 T12 T13 T21 T22 T23 T31 T32 T33\n";
        for (const auto& snap : snaps_) {
            fout << snap.time << " ";
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    fout << snap.T[i][j] << " ";
                }
            }
            fout << "\n";
        }
        
        if (print_flag_) {
            std::cerr << "[TT] Wrote restart data to " << fname << "\n";
        }
    }
    
    //! Read tidal tensor data from restart file
    /*! 
      @param[in] fname: file to read data from
      @return true if successful, false otherwise
    */
    bool readRestartData(const std::string& fname) {
        std::ifstream fin(fname.c_str());
        if (!fin) {
            std::cerr << "[TT] Warning: Cannot open restart file " << fname << "\n";
            return false;
        }
        
        std::string line;
        double time;
        size_t idx;
        
        // Skip comments and read last_time_ and last_idx_
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            if (!(iss >> last_time_ >> last_idx_)) {
                std::cerr << "[TT] Error reading restart data (time and index)\n";
                return false;
            }
            break;
        }
        
        // Read current tensor state
        for (int i = 0; i < 3; i++) {
            if (!std::getline(fin, line)) {
                std::cerr << "[TT] Error reading tensor data\n";
                return false;
            }
            std::istringstream iss(line);
            for (int j = 0; j < 3; j++) {
                if (!(iss >> Tcur_[i][j])) {
                    std::cerr << "[TT] Error reading tensor element [" << i << "][" << j << "]\n";
                    return false;
                }
            }
        }
        
        // Read snapshots
        size_t num_snaps = 0;
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (line.find("Number of snapshots:") != std::string::npos) {
                std::istringstream iss(line.substr(line.find(":") + 1));
                if (!(iss >> num_snaps)) {
                    std::cerr << "[TT] Error reading number of snapshots\n";
                    return false;
                }
                snaps_.reserve(num_snaps);
                continue;
            }
            
            Snapshot snap;
            std::istringstream iss(line);
            if (!(iss >> snap.time)) continue;
            
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if (!(iss >> snap.T[i][j])) {
                        std::cerr << "[TT] Error reading snapshot tensor data\n";
                        return false;
                    }
                }
            }
            snaps_.push_back(snap);
        }
        
        if (print_flag_) {
            std::cerr << "[TT] Read restart data from " << fname 
                     << " (" << snaps_.size() << " snapshots)\n";
        }
        
        return true;
    }

private:
    void copy(const double Tin[3][3]) {
        for (int i=0;i<3;i++)
            for (int j=0;j<3;j++)
                Tcur_[i][j] = Tin[i][j];
    }

    // Log tidal tensor data to a file
    void logTidalTensor(const std::string& fname, double time, const double T[3][3]) {
        std::ofstream fout(fname, std::ios::app);
        if (!fout) return;
        
        fout << std::scientific << std::setprecision(15) << time << " ";
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                fout << T[i][j] << " ";
        fout << "\n";
    }

    // Write header for the tidal tensor log file
    void writeLogHeader(const std::string& fname) {        
        std::ofstream fout(fname);
        if (!fout) return;
        
        fout << "# Tidal Tensor Log\n";
        fout << "# Time "
             << "T11 T12 T13 T21 T22 T23 T31 T32 T33\n";
    }
};