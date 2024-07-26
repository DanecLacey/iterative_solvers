#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <sys/time.h>

class Stopwatch
{

long double wtime{};

public:
    timeval *begin;
    timeval *end;
    Stopwatch(timeval *_begin, timeval *_end) : begin(_begin), end(_end) {};
    Stopwatch() : begin(), end() {};

    void start_stopwatch(void)
    {
        gettimeofday(begin, 0);
    }

    void end_stopwatch(void)
    {
        gettimeofday(end, 0);
        long seconds = end->tv_sec - begin->tv_sec;
        long microseconds = end->tv_usec - begin->tv_usec;
        wtime += seconds + microseconds*1e-6;
        // std::cout << "begin->tv_sec = " << begin->tv_sec << std::endl;
        // std::cout << "end->tv_sec = " << end->tv_sec << std::endl;
        // std::cout << "begin->tv_usec = " << begin->tv_usec << std::endl;
        // std::cout << "end->tv_usec = " << end->tv_usec << std::endl;
    }

    long double get_wtime(
    ){
        return wtime;
    }
};

struct Timers
{
    Stopwatch *total_wtime;
    Stopwatch *solver_harness_wtime;
    Stopwatch *solver_wtime;
    Stopwatch *preprocessing_wtime;

    Stopwatch *gmres_spmv_wtime;
    Stopwatch *gmres_orthog_wtime;
    Stopwatch *gmres_mgs_wtime;
    Stopwatch *gmres_mgs_dot_wtime;
    Stopwatch *gmres_mgs_sub_wtime;
    Stopwatch *gmres_leastsq_wtime;
    Stopwatch *gmres_compute_H_tmp_wtime;
    Stopwatch *gmres_compute_Q_wtime;
    Stopwatch *gmres_compute_R_wtime;
    Stopwatch *gmres_get_x_wtime;
};

#endif