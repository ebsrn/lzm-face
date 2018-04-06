#ifndef LZM_H
#define LZM_H

#define _CRT_SECURE_NO_WARNINGS

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <map>
#include <iomanip>
#include <string>
#include <complex>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <omp.h>

namespace cvip
{

    typedef std::vector< std::vector<std::complex<double> > > ZernikeFilters;

    enum FlagLevels {LZM_ONLY_L1, LZM_ONLY_L2, LZM_BOTH_L1L2};
    enum FlagReIm {LZM_ONLY_REAL, LZM_ONLY_IMAGINARY, LZM_BOTH_REALIMAGINARY};
    static const double PI = 3.14159265;

    class LZM
    {
    public:
        LZM(int _nmax1, int _nmax2, int _w1, int _w2, FlagLevels _flagLevels, FlagReIm _flagReIm);
        ~LZM () {}

        std::vector<cv::Mat> compute2LevelComponents(const cv::Mat& I, int do_norm=1);
        std::vector<cv::Mat> zernikeMoments(const cv::Mat& im, int zmcount, const ZernikeFilters& Vnls,
                                            const std::vector<int>& ns, int w, const std::vector<int>& selectedMoms,
                                            int do_norm=1);
        double factorial(int n);

    public:
        int activeMoments1;
        int activeMoments2;

    private:
        int momentSelector(int nmax, std::vector<int> &selectedMoms);
        void _prepareVnls(int nmax, int w, const std::vector<double> &factorials, int& zmcount, ZernikeFilters& Vnls,
                          std::vector<int>& ns);

    protected:
        int w1;
        int w2;
        int nmax1;
        int nmax2;
        std::vector<int> selectedMoms1;
        std::vector<int> selectedMoms2;
        std::vector<double> factorials1;
        std::vector<double> factorials2;

        FlagLevels fLevels;
        FlagReIm fReIm;

        // internal vectors needed in zernike transformation
        std::vector<int> ns1; // n's: the "n" value corresponding to ith moment component
        // ns and ms of the second zernike transformation
        std::vector<int> ns2; // n's: the "n" value corresponding to ith moment component

        // number of zernike moments of the first and second transformation respectively
        int zmcount1;
        int zmcount2;

        // the zernike filters: V_k
        ZernikeFilters Vnls1;
        ZernikeFilters Vnls2;

    };

}
#endif // LZM_H
