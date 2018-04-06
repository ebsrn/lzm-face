#include "LZM.h"

using namespace cvip;

LZM::LZM(int _nmax1, int _nmax2, int _w1, int _w2, FlagLevels _flagLevels, FlagReIm _flagReIm):zmcount1(0), zmcount2(0)
{
    w1 = _w1;
    w2 = _w2;

    nmax1 = _nmax1;
    nmax2 = _nmax2;

    fLevels = _flagLevels;
    fReIm = _flagReIm;

    activeMoments1 = momentSelector(nmax1, selectedMoms1);
    activeMoments2 = momentSelector(nmax2, selectedMoms2);

    for (int i=0; i<=nmax1; i++)
        factorials1.push_back(factorial(i));

    for (int i=0; i<=nmax2; i++)
        factorials2.push_back(factorial(i));

    // prepare the zernike filters
    _prepareVnls(nmax1,w1,factorials1,zmcount1,Vnls1,ns1);
    _prepareVnls(nmax2,w2,factorials2,zmcount2,Vnls2,ns2);

}

std::vector<cv::Mat> LZM::compute2LevelComponents(const cv::Mat& I, int do_norm)
{
    // Compute the first stage moments
    std::vector<cv::Mat> firstMoments = zernikeMoments(I, zmcount1, Vnls1, ns1, w1, selectedMoms1);
    std::vector<cv::Mat> secondMoments;

    if (fLevels == LZM_ONLY_L1 || fLevels == LZM_BOTH_L1L2) {
        for (int j=0;j<firstMoments.size();j++)
            secondMoments.push_back(firstMoments[j]);
    }

    if (fLevels == LZM_ONLY_L1)
        return secondMoments;

    // For each moment computed by the first stage, compute a new set of moments
    for (int i=0;i<firstMoments.size();i++)
    {
        // Seperate the channels
        std::vector<cv::Mat> channels;
        cv::split(firstMoments[i],channels);

        cv::Mat Re = channels[0];
        cv::Mat Im = channels[1];

        // Compute the moments for the real and imaginary parts separately
        std::vector<cv::Mat> tempMomsRe;
        std::vector<cv::Mat> tempMomsIm;

        if (fReIm == LZM_ONLY_REAL)
            tempMomsRe = zernikeMoments(Re, zmcount2, Vnls2, ns2, w2, selectedMoms2);
        else if (fReIm == LZM_ONLY_IMAGINARY)
            tempMomsIm = zernikeMoments(Im, zmcount2, Vnls2, ns2, w2, selectedMoms2);
        else if (fReIm == LZM_BOTH_REALIMAGINARY) {
            tempMomsRe = zernikeMoments(Re, zmcount2, Vnls2, ns2, w2, selectedMoms2, do_norm);
            tempMomsIm = zernikeMoments(Im, zmcount2, Vnls2, ns2, w2, selectedMoms2, do_norm);
        }
        else {
            std::cout << "Error with Real/Imaginary flag of LZM" << std::endl;
            exit(-1);
        }

        // Add the computed moments to the list of second stage moments
        for (int j=0;j<tempMomsIm.size();j++)
        {
            if (fReIm == LZM_ONLY_REAL)
                secondMoments.push_back(tempMomsRe[j]);
            else if (fReIm == LZM_ONLY_IMAGINARY)
                secondMoments.push_back(tempMomsIm[j]);
            else if (fReIm == LZM_BOTH_REALIMAGINARY) {
                secondMoments.push_back(tempMomsRe[j]);
                secondMoments.push_back(tempMomsIm[j]);
            }
        }
    }

    return secondMoments;
}

int LZM::momentSelector(int nmax, std::vector<int> &selectedMoms) {

    int m = 0;
    for (int i=0; i<=nmax; i++) {
        for (int j=0; j<=i; j++) {
            if ((i-j)%2 != 0)
                continue;

            if (j != 0) {
                selectedMoms.push_back(1);
                m++;
            }
            else
                selectedMoms.push_back(0);
        }
    }

    return m;
}

void LZM::_prepareVnls(int nmax, int w, const std::vector<double> &factorials, int& zmcount,
                       std::vector< std::vector<std::complex<double> > >& Vnls, std::vector<int>& ns)
{
    // diameter of the circle for zernike elements
    double D = (double) w*std::sqrt((double)2);
    int w2 = w*w;

    // construct a lookup table for zernike moments for efficient computation
    for (int _n=0; _n<=nmax; _n++) {
        for (int _m=0; _m<=_n; _m++) {
            if ((_n-_m) % 2 == 0) {
                ns.push_back(_n);
                zmcount++;

                // fill the lookup table
                Vnls.push_back(std::vector<std::complex<double> >(w2, 0.));
                for (int y=0; y<w; ++y) {
                    for (int x=0; x<w; ++x) {
                        double xn = (double)(2*x+1-w)/D;
                        double yn = (double)(2*y+1-w)/D;

                        for (int xm = 0; xm <= (_n-_m)/2; xm++) {

                            // theta must change bw 0,2pi
                            double theta = atan2(yn,xn);
                            if (theta<0)
                                theta = 2*PI+theta;

                            Vnls.back()[w*y+x] +=
                                    (pow(-1.0, (double)xm)) * ( factorials[_n-xm] ) /
                                    (factorials[xm] * (factorials[(_n-2*xm+_m)/2]) *
                                     (factorials[(_n-2*xm-_m)/2]))  *
                                    (pow( sqrt(xn*xn + yn*yn), (_n - 2.*xm)) ) *
                                    4./(D*D)* // Delta_x* Delta_y
                                    std::polar(1., _m*theta);
                        }
                    }
                }
            }
        }
    }
}


/**
  * @param im - image to filter
  * @param zmcount - number of zernike moments; computed previously in recognizerLZM class
  * @param vector<vector<complex>> Vnls - the zernike filter kernels
  * @param vector<int> ns: the "n" value corresponding to ith moment component
  */
std::vector<cv::Mat> LZM::zernikeMoments(const cv::Mat& im, int zmcount, const ZernikeFilters& Vnls,
                                         const std::vector<int>& ns, int w, const std::vector<int>& selectedMoms,
                                         int do_norm)
{
    double EPS = 0.0000001;

    int m = im.rows;
    int n = im.cols;


    // diameter of the circle for zernike elements
    double D = (double) w*std::sqrt((double)2);
    int step = w/2;
    int w2 = w*w;

    std::vector<cv::Mat> zernikeIm;

    //We need to register space beforehand due to the silly design of Evangelos Sariyanidi
    for (int p=0; p<zmcount; ++p) {
        if (!selectedMoms[p])
            continue;
        zernikeIm.push_back(cv::Mat(m, n, CV_64FC2));
    }

    int i;

#pragma omp parallel
{
    #pragma omp for nowait
    for (i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            double m00 = 0.0;

            // compute m00 to obtain normalized values later on
            for (int ii=i-step, y=1; ii<=i+step; ++ii, ++y) {
                for (int jj=j-step, x=1; jj<=j+step; ++jj, ++x) {
                    // check boundary conditions, do the padding here
                    int jAct = jj<0 ? -jj-1 : (jj >= n ? 2*n-jj-1 : jj);
                    int iAct = ii<0 ? -ii-1 : (ii >= m ? 2*m-ii-1 : ii);
                    m00 += im.at<double>(iAct, jAct);//*(im+m*jAct+iAct);
                }
            }

            double meanVal=m00/w2;
            double stdev=EPS;

            // compute st dev
            for (int ii=i-step, y=1; ii<=i+step; ++ii, ++y) {
                for (int jj=j-step, x=1; jj<=j+step; ++jj, ++x) {
                    // check boundary conditions, do the padding here
                    int jAct = jj<0 ? -jj-1 : (jj >= n ? 2*n-jj-1 : jj);
                    int iAct = ii<0 ? -ii-1 : (ii >= m ? 2*m-ii-1 : ii);
                    stdev += std::pow(im.at<double>(iAct, jAct)-meanVal, 2);//std::pow(im[m*jAct+iAct]-meanVal, 2);
                }
            }
            stdev=std::sqrt(stdev/w2);

            // get the actual zernike moments by "convolving" Vnm values within the lookup table
            int pcounter = 0;
            for (int p=0; p<zmcount; ++p) {
                if (!selectedMoms[p])
                    continue;

                int _n = ns[p];

                std::complex<double> sum = 0;

                for (int y=0; y<w; y++)
                {
                    for (int x=0; x<w; x++)
                    {
                        int ii = i+y-step; // indices messed up!
                        int jj = j+x-step;

                        int iAct = ii<0 ? -ii-1 : (ii>=m ? 2*m-ii-1 : ii);
                        int jAct = jj<0 ? -jj-1 : (jj>=n ? 2*n-jj-1 : jj);

                        sum += (im.at<double>(iAct, jAct)-meanVal)*conj(Vnls[p][y*w+x]);
                    }
                }

                sum *= (_n+1)/(PI*stdev);

                zernikeIm[pcounter].at<cv::Vec2d>(i, j)[0] = real(sum);
                zernikeIm[pcounter].at<cv::Vec2d>(i, j)[1] = imag(sum);

                pcounter++;
            }
        }
    }
}

    return zernikeIm;
}

double LZM::factorial(int n) {
    double result;

    for (int i=0; i<=n; i++) {
        if (i == 0)
            result = 1;
        else
            result = result * i;
    }

    return result;
}
