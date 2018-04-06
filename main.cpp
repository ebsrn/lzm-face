#include <iostream>
#include <vector>
#include "LZM.h"
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace std;

cv::Mat prepareImage(const cv::Mat inputImage)
{
    cv::Mat grayImage(inputImage.rows, inputImage.cols, CV_64FC1);

    if (inputImage.channels() == 1) {
        inputImage.convertTo(grayImage, CV_64FC1);
    }
    else if (inputImage.channels() == 3) {
        cv::Mat grayImage8U;
        cv::cvtColor(inputImage, grayImage8U, CV_RGB2GRAY);
        grayImage8U.convertTo(grayImage, CV_64FC1);
    }
    else {
        std::cerr << "ERROR in number of channels!!!" << std::endl;
        exit(-1);
    }

    //normalization
    cv::Scalar mean;
    cv::Scalar std;
    cv::meanStdDev(grayImage, mean, std);

    {
        for (int reci = 0; reci < grayImage.rows; reci++) {
            for (int recj = 0; recj < grayImage.cols; recj++) {
                grayImage.at<double>(reci,recj) =
                        (grayImage.at<double>(reci,recj) - (mean.val[0] +1e-10)) /  (std.val[0]+1e-10);
            }
        }

    }

    return grayImage;
}

int main(int argc, char *argv[])
{
    //lzm
    int lzm_n1 = 4;
    int lzm_n2 = 4;
    int lzm_k1 = 5;
    int lzm_k2 = 7;
    int lzm_bin = 18;
    int lzm_cell = 10;

    cvip::LZM lzm(lzm_n1, lzm_n2, lzm_k1, lzm_k2, cvip::LZM_ONLY_L1, cvip::LZM_BOTH_REALIMAGINARY);

    //lzm output components
    cv::Mat image = cv::imread("/home/basaran/Dropbox/TEZ/kod/lzm/FaceRecognitionV2/Aaron_Eckhart_0001_c.jpg");
    image = prepareImage(image);
    vector<cv::Mat> lzmComps = lzm.compute2LevelComponents(image);

    for(int i = 0; i < lzmComps.size(); i++)
    {
        // Seperate the channels
        std::vector<cv::Mat> channels;
        cv::split(lzmComps[i],channels);
        cv::normalize(channels[0], channels[0], 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::normalize(channels[1], channels[1], 0, 255, cv::NORM_MINMAX, CV_8UC1);

        cv::imshow("RE", channels[0]);
        cv::imshow("IM", channels[1]);
        cv::waitKey(0);
    }

    return 0;
}
