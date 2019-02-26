#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

cv::Mat boxBlur(cv::Mat img, float n1, int marker_size){
    int w = int(((double) rand() / (RAND_MAX))*(n1*marker_size - 1) + 1);
    int h = int(((double) rand() / (RAND_MAX))*(n1*marker_size - 1) + 1);
    cv::Size kernel = cv::Size(w,h);
    cv::Mat dst = cv::Mat::zeros(img.rows, img.cols, img.type());
    cv::blur(img, dst, kernel);
    return dst;
}

cv:: Mat radial_gradient(cv::Mat img, float marker_size){
    cv::Mat dst = cv::Mat::zeros(img.rows, img.cols, img.type());
    Eigen::MatrixXf centre(1,2);
    centre(0) = ((double) rand() / (RAND_MAX))*(-1*marker_size);
    centre(1) = (((double) rand() / (RAND_MAX))+1)*(marker_size);
    Eigen::MatrixXf I = Eigen::Matrix2Xf::Identity(2,2);
    float sigma = ((double) rand() / (RAND_MAX))*(marker_size/4.0f) + marker_size/4.0f;
    Eigen::MatrixXf Sigma = sigma*sigma*I;
    Eigen::MatrixXf Sigma_inv = Sigma.inverse();

    Eigen::MatrixXf pixel_index(1,2);
    for (int r = 0; r < img.rows; r++){
        for(int c = 0; c< img.cols; c++){
            pixel_index(0)  = r;
            pixel_index(1) = c;
            Eigen::MatrixXf dif = (pixel_index - centre);
            Eigen::MatrixXf dif_t = dif.transpose();
            Eigen::MatrixXf out = -0.5f*dif*Sigma_inv*dif_t;
            float ex = exp(out(0));
            for(int k = 0; k < 3; k++){
                dst.at<cv::Vec3b>(cv::Point(c,r))[k] =std::min(255, int(img.at<cv::Vec3b>(cv::Point(c,r))[k] + 50*ex));
            }
        }
    }
    return dst;
}


int main(){


    std::string filename = "images/0.png";
    cv::Mat m = cv::imread(filename);

    int marker_size = m.rows;
    cv::Mat dst = boxBlur(m, 0.05, marker_size);

    std::string filename_out = filename.substr(0, filename.size()-4) + "_"+ std::to_string(1) + ".png";
    cv::imwrite(filename_out, dst);

   dst =  radial_gradient(m, marker_size);

    std::cout<<m.at<cv::Vec3b>(cv::Point(0,0))<<"\n";

    cv::imwrite(filename_out, dst);

    return 0;
}