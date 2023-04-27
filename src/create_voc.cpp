//
// Created by gcc on 4/19/23.
//
#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "alike.hpp"
#include "utils.h"



using namespace cv;
using namespace std;



int main( int argc, char** argv ) {
    // read the image
    cout<<"reading images... "<<endl;
    vector<Mat> images;
    string image_folder_path = "/mnt/data/Bovisa_2008-09-01-FRONTAL/FRONTAL/";
    vector<String> image_filenames;
    glob(image_folder_path + "*.png", image_filenames, false);

    for (size_t i = 0; i < image_filenames.size(); i++)
    {
        Mat image = imread(image_filenames[i]);
        if (image.empty())
        {
            cerr << "Failed to read image: " << image_filenames[i] << endl;
            continue;
        }
        images.push_back( image );
    }
    cout << images.size() << endl;
//    bool use_cuda = true;
    auto device = (true) ? torch::kCUDA : torch::kCPU;
    auto alike = alike::ALIKE("/home/gcc/repo/ALIKE-cpp/models/alike-n.pt", true, 2, -1,  0.2, 2000, false);



//        cv::cvtColor(image, img_rgb, cv::COLOR_GRAY2RGB);



    // detect ORB features
    cout<<"detecting ORB features ... "<<endl;
//    Ptr< Feature2D > detector = ORB::create();
    vector<Mat> descriptors_sum;
    for ( Mat& image:images )
    {

        torch::Tensor score_map, descriptor_map;
        torch::Tensor keypoints_t, dispersitys_t, kptscores_t, descriptors_t;
        std::vector<cv::KeyPoint>
                keypoints;
        cv::Mat descriptors;
        cv::Mat img_rgb;
//        Mat descriptor;
        cv::cvtColor(image, img_rgb, cv::COLOR_BGR2RGB);

        auto img_tensor = alike::mat2Tensor(image).permute({2, 0, 1}).unsqueeze(0).to(device).to(torch::kFloat) / 255;
        alike.extract(img_tensor, score_map, descriptor_map);
//        alike.detectAndCompute(score_map, descriptor_map, keypoints_t, dispersitys_t, kptscores_t, descriptors_t);
        alike.detectAndCompute(score_map, descriptor_map, keypoints_t, dispersitys_t, kptscores_t, descriptors_t);
        alike.toOpenCVFormat(keypoints_t,dispersitys_t,kptscores_t,descriptors_t,keypoints,descriptors);

//        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        descriptors_sum.push_back( descriptors );
        cout << descriptors_sum.size() <<endl;
    }
//    std::vector<Mat>().swap(images);
    // create vocabulary
    cout<<"creating vocabulary ... "<<endl;
    DBoW3::Vocabulary vocab;
    vocab.create( descriptors_sum );
    cout<<"vocabulary info: "<<vocab<<endl;
    vocab.save( "alike_vocabulary02.yml.gz" );
    cout<<"done"<<endl;

    return 0;
}