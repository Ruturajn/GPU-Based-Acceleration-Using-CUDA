#include <iostream>
#include <ios>
#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    
    // Starting the video capture from the Webcam
    VideoCapture cap(0);

    // Check whether webcam started
    if (cap.isOpened() == false){
        cout << "Error opening Webcam" << endl;
        return -1;
    }

    // Naming the window
    String win_name = "Webcam Stream";
    String win_name1 = "Edge Detect";
    String win_name2 = "Blurred";

    Size frame_size(640, 480);

    // Creating variables on the device
    cuda::GpuMat d_img, d_res_blur, d_resf;

    // Creating variables on the host
    Mat h_resf, h_res_blur;

    while(true){
        // Create frame object
        Mat frame;

        // Reading the frame
        bool flag = cap.read(frame);

        // Showing the frame 
        imshow(win_name, frame);

        // Convert frame to grayscale
        cvtColor(frame, frame, COLOR_BGR2GRAY);

        // Uplading the frame to device
        d_img.upload(frame);

        // Blurring the frame
        Ptr<cuda::Filter> filter5x5, filter3, filterd;
        filter5x5 = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(3,3), 1, 1);
        filter5x5->apply(d_img, d_res_blur);
        d_res_blur.download(h_res_blur);
        imshow(win_name2, h_res_blur);

        // Edge Detection
        filter3 = cuda::createLaplacianFilter(CV_8UC1, CV_8UC1, 3);
        filter3->apply(d_res_blur, d_resf);

        // Download result to host
        d_resf.download(h_resf);

        // Showing the detected Edges
        float x = cap.get(CAP_PROP_FPS);
        putText(h_resf, to_string(x), Point(50, 450), FONT_HERSHEY_COMPLEX, 2, Scalar(255,255,255));
        imshow(win_name1, h_resf);

        // Waiting for quit action
        if (waitKey(1) == 'q')
            break;
        
    }
    
    return 0;
}