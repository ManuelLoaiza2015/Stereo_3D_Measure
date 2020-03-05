// Device ID: xinput list
// Video Stream: ls -ltrh /dev/video*

#ifdef _WIN32 
	#include <direct.h>
	#define GetCurrentDir _getcwd
#elif __unix__
	#include <unistd.h>
	#define GetCurrentDir getcwd
#endif

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/ximgproc.hpp>


#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"



#define webCam 8 
#define usbCam 17

//using namespace cv;
//using namespace std;

std::string MatType(cv::Mat inputMat)
{
	int inttype = inputMat.type();

	std::string r, a;
	uchar depth = inttype & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (inttype >> CV_CN_SHIFT);
	switch (depth) {
	case CV_8U:  r = "8U";   a = "Mat.at<uchar>(y,x)"; break;
	case CV_8S:  r = "8S";   a = "Mat.at<schar>(y,x)"; break;
	case CV_16U: r = "16U";  a = "Mat.at<ushort>(y,x)"; break;
	case CV_16S: r = "16S";  a = "Mat.at<short>(y,x)"; break;
	case CV_32S: r = "32S";  a = "Mat.at<int>(y,x)"; break;
	case CV_32F: r = "32F";  a = "Mat.at<float>(y,x)"; break;
	case CV_64F: r = "64F";  a = "Mat.at<double>(y,x)"; break;
	default:     r = "User"; a = "Mat.at<UKNOWN>(y,x)"; break;
	}
	r += "C";
	r += (chans + '0');
	//std::cout << "Mat is of type " << r << " and should be accessed with " << a << std::endl;

	std::string returnStr;
	returnStr = "Mat is of type " + r + " and should be accessed with " + a;
	return returnStr;

}

std::string get_current_dir() {
	char buff[FILENAME_MAX]; //create string buffer to hold path
	GetCurrentDir(buff, FILENAME_MAX);
	std::string current_working_dir(buff);
	return current_working_dir;
}

void drawComposedImage(
	cv::Mat &matIn_L, cv::Mat &matIn_R,
	cv::Size boardSize,
	std::vector<std::vector<cv::Point2f>> &imgPtsBuff_L_global, std::vector<std::vector<cv::Point2f>> &imgPtsBuff_R_global,
	int pattern_count)
{
	double scale_factor = 0.5;
	cv::Mat matOut;
	cv::Mat matIn_L_resize, matIn_R_resize, matOut_resize;

	cv::resize(matIn_L, matIn_L_resize, cv::Size(), scale_factor, scale_factor);
	cv::resize(matIn_R, matIn_R_resize, cv::Size(), scale_factor, scale_factor);

	cv::Mat matIn_L_Pattern, matIn_R_Pattern;
	cv::Mat matIn_L_Pattern_resize, matIn_R_Pattern_resize;

	matIn_L_Pattern = cv::Mat::zeros(matIn_L.size(), CV_8UC3);
	matIn_R_Pattern = cv::Mat::zeros(matIn_R.size(), CV_8UC3);
	matOut = cv::Mat::zeros(matIn_L.size(), CV_8UC3);

	// Draw image stereo to first row of composed image
	if (matIn_L_resize.channels() == 1) // Grayscale to Color
		cvtColor(matIn_L_resize, matIn_L_resize, cv::COLOR_GRAY2BGR);

	matIn_L_resize.copyTo(
		matOut(
			//cv::Rect(0, 0, matIn_L_resize.cols, matIn_L_resize.rows)));
			cv::Rect(matIn_L_resize.cols, 0, matIn_L_resize.cols, matIn_L_resize.rows)));


	if (matIn_R_resize.channels() == 1) // Grayscale to Color
		cvtColor(matIn_R_resize, matIn_R_resize, cv::COLOR_GRAY2BGR);

	matIn_R_resize.copyTo(
		matOut(
			cv::Rect(0, 0, matIn_R_resize.cols, matIn_R_resize.rows)));
			//cv::Rect(matIn_R_resize.cols, 0, matIn_R_resize.cols, matIn_R_resize.rows)));

	// Draw pattern history to image base
	for (int i = 0; i < pattern_count; i++)
	{
		drawChessboardCorners(matIn_L_Pattern, boardSize, cv::Mat(imgPtsBuff_L_global[i]), true);
		drawChessboardCorners(matIn_R_Pattern, boardSize, cv::Mat(imgPtsBuff_R_global[i]), true);
	};
	cv::resize(matIn_L_Pattern, matIn_L_Pattern_resize, cv::Size(), scale_factor, scale_factor);
	cv::resize(matIn_R_Pattern, matIn_R_Pattern_resize, cv::Size(), scale_factor, scale_factor);
	
	// Draw image stereo of pattern history to second row of composed image
	if (matIn_L_Pattern_resize.channels() == 1) // Grayscale to Color
		cvtColor(matIn_L_Pattern_resize, matIn_L_Pattern_resize, cv::COLOR_GRAY2BGR);

	matIn_L_Pattern_resize.copyTo(
		matOut(
			cv::Rect(matIn_L_Pattern_resize.cols, matIn_L_Pattern_resize.rows, matIn_L_Pattern_resize.cols, matIn_L_Pattern_resize.rows)));


	if (matIn_R_Pattern_resize.channels() == 1) // Grayscale to Color
		cvtColor(matIn_R_Pattern_resize, matIn_R_Pattern_resize, cv::COLOR_GRAY2BGR);

	matIn_R_Pattern_resize.copyTo(
		matOut(
			cv::Rect(0, matIn_R_Pattern_resize.rows, matIn_R_Pattern_resize.cols, matIn_R_Pattern_resize.rows)));

	// Display the resulting frame
	imshow("Frame composed: ", matOut);

	return;
}

int main(int argc, char* argv[])
{
	 //Open the default video camera
	 cv::VideoCapture cam_L(0);
	 cv::VideoCapture cam_R(1);

	 // if not success, exit program
	 if (cam_L.isOpened() == false && cam_R.isOpened() == false)
	 {
	  std::cout << "Cannot open the video camera" << std::endl;
	  std::cin.get(); //wait for any key press
	  return -1;
	 } 

	 double wImg = 640/*960*/, hImg = 480/*720*/;

	 cam_L.set(cv::CAP_PROP_FRAME_WIDTH, wImg);
	 cam_L.set(cv::CAP_PROP_FRAME_HEIGHT, hImg);

	 cam_R.set(cv::CAP_PROP_FRAME_WIDTH, wImg);
	 cam_R.set(cv::CAP_PROP_FRAME_HEIGHT, hImg);

	 cam_L.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
	 cam_L.set(cv::CAP_PROP_EXPOSURE, 1000);

	 cam_R.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25); 
	 cam_R.set(cv::CAP_PROP_EXPOSURE, 1000);

	 double dWidth = cam_L.get(cv::CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	 double dHeight = cam_L.get(cv::CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	 std::cout << "Resolution of the video : " << dWidth << " x " << dHeight << std::endl;

	 cv::Mat frame_L, frame_R; 
	 cv::Mat frame_L_gray, frame_R_gray;
	 cv::Mat frame_L_draw, frame_R_draw;

	 cv::Size boardSize;

	 boardSize.width = 9;
	 boardSize.height = 6;

	 std::vector< cv::Point3f > objPtsBuff;
	 std::vector<std::vector< cv::Point3f >> objPtsBuff_global;

	 std::vector<cv::Point2f> imgPtsBuff_L, imgPtsBuff_R;
	 std::vector<std::vector<cv::Point2f>> imgPtsBuff_L_global, imgPtsBuff_R_global;

	 std::string path , image_name_L, image_name_R, image_name = "image";
	 

	 path = get_current_dir(); 
	 std::cout << "Current path is " << path << '\n';

	 float square_size = 0.0245f;//13.f;
	 for (int i = 0; i < boardSize.height; i++)
	 {
		 for (int j = 0; j < boardSize.width; j++)
		 {
			 objPtsBuff.push_back(cv::Point3f((float)j * square_size, (float)i * square_size, 0));
		 }
	 }


	 bool bSuccess = false, found_L = false, found_R = false;
	 
	 // Flags to do stereo camera calibration
	 bool flagFindPattern = false, flagCalibrateStereoSystem = false;

	 // Parameters to control pattern capture
	 int image_count = 0;

	 // Parametrs to stereo camera calibration
	 cv::Mat K_l, K_r;
	 cv::Mat D_l, D_r;
	 std::vector< cv::Mat > rvecs_l, tvecs_l, rvecs_r, tvecs_r;
	 int flag = 0;

	 cv::Mat O_K_l, O_K_r;
	 cv::Size img_L_NewSize, img_R_NewSize;
	 cv::Rect validPixROI_L, validPixROI_R;

	 // Parameters to stereo camera calibration
	 cv::Mat  R, F, E;
	 cv::Vec3d T;
	 cv::Mat RL, RR, PL, PR, Q;
	 cv::Size stereo_NewSize;
	 cv::Rect stereo_ROI_L, stereo_ROI_R;
	 cv::Mat Left_Stereo_Map_1, Left_Stereo_Map_2,
		 Right_Stereo_Map_1, Right_Stereo_Map_2;

	 // Parameters to stereo vision
	 int window_size = 3;
	 int minDisparity = 2;
	 int numDisparities = 130 - minDisparity;
	 int blockSize = window_size;
	 int P1 = 8 * 3 * window_size*window_size;
	 int P2 = 32 * 3 * window_size*window_size;
	 int disp12MaxDiff = 0;
	 int preFilterCap = 0;
	 int uniquenessRatio = 10;
	 int speckleWindowSize = 100;
	 int speckleRange = 32;
	 int mode = cv::StereoSGBM::MODE_SGBM;

	 cv::Ptr<cv::StereoSGBM> stereo_L;
	 cv::Ptr<cv::StereoMatcher> stereo_R;

	 // WLS FILTER Parameters
	 double lmbda = 80000;
	 double sigma = 1.8;
	 double visual_multiplier = 1.0;

	 cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;

	 // Flags to do stereo visualization
	 bool flagStereoVisualization = false;
	 cv::Mat Left_nice, Right_nice;

	 cv::Mat dispL, dispR;
	 cv::Mat filteredImg, filteredImgNormalize;
	 cv::Mat disp;

	 while (true)
	 {
		// read a new frame from stereo video camera setup
		bool bSuccess = cam_L.read(frame_L) && cam_R.read(frame_R);

		//Breaking the while loop if the frames cannot be captured
		if (bSuccess == false)
		{
			std::cout << "Video camera is disconnected" << std::endl;
			std::cin.get(); //Wait for any key press
			break;
		}

		frame_L.copyTo(frame_L_draw);
		frame_R.copyTo(frame_R_draw);

		if (flagFindPattern)
		{
			// convert frames from color to grayscale to search a pattern
			cvtColor(frame_L, frame_L_gray, cv::COLOR_BGR2GRAY);
			cvtColor(frame_R, frame_R_gray, cv::COLOR_BGR2GRAY);

			found_L = findChessboardCorners(frame_L_gray, boardSize, imgPtsBuff_L,
				cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

			found_R = findChessboardCorners(frame_R_gray, boardSize, imgPtsBuff_R,
				cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

			if (found_L)
			{
				cornerSubPix(frame_L_gray, imgPtsBuff_L, cv::Size(10, 10),
					cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.0001));
			}

			if (found_R)
			{
				cornerSubPix(frame_R_gray, imgPtsBuff_R, cv::Size(10, 10),
					cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.0001));
			}

			if (found_L && found_R)
			{
				drawChessboardCorners(frame_L_draw, boardSize, cv::Mat(imgPtsBuff_L), found_L);
				drawChessboardCorners(frame_R_draw, boardSize, cv::Mat(imgPtsBuff_R), found_R);
			}

		}

		if (flagCalibrateStereoSystem)
		{
			std::cout << "Starting Calibration\n";

			// Set flag to individual camera calibration
			flag = 0;
			flag |= cv::CALIB_FIX_K4;
			flag |= cv::CALIB_FIX_K5;

			// Calibrate cameras individually
			std::cout << "\nCalibration reprojection error Left: " << calibrateCamera(objPtsBuff_global, imgPtsBuff_L_global, frame_L.size(), K_l, D_l, rvecs_l, tvecs_l, flag);
			std::cout << "\nCalibration reprojection error Right: " << calibrateCamera(objPtsBuff_global, imgPtsBuff_R_global, frame_R.size(), K_r, D_r, rvecs_r, tvecs_r, flag);
			
			O_K_l = cv::getOptimalNewCameraMatrix(K_l, D_l, frame_L.size(), 1.0, img_L_NewSize, &validPixROI_L);
			O_K_r = cv::getOptimalNewCameraMatrix(K_r, D_r, frame_L.size(), 1.0, img_R_NewSize, &validPixROI_R);

			std::cout << "\n\n K_l: " << K_l << "\n\n Opt_K_l: " << O_K_l << "\n\n D_l: " << D_l;
			std::cout << "\n\n K_r: " << K_r << "\n\n Opt_K_r: " << O_K_r << "\n\n D_r: " << D_r;

			// End camera calibration individually
			flag = 0;
			flag |= cv::CALIB_FIX_INTRINSIC;
			std::cout << "\nStereo calibration error: " <<
				cv::stereoCalibrate(objPtsBuff_global, imgPtsBuff_L_global, imgPtsBuff_R_global, K_l, D_l, K_r, D_r, frame_L.size(), R, T, E, F, flag,
					cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.0001));

			// StereoRectify
			cv::stereoRectify(K_l, D_l, K_r, D_r, frame_L.size(), R, T, RL, RR, PL, PR, Q, 0, 0, stereo_NewSize, &stereo_ROI_L, &stereo_ROI_R);
			
			// Undistorted images
			cv::initUndistortRectifyMap(
				K_l, D_l, RL, PL, frame_L.size(), CV_16SC2, Left_Stereo_Map_1, Left_Stereo_Map_2);

			cv::initUndistortRectifyMap(
				K_r, D_r, RR, PR, frame_R.size(), CV_16SC2, Right_Stereo_Map_1, Right_Stereo_Map_2);

			// Create stereo correspondence
			stereo_L = cv::StereoSGBM::create(
				minDisparity, numDisparities, blockSize , P1,P2, disp12MaxDiff,preFilterCap,uniquenessRatio,speckleWindowSize,speckleRange, mode );
			
			// Used to filter the image
			stereo_R = cv::ximgproc::createRightMatcher(stereo_L);

			wls_filter = cv::ximgproc::createDisparityWLSFilter(stereo_L);
			wls_filter->setLambda(lmbda);
			wls_filter->setSigmaColor(sigma);
			  
			std::cout << "\nEnd stereo calibration process ...";
			image_count = 0;
			objPtsBuff_global.clear();
			imgPtsBuff_L_global.clear();
			imgPtsBuff_R_global.clear();
			flagCalibrateStereoSystem = false;
			flagFindPattern = false;

		}
		
		if (flagStereoVisualization)
		{
			// undistort image
			cv::remap(frame_L, Left_nice, Left_Stereo_Map_1, Left_Stereo_Map_2, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
			cv::remap(frame_R, Right_nice, Right_Stereo_Map_1, Right_Stereo_Map_2, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);

			// convert frames from color to grayscale to search a pattern
			cv::cvtColor(Left_nice, frame_L_gray, cv::COLOR_BGR2GRAY);
			cv::cvtColor(Right_nice, frame_R_gray, cv::COLOR_BGR2GRAY);

			// Compute the 2 images for the Depth_image
			stereo_L->compute(frame_L_gray, frame_R_gray, dispL);
			stereo_R->compute(frame_R_gray, frame_L_gray, dispR);

			// Using the WLS filter
			cv::Rect ROI_disp;
			wls_filter->filter(dispL, frame_L_gray, filteredImg, dispR, ROI_disp, frame_R_gray);

			double minVal, maxVal;

			cv::normalize(filteredImg, filteredImgNormalize, 255.0, 0.0, cv::NORM_MINMAX, CV_8UC1);
			
			dispL.convertTo( disp, CV_32FC1);
			disp = ((disp / 16)-1) / 128;
			//disp = ((disp / 16) - minDisparity) / numDisparities;

			//	# Filtering the Results with a closing filter
			//	closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

			//	# Colors map
			//	dispc= (closing-closing.min())*255
			//	dispC= dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
			//	disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
			//	filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN) 

			//	# Show the result for the Depth_image
			//	#cv2.imshow('Disparity', disp)
			//	#cv2.imshow('Closing',closing)
			//	#cv2.imshow('Color Depth',disp_Color)
			//	cv2.imshow('Filtered Color Depth',filt_Color)

			// Filtering the Results with a closing filter
			// kernel of 3x3
			int morph_elem = 0;
			int morph_size = 1;
			int morph_iteration = 1;
			cv::Mat element = getStructuringElement(morph_elem,
				cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
				cv::Point(morph_size, morph_size));

			cv::Mat closing;
			morphologyEx(disp, closing, cv::MORPH_CLOSE, element,
				cv::Point(-1, -1),
				morph_iteration);

			imshow("Disparity", closing);

			cv::minMaxLoc(closing, &minVal, &maxVal);
			std::cout << "\n Closing max = " << maxVal << "\t min = " << minVal;

			// Get the in value of closing 
			cv::Mat dispc;
			cv::Mat minClosing;
						
			minClosing = cv::Mat(closing.size(), closing.type(), minVal);
			dispc = (closing - minClosing) * 255;

			
			
			//double minVal, maxVal;
			cv::minMaxLoc(dispc, &minVal, &maxVal);

			std::cout << "\n Dispc max = " << maxVal << "\t min = " << minVal;

			//cv::cvtColor(filteredImgNormalize_8U, filteredImgNormalize_8U, cv::COLOR_BGR2GRAY);

			//imshow("stereo_L", filteredImgNormalize);
			//imshow("stereo_R", filteredImgNormalize_8U);

			std::cout << "\nType of  disparity matrix filteredImg:  " << MatType(filteredImg);
			std::cout << "\nType of  disparity matrix filteredImgNormalize:  " << MatType(filteredImgNormalize);

			//std::cout << "\nType of  disparity matrix:  "<< MatType(dispL)<<" - "<< MatType(dispR);
			//cv::Mat disp8_L, disp8_R, disp8cc_L, disp8cc_R;
			//dispL.convertTo(disp8_L, CV_8U , 1/16. );
			//cv::applyColorMap(disp8_L, disp8cc_L, cv::COLORMAP_JET);
			//cv::imshow("disparite", disp8cc_L);
			//imshow("stereo_L", dispL);
			//imshow("stereo_R", dispR);
			
			
			cv::Mat raw_disp_vis;
			double vis_mult = 2.0;
			cv::ximgproc::getDisparityVis(dispL, raw_disp_vis, vis_mult);

			imshow("raw disparity", raw_disp_vis);
			cv:: Mat filtered_disp_vis;
			cv::ximgproc::getDisparityVis(filteredImg, filtered_disp_vis, vis_mult);
			//namedWindow("filtered disparity", WINDOW_AUTOSIZE);
			imshow("filtered disparity", filtered_disp_vis);
			
			
		}
		
		  //show the frame in the created window
		  drawComposedImage(frame_L_draw, frame_R_draw, boardSize, imgPtsBuff_L_global, imgPtsBuff_R_global, image_count);

		  //wait for for 10 ms until any key is pressed.  
		  //If the 'Esc' key is pressed, break the while loop.
		  //If the any other key is pressed, continue the loop 
		  //If any key is not pressed withing 10 ms, continue the loop 
		  int iKeyPressed = -1;
		  
		  iKeyPressed = cv::waitKey(10);
		  //if (cv::waitKey(10) == 27)
		  if (iKeyPressed == 27)
		  {
			  std::cout << "Esc key is pressed by user. Stoppig the video" << std::endl;
			  break;
		  }		
		  else if (iKeyPressed == (int)('f')) // "f" find a pattern 
		  {
			  if (flagFindPattern)
			  {
				  flagFindPattern = false;
				  std::cout << "\nDeactive camera calibration phase";
			  }
			  else
			  {
				  flagFindPattern = true;
				  std::cout << "\nActive camera calibration phase";
			  }
		  }
		  else if (iKeyPressed == (int)('s')) // "s" save stereo image if user required to calibrate camera
		  {
			  if (flagFindPattern && found_L && found_R)
			  {
				  image_name_L = path + "\\" + image_name + "_L_" + std::to_string(image_count) + ".png";
				  image_name_R = path + "\\" + image_name + "_R_" + std::to_string(image_count) + ".png";

				  imwrite(image_name_L, frame_L);
				  imwrite(image_name_R, frame_R);

				  // Save pattern history
				  imgPtsBuff_L_global.push_back(imgPtsBuff_L);
				  imgPtsBuff_R_global.push_back(imgPtsBuff_R);

				  objPtsBuff_global.push_back(objPtsBuff);

				  // Information about number of pattern image captured
				  std::cout << "\nSave stereo image : " << image_name_L << std::endl << image_name_R;
				  std::cout << "\nNumber image saved until now : " << image_count++;
			  }
			  else
			  {
				  std::cout << "\nNot found pattern in cameras : " ;
			  }
		  }
		  else if (iKeyPressed == (int)('c')) // "c" calibrate stereo system 
		  {
			  flagCalibrateStereoSystem = true;
			  std::cout << "\nInit stereo calibration phase";
		  }
		  else if (iKeyPressed == (int)('v')) // "v" visualization stereo disparity
		  {
			  if (flagStereoVisualization)
			  {
				  flagStereoVisualization = false;
				  std::cout << "\nDeactive stereoVisualization phase";
			  }
			  else
			  {
				  flagStereoVisualization = true;
				  std::cout << "\nActive stereoVisualization phase";
			  }
		  }
	 }

	 return 0;

}

// 

//  

//  
//  
//  if (cv::waitKey(10) == 'c')
//  {
//	  std::cout<<"Starting Calibration\n";

//	  cv::Mat K_l, K_r;
//	  cv::Mat D_l, D_r;
//	  std::vector< cv::Mat > rvecs_l, tvecs_l, rvecs_r, tvecs_r;
//	  int flag = 0;
//	  flag |= cv::CALIB_FIX_K4;
//	  flag |= cv::CALIB_FIX_K5;

//	  // calibrate cameras individually
//	  std::cout << "\nCalibration error Left: " << calibrateCamera(objPtsBuff_global, imgPtsBuff_L_global, frame_L.size(), K_l, D_l, rvecs_l, tvecs_l, flag);
//	  std::cout << "\nCalibration error Right: " << calibrateCamera(objPtsBuff_global, imgPtsBuff_R_global, frame_R.size(), K_r, D_r, rvecs_r, tvecs_r, flag);
//	 

//	  cv::Mat O_K_l, O_K_r;
//	  cv::Size img_L_NewSize, img_R_NewSize;
//	  cv::Rect validPixROI_L, validPixROI_R;

//	  O_K_l = getOptimalNewCameraMatrix(K_l, D_l, frame_L.size(), 1.0, img_L_NewSize, &validPixROI_L);
//	  O_K_r = getOptimalNewCameraMatrix(K_r, D_r, frame_L.size(), 1.0, img_R_NewSize, &validPixROI_R);

//	  std::cout << "\n K_l: " << K_l << "\n Opt_K_l: " << O_K_l << "\n D_l: " << D_l;
//	  std::cout << "\n K_r: " << K_r << "\n Opt_K_r: " << O_K_r << "\n D_r: " << D_r;

//	  cv::Mat  R, F, E;
//	  cv::Vec3d T;
//	  
//	  flag |= cv::CALIB_FIX_INTRINSIC;

//	  std::cout << "\nStereo calibration error: " << 
//		  stereoCalibrate(objPtsBuff_global, imgPtsBuff_L_global, imgPtsBuff_R_global, K_l, D_l, K_r, D_r, frame_L.size(), R, T, E, F, flag, 
//		  cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.0001));

//	  
//	  cv::Mat RL, RR, PL, PR, Q;
//	  cv::Size stereo_NewSize;
//	  cv::Rect stereo_ROI_L, stereo_ROI_R;
//	  stereoRectify( K_l, D_l, K_r, D_r, frame_L.size(), R, T, RL, RR, PL, PR, Q, 0, 0, stereo_NewSize, &stereo_ROI_L, &stereo_ROI_R);

//	  cv::Mat map1_L, map2_L, map1_R, map2_R;

//	  initUndistortRectifyMap(
//		  K_l,
//		  D_l,
//		  RL,
//		  PL,
//		  frame_L.size(),
//		  CV_16SC2,
//		  map1_L, map2_L);

//	  initUndistortRectifyMap(
//		  K_r,
//		  D_r,
//		  RR,
//		  PR,
//		  frame_L.size(),
//		  CV_16SC2,
//		  map1_R, map2_R);

//	  // Parameters for stereo vision
//	  int window_size = 3;
//	      int  	minDisparity = 2;
//		  int  	numDisparities = 130 - minDisparity;
//		  int  	blockSize = window_size;
//		  int  	P1 = 8 * 3 * window_size*window_size;
//		  int  	P2 = 32 * 3 * window_size*window_size;
//		  int  	disp12MaxDiff = 0;
//		  int  	preFilterCap = 0;
//		  int  	uniquenessRatio = 10;
//		  int  	speckleWindowSize = 100;
//		  int  	speckleRange = 32;
//		  int  	mode = cv::StereoSGBM::MODE_SGBM;

//	  cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
//		 	minDisparity, numDisparities, blockSize , P1,P2,disp12MaxDiff,preFilterCap,uniquenessRatio,speckleWindowSize,speckleRange,mode );

//	  cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(sgbm);

//	  cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
//	  wls_filter = cv::ximgproc::createDisparityWLSFilter(sgbm);
//	  wls_filter->setLambda(80000);
//	  wls_filter->setSigmaColor(1.8);


//	//*************************************
//	//***** Starting the StereoVision *****
//	//*************************************
//	//while True:
//	//# Start Reading Camera images
//	//retR, frameR= CamR.read()
//	//retL, frameL= CamL.read()

//	// Rectify the images on rotation and alignement
//	cv::Mat Left_nice, Right_nice;
//	cv::remap(frame_L_draw, Left_nice, map1_L, map2_L, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
//	cv::remap(frame_R_draw, Right_nice, map1_R, map2_R, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
// 					
//	//Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the kalibration parameters founds during the initialisation
//	//Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

//	// Convert from color(BGR) to gray
//	cv::Mat grayL, grayR;
//	cv::cvtColor(Left_nice, grayL,cv::COLOR_BGR2GRAY);
//	cv::cvtColor(Right_nice, grayR, cv::COLOR_BGR2GRAY);
//	
//	//grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
//	//grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)

//	// Compute the 2 images for the Depth_image
//	cv::Mat dispL, dispR;
//	sgbm->compute(grayL, grayR, dispL);
//	right_matcher->compute(grayR, grayL, dispR);

//	//disp= stereo.compute(grayL,grayR)#.astype(np.float32)/ 16
//	//dispL= disp
//	//dispR= stereoR.compute(grayR,grayL)
//	//dispL= np.int16(dispL)
//	//dispR= np.int16(dispR)

//	// Using the WLS filter
//	cv::Mat filteredImg, filteredImgNormalize;
//	wls_filter->filter(dispL, grayL, filteredImg, dispR);
//	cv::normalize(filteredImg, filteredImgNormalize, 255, cv::NORM_MINMAX);

//	cv::Mat disp;
//	dispL.copyTo(disp);
//	disp = ((disp / 16) - minDisparity) / numDisparities;
//	
//	// Filtering the Results with a closing filter
//	// kernel of 3x3
//	int morph_elem = 0;
//	int morph_size = 1;
//	int morph_iteration = 1;
//	cv::Mat element = getStructuringElement(morph_elem,
//		cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
//		cv::Point(morph_size, morph_size));

//	cv::Mat closing;
//	morphologyEx(disp, closing, cv::MORPH_CLOSE, element,
//		cv::Point(-1, -1),
//		morph_iteration);

//	// Get the in value of closing 
//	cv::Mat dispc;
//	cv::Mat minClosing;
//	double minVal, maxVal;
//	cv::minMaxLoc( closing, &minVal, &maxVal, NULL, NULL, NULL);

//	minClosing = cv::Mat(closing.size(), closing.type(), minVal);
//	dispc = (closing - minClosing) * 255;


//	/*
//	filteredImg = wls_filter.filter(dispL,grayL,None,dispR)
//	filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
//	filteredImg = np.uint8(filteredImg)
//	#cv2.imshow('Disparity Map', filteredImg)
//	disp= ((disp.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect

//##    # Resize the image for faster executions
//##    dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

//	# Filtering the Results with a closing filter

//	closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

//	# Colors map
//	dispc= (closing-closing.min())*255
//	dispC= dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
//	disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
//	filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN) 

//	# Show the result for the Depth_image
//	#cv2.imshow('Disparity', disp)
//	#cv2.imshow('Closing',closing)
//	#cv2.imshow('Color Depth',disp_Color)
//	cv2.imshow('Filtered Color Depth',filt_Color)

//	*/

//  }