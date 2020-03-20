#include <opencv2/opencv.hpp> 
#include <opencv2/stereo.hpp> 
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>

#define WIDTH_SCREEN 1200
#define HEIGHT_SCREEN 900

#ifdef USE_VTK
#include <opencv2/viz.hpp>
#include <opencv2/viz/widget_accessor.hpp>
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iterator>
#include <thread>        
#include <mutex> 
#include <inttypes.h>

using namespace cv;
using namespace std;

#define NBCAMERA 2

struct ParamAlgoStereo{
    Ptr<StereoSGBM> sgbm;
    Ptr<StereoBM> bm;
    
    int typeAlgoSGBM;
    int typePreFilter= StereoBM::PREFILTER_XSOBEL;
    int blockSize=3;
    int nbDisparity=8;
    int extendedSpeckle=1;
    int sizeSpeckle=10;
    int uniqueness=3;
    int preFilterCap=31;
};

struct ParamTarget {
    int nbL = 9, nbC = 6;
    int nbLAruco = 5, nbCAruco = 8;
    float dimCarre = 0.0275;
    float dimAruco = 0.034625;
    float sepAruco = 0.02164;
    int dict = cv::aruco::DICT_7X7_250;
    Ptr<aruco::Dictionary> dictionary;
    Ptr<aruco::CharucoBoard> gridboard;
    Ptr<aruco::Board> board;
    Ptr<aruco::DetectorParameters> detectorParams;
};
struct ParamCalibration3D {
    vector<int> typeCalib3D = { CALIB_FIX_INTRINSIC + CALIB_ZERO_DISPARITY,CALIB_ZERO_DISPARITY };
    Mat R, T, E, F;
    Mat R1,R2,P1,P2;
    Mat Q;
    vector<Mat> m;
    vector<Mat> d;
    Rect valid1,valid2;
    double rms;
    String dataFile;
    vector<vector<Point2f>>  pointsCameraLeft;
    vector<vector<Point2f>>  pointsCameraRight;
    vector<vector<Point3f> > pointsObjets;
};

struct TrackingDistance{
    ParamCalibration3D *pStereo;
    Mat disparity;
    Mat m;
    vector<Point3d> p;
    float zoomDisplay;
};

struct ParamCalibration {
    vector<int> typeCalib2D = { 0,CALIB_FIX_K4 + CALIB_FIX_K5 + CALIB_FIX_K6 + CALIB_ZERO_TANGENT_DIST,CALIB_FIX_K4 + CALIB_FIX_K5 + CALIB_FIX_K6+ CALIB_ZERO_TANGENT_DIST };
    int indexUSB;
    int index;
    vector<Mat> rvecs, tvecs;
    Mat cameraMatrix, distCoeffs;
    Mat mapx;
    Mat mapy;
    int nbGridImage=0;
    vector<Point3f>  gridPoints;
    vector<vector<Point2f>>  pointsCamera;
    vector<vector<Point3f> > pointsObjets;
    Size sizeImage;
    double rms;
    String dataFile;
};

struct ParamCamera {
    VideoCapture *v;
    ParamCalibration *pc;
    int64 tpsGlobal;
    int tpsCapture;
    int captureImage;
    int cmd;
    double timeDelay;
    vector<Mat> imAcq;
    vector<int64> debAcq,finAcq;
    Mat lastImage;
};


vector<mutex> mtxTimeStamp(NBCAMERA);
vector<vector<int64>> tps(mtxTimeStamp.size());

int stopThread=0;

static void videoacquire(ParamCamera *pc);
static vector<Mat> ReadImages(vector<ParamCamera> *pc, vector<VideoCapture> *v,vector<Mat> &x);
static vector<Mat> ReadImagesSynchro(vector<ParamCamera> *pc, vector<VideoCapture> *v);
static vector<VideoCapture> SearchCamera();
static void AddSlide(String nameSlide, String nameWindow, int minSlide, int maxSlide, int defaultValue, int *valSlide, void(*f)(int, void *), void *r = NULL);
static void UpdateParamStereo(int x, void *r);
static void ManagementCmdCamera(ParamCamera *pc);
static double EpipolarRightError(ParamCalibration3D sys3d);
static Mat zoom(Mat , float, TrackingDistance * = NULL);
static bool LoadConfiguration(String fileNameConfiguration, ParamTarget &target, vector<ParamCalibration> &pc, ParamCalibration3D &sys3d, ParamAlgoStereo &);
static void SaveConfiguration(String fileNameConfiguration, ParamTarget target, vector<ParamCalibration> pc, ParamCalibration3D &sys3d,ParamAlgoStereo &);
static void SaveCalibrationData(ParamCalibration *pc1, ParamCalibration *pc2, ParamCalibration3D *sys3d);
static void LoadCalibrationData(String nomFichier, ParamCalibration *pc, ParamCalibration3D *sys3d);
static bool AnalysisCharuco(Mat x,  ParamCalibration &pc, Mat frame, ParamTarget &target);
static bool AnalysisGrid(Mat x,  ParamCalibration *pc, ParamCalibration3D *sys3d, int index, Mat frame, ParamTarget &target);

#ifdef USE_VTK
static void VizWorld(Mat img,  Mat xyz);
static void Vizdisparity(Mat img, Mat disp);
#endif

static void MesureDistance(int event, int x, int y, int flags, void *userdata);


#define DISPLAY_MODE 0x100
#define CALIBRATION_MODE 0x200
#define MAPSTEREO_MODE 0x400
#define EPIPOLAR_MODE 0x800
#define CAMERASETUP_MODE  0x1000


int main (int argc,char **argv)
{

    ParamTarget target;
    int typeDistorsion =0;
    target.dictionary =aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(target.dict));
    target.gridboard = aruco::CharucoBoard::create(target.nbCAruco, target.nbLAruco, target.dimAruco, target.sepAruco, target.dictionary);
    target.board = target.gridboard.staticCast<aruco::Board>();
    target.detectorParams = aruco::DetectorParameters::create();
    target.detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
    target.detectorParams->cornerRefinementMinAccuracy = 0.01;
    float zoomDisplay = 1;

    vector<thread> th;
    vector<double> tpsAverageTime;
    ParamCalibration3D sys3d;
    TrackingDistance sDistance;
    ParamAlgoStereo pStereo;
    vector<ParamCamera> pCamera(2);
    vector<ParamCalibration> pc(pCamera.size());
    vector<VideoCapture> v(pCamera.size());

    Size globalSize(0,0);
    vector<Size> webcamSize(pCamera.size());
    bool configActive = LoadConfiguration("config.yml", target, pc, sys3d,pStereo);
    int64 tpsGlobal = getTickCount();
    int nbCamera=0;

    for (int i = 0; i<pc.size();i++)
    {
        //std::cout<<"\nCamera Index -> pc[i].indexUSB: "<< pc[i].indexUSB << " + CAP_DSHOW: "<< CAP_DSHOW <<" = " <<pc[i].indexUSB+CAP_DSHOW;
        //VideoCapture vid(pc[i].indexUSB+CAP_DSHOW);
        VideoCapture vid(pc[i].indexUSB);
        if (vid.isOpened())
        {
            nbCamera++;
            v[i]=vid;
            vid.set(CAP_PROP_FRAME_WIDTH, pc[i].sizeImage.width);
            vid.set(CAP_PROP_FRAME_HEIGHT, pc[i].sizeImage.height);
            webcamSize[i]=Size(v[i].get(CAP_PROP_FRAME_WIDTH), v[i].get(CAP_PROP_FRAME_HEIGHT));
            pc[i].sizeImage = webcamSize[i];

            if (i==0)
                globalSize= webcamSize[i];
            else if (i%2==1)
                globalSize.width += webcamSize[i].width;
            else
                globalSize.height += webcamSize[i].height;
            
            pCamera[i].v = &v[i];
            pCamera[i].pc = &pc[i];
            pCamera[i].cmd = 0;
            pCamera[i].tpsGlobal= tpsGlobal;
            pCamera[i].timeDelay = 0.1;
            
            thread t(videoacquire, &pCamera[i]);
            t.detach();
        }
        else
            th.push_back( thread() );
    }

    if (nbCamera != 2)
    {
        cout<< "\nNumber cameras: " << nbCamera << "\nError: Not found stereo cameras configuration ..";
        cout<<"\nOnly 2 cameras are needed!\n";
        return 0;
    }
    else
    {
        cout<< "\nNumber cameras: "<< nbCamera << "\nStereo cameras configuration found succesfully !!!";
    }

    
    if (!configActive)
        SaveConfiguration("config.yml", target, pc, sys3d,pStereo);

    if (globalSize.width > WIDTH_SCREEN || globalSize.height > HEIGHT_SCREEN)
        zoomDisplay = min(WIDTH_SCREEN/float(globalSize.width) ,  HEIGHT_SCREEN/float(globalSize.height) );

    Mat frame(globalSize,CV_8UC3,Scalar(0,0,0));
    Point center(20,20);
    vector<int64> tCapture;
    vector<Mat> x;
    int displayMode = 0;
    int algoStereo = 0;

    Mat map11, map12, map21, map22;

    if (!sys3d.R1.empty())
        initUndistortRectifyMap(sys3d.m[0], sys3d.d[0], sys3d.R1, sys3d.P1, webcamSize[0], CV_16SC2, map11, map12);
    if (!sys3d.R2.empty())
        initUndistortRectifyMap(sys3d.m[1], sys3d.d[1], sys3d.R2, sys3d.P2, webcamSize[1], CV_16SC2, map21, map22);
    
    sDistance.pStereo =&sys3d;
    sDistance.zoomDisplay = zoomDisplay;
    
    imshow("Cameras",zoom(frame, zoomDisplay));
    namedWindow("Control",WINDOW_NORMAL);
    
    if (!map11.empty() && !map21.empty())
    {
        pStereo.bm = StereoBM::create(16*pStereo.nbDisparity, 2 * pStereo.blockSize + 1);
        pStereo.sgbm = StereoSGBM::create(0, 16*pStereo.nbDisparity, 2 * pStereo.blockSize + 1);
        pStereo.bm->setPreFilterType(pStereo.typePreFilter);
        pStereo.bm->setUniquenessRatio(pStereo.uniqueness);
        pStereo.sgbm->setUniquenessRatio(pStereo.uniqueness);
        pStereo.bm->setSpeckleWindowSize(pStereo.sizeSpeckle);
        pStereo.sgbm->setSpeckleWindowSize(pStereo.sizeSpeckle);
        pStereo.bm->setSpeckleRange(pStereo.extendedSpeckle);
        pStereo.sgbm->setSpeckleRange(pStereo.extendedSpeckle);
        
        AddSlide("blockSize", "Control", 2, 100, pStereo.blockSize, &pStereo.blockSize, UpdateParamStereo, &pStereo);
        AddSlide("nbDisparity", "Control", 1, 100, pStereo.nbDisparity, &pStereo.nbDisparity, UpdateParamStereo, &pStereo);
        AddSlide("uniqueness", "Control", 3, 100, pStereo.uniqueness, &pStereo.uniqueness, UpdateParamStereo, &pStereo);
        AddSlide("extendedSpeckle", "Control", 1, 10, pStereo.extendedSpeckle, &pStereo.extendedSpeckle, UpdateParamStereo, &pStereo);
        AddSlide("sizeSpeckle", "Control", 3, 100, pStereo.sizeSpeckle, &pStereo.sizeSpeckle, UpdateParamStereo, &pStereo);

        int alg= StereoSGBM::MODE_SGBM;
        pStereo.sgbm->setMode(alg);
    }

    for (int i = 0; i < target.nbL; i++)
        for (int j = 0; j < target.nbC; j++)
            for (int k=0;k<pc.size();k++)
                pc[k].gridPoints.push_back(
                    Point3f(float(j*target.dimCarre), float(i*target.dimCarre), 0));
    int code =0;
    int flags= 0;
    int indexCamera=0,indImage=0;


    Mat art0, art1;
    Mat disparity;
    
    vector<vector<Point2d>> p2D;
    vector<Point2f> segment;
    for (int i = 0; i < 10;i++)
        segment.push_back(Point2f(webcamSize[0].width/2, webcamSize[0].height*i / 10.0));
    
    Mat equEpipolar;
    namedWindow("Webcam 0");
    namedWindow("disparity");
    setMouseCallback("Webcam 0", MesureDistance, &sDistance);
    setMouseCallback("disparity", MesureDistance, &sDistance);

    int cameraSelect = 0;

    do 
    {
        code = waitKey(1);

        if ( displayMode & CAMERASETUP_MODE )
        {

            switch (code) 
            {
            case '0':
                mtxTimeStamp[0].lock();
                pCamera[0].cmd = pCamera[0].cmd | (CAMERASETUP_MODE );
                mtxTimeStamp[0].unlock();
                mtxTimeStamp[1].lock();
                pCamera[1].cmd = pCamera[1].cmd & (~CAMERASETUP_MODE );
                mtxTimeStamp[1].unlock();
                cameraSelect=0;
                break;
            case '1':
                mtxTimeStamp[0].lock();
                pCamera[0].cmd = pCamera[0].cmd & (~CAMERASETUP_MODE );
                mtxTimeStamp[0].unlock();
                mtxTimeStamp[1].lock();
                pCamera[1].cmd = pCamera[1].cmd | (CAMERASETUP_MODE );
                mtxTimeStamp[1].unlock();
                cameraSelect=1;
                break;
            case 'g':
            case 'G':
            case 'b':
            case 'B':
            case 'E':
            case 'e':
            case 'c':
            case 'C':
            case 'w':
                mtxTimeStamp[cameraSelect].lock();
                pCamera[cameraSelect].cmd = pCamera[cameraSelect].cmd | code;
                mtxTimeStamp[cameraSelect].unlock();
                break;
            case 'R':
                displayMode = displayMode& (~CAMERASETUP_MODE );
                mtxTimeStamp[cameraSelect].lock();
                pCamera[cameraSelect].cmd = pCamera[cameraSelect].cmd & (~CAMERASETUP_MODE );
                mtxTimeStamp[cameraSelect].unlock();
                break;
            }
        }
        else
        {
            switch (code) 
            {
            case 'R':
                displayMode = displayMode| (CAMERASETUP_MODE );                    
                mtxTimeStamp[cameraSelect].lock();
                pCamera[cameraSelect].cmd = pCamera[cameraSelect].cmd | CAMERASETUP_MODE ;
                mtxTimeStamp[cameraSelect].unlock();
                break;
            case 'a':
                for (int i = 0; i < v.size(); i++)
                {
                    mtxTimeStamp[i].lock();
                    if (pCamera[i].cmd & DISPLAY_MODE)
                        pCamera[i].cmd = pCamera[i].cmd & ~DISPLAY_MODE;
                    else
                        pCamera[i].cmd = pCamera[i].cmd | DISPLAY_MODE;
                    displayMode = pCamera[i].cmd;
                    mtxTimeStamp[i].unlock();
                }
                break;
            case 'b':
                frame = Mat::zeros(globalSize, CV_8UC3);
                for (int i = 0; i < pc.size(); i++)
                {
                    pc[i].pointsObjets.clear();
                    pc[i].dataFile="";
                    pc[i].pointsCamera.clear();
                }
                sys3d.pointsCameraRight.clear();
                sys3d.pointsCameraLeft.clear();
                sys3d.dataFile="";
                sys3d.pointsObjets.clear();
                break;
            case 'c':
                if (pc.size() <= indexCamera || pc[indexCamera].pointsCamera.size() == 0)
                {
                    cout << "Aucune grille pour la calibration\n";
                    break;
                }
                if (pc[indexCamera].dataFile.length() == 0)
                    SaveCalibrationData(&pc[indexCamera], NULL, NULL);
                pc[indexCamera].cameraMatrix = Mat();
                pc[indexCamera].distCoeffs = Mat();
                for (int i = 0; i<pc[indexCamera].typeCalib2D.size(); i++)
                    pc[indexCamera].rms = calibrateCamera(pc[indexCamera].pointsObjets, pc[indexCamera].pointsCamera, webcamSize[indexCamera], pc[indexCamera].cameraMatrix,
                        pc[indexCamera].distCoeffs, pc[indexCamera].rvecs, pc[indexCamera].tvecs, pc[indexCamera].typeCalib2D[i], TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 5000000, 1e-8));
                cout << "RMS = " << pc[indexCamera].rms << "\n";
                cout << pc[indexCamera].cameraMatrix << "\n";
                cout << pc[indexCamera].distCoeffs << "\n";

                SaveConfiguration("config.yml", target, pc, sys3d,pStereo);
                break;
            case 'D':
                if (indexCamera != 1)
                    pc[indexCamera].pointsCamera.clear();
                indexCamera = 1;
                x = ReadImages(&pCamera, &v, x);
                if (x.size() == 2 && !x[0].empty() && !x[1].empty())
                {
                    Rect dst(Point(webcamSize[1].width, 0), webcamSize[1]);
                    Mat y(webcamSize[1], CV_8UC3, Scalar(0, 0, 0));

                    if (AnalysisCharuco(x[indexCamera], pc[indexCamera], y, target))
                    {
                        frame(dst) = y + frame(dst);
                        imshow("Cameras", zoom(frame, zoomDisplay,NULL));
                        if (pc[indexCamera].dataFile.length() != 0)
                            pc[indexCamera].dataFile = "";
                    }
                }
                break;
            case 'd':
                if (indexCamera != 1)
                {
                    pc[indexCamera].pointsCamera.clear();
                }
                indexCamera = 1;
                x = ReadImages(&pCamera, &v, x);
                if (x.size() == 2 && !x[0].empty() && !x[1].empty())
                {
                    Rect dst(Point(webcamSize[1].width, 0), webcamSize[1]);
                    Mat y(webcamSize[1], CV_8UC3, Scalar(0, 0, 0));
                    if (AnalysisGrid(x[indexCamera], &pc[indexCamera], NULL, 0, y, target))
                    {
                        frame(dst) = y + frame(dst);
                        imshow("Cameras", zoom(frame, zoomDisplay));
                        if (pc[indexCamera].dataFile.length() != 0)
                            pc[indexCamera].dataFile = "";
                    }
                }
                break;
            case 'e':
                imwrite("imgL.png", x[0]);
                imwrite("imgR.png", x[1]);
                if (!sDistance.disparity.empty())
                {
                    sDistance.disparity.convertTo(disparity, CV_32F, 1 / 16.);
                    FileStorage fs("disparity.yml", FileStorage::WRITE);
                    if (fs.isOpened())
                        fs << "Image" << disparity;
                    FileStorage fs2("disparitybrut.yml", FileStorage::WRITE);
                    if (fs2.isOpened())
                        fs2 << "Image" << sDistance.disparity;
                }
                break;
            case 'G':
                if (indexCamera != 0)
                    pc[indexCamera].pointsCamera.clear();
                indexCamera = 0;
                x = ReadImages(&pCamera, &v, x);
                if (x.size() == 2)
                    if (AnalysisCharuco(x[indexCamera], pc[indexCamera],frame,target))
                    {
                        if (pc[indexCamera].dataFile.length() != 0)
                            pc[indexCamera].dataFile = "";
                        imshow("Cameras", zoom(frame, zoomDisplay));
                    }
                break;
            case 'g':
                if (indexCamera != 0)
                    pc[indexCamera].pointsCamera.clear();
                indexCamera = 0;
                x = ReadImages(&pCamera, &v, x);
                if (x.size() == 2 && !x[0].empty() && !x[1].empty())
                    if (AnalysisGrid(x[indexCamera], &pc[indexCamera], NULL, 0, frame, target))
                    {
                        if (pc[indexCamera].dataFile.length() != 0)
                            pc[indexCamera].dataFile = "";
                        imshow("Cameras", zoom(frame, zoomDisplay));
                    }
                break;
            case 'l':
                if (displayMode&EPIPOLAR_MODE)
                    displayMode = displayMode & (~EPIPOLAR_MODE);
                else
                    displayMode = displayMode | (EPIPOLAR_MODE);

                if (!sys3d.F.empty())
                {
                    computeCorrespondEpilines(segment, 1, sys3d.F, equEpipolar);
                    cout << equEpipolar;
                }
                break;
            case 'o':
                if (!sDistance.disparity.empty())
                {
                    Mat xyz;
                    sDistance.disparity.convertTo(disparity, CV_32F, 1 / 16.);
                    reprojectImageTo3D(disparity, xyz, sys3d.Q, true);
#ifdef USE_VTK
                    VizWorld(x[0],  xyz);
#else
                    cout<<"VTK non installé";
#endif
                }
                break;
            case 'O':
                if (!sDistance.disparity.empty())
                {
                    sDistance.disparity.convertTo(disparity, CV_32F, 1 / 16.);
#ifdef USE_VTK
                    Vizdisparity(x[0], disparity);
#else
                    cout << "VTK non installé";
#endif
                }
                break;
            case 's':
                if (indexCamera != 3)
                {
                    sys3d.pointsCameraLeft.clear();
                    sys3d.pointsCameraRight.clear();
                    sys3d.pointsObjets.clear();
                    if (sys3d.dataFile != "")
                        sys3d.dataFile = "";
                }
                indexCamera = 3;
                x = ReadImages(&pCamera, &v,x);
                if (x.size() == 2)
                {
                    vector<Point2f> echiquierg, echiquierd;
                    bool grilleg = findChessboardCorners(x[0], Size(target.nbC, target.nbL), echiquierg, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
                    bool grilled = findChessboardCorners(x[1], Size(target.nbC, target.nbL), echiquierd, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
                    if (grilleg && grilled)
                    {
                        Rect dst(Point(webcamSize[1].width, 0), webcamSize[1]);
                        Mat y(webcamSize[1], CV_8UC3, Scalar(0, 0, 0));
                        AnalysisGrid(x[0], &pc[0], &sys3d, 0, frame, target);
                        AnalysisGrid(x[1], &pc[0], &sys3d, 1, y, target);
                        frame(dst) = y + frame(dst);
                        imshow("Cameras", zoom(frame, zoomDisplay));
                    }
                }
                break;
            case 'S':
                if (indexCamera != 3)
                {
                    for (int i = 0; i < pc.size(); i++)
                    {
                        sys3d.pointsCameraLeft.clear();
                        sys3d.pointsCameraRight.clear();
                        sys3d.pointsObjets.clear();
                        if (sys3d.dataFile != "")
                            sys3d.dataFile = "";
                    }
                }
                indexCamera = 3;
                x = ReadImagesSynchro(&pCamera, &v);
                if (x.size() == 2)
                {
                    vector<Point2f> echiquierg, echiquierd;
                    vector< int > idsg;
                    vector< vector< Point2f > > refArucog, refus;
                    vector< int > idsd;
                    vector< vector< Point2f > > refArucod;
                    aruco::detectMarkers(x[0], target.dictionary, refArucog, idsg, target.detectorParams, refus);
                    aruco::refineDetectedMarkers(x[0], target.board, refArucog, idsg, refus);
                    aruco::detectMarkers(x[1], target.dictionary, refArucod, idsd, target.detectorParams, refus);
                    aruco::refineDetectedMarkers(x[1], target.board, refArucod, idsd, refus);
                    if (idsg.size() > 0 && idsd.size()>0)
                    {
                        aruco::drawDetectedMarkers(frame, refArucog, idsg);
                        Rect dst(Point(webcamSize[1].width, 0), webcamSize[1]);
                        Mat y(webcamSize[1], CV_8UC3, Scalar(0, 0, 0));
                        aruco::drawDetectedMarkers(y, refArucod, idsd);
                        frame(dst) = y + frame(dst);
                        imshow("Cameras", zoom(frame, zoomDisplay));
                        vector<Point3f> pReel;
                        for (int ir = 0; ir<refArucog.size(); ir++)
                        {
                            vector<int>::iterator p=find(idsd.begin(),idsd.end(), idsg[ir]);
                            if (p != idsd.end())
                            {
                                int id=(p- idsd.begin());
                                for (int jr = 0; jr<refArucog[ir].size(); jr++)
                                {
                                    echiquierg.push_back(refArucog[ir][jr]);
                                    echiquierd.push_back(refArucod[id][jr]);
                                    pReel.push_back(target.board.get()->objPoints[idsg[ir]][jr]);
                                }
                            }
                        }
                        sys3d.pointsObjets.push_back(pReel);
                        sys3d.pointsCameraLeft.push_back(echiquierg);
                        sys3d.pointsCameraRight.push_back(echiquierd);
                    }
                }
                break;
            case 't':
                if (!sys3d.R1.empty() && !sys3d.R2.empty())
                    if (algoStereo != 0)
                        algoStereo = 0;
                    else if (pStereo.bm)
                        algoStereo = 1;
                break;
            case 'T':
                if (!sys3d.R1.empty() && !sys3d.R2.empty())
                    if (algoStereo != 0)
                        algoStereo = 0;
                    else if (pStereo.sgbm)
                        algoStereo = 2;
                break;
            case 'u':
                typeDistorsion = (typeDistorsion +1)%4;
                cout<< typeDistorsion <<"\n";
                switch (typeDistorsion) {
                case 0:
                    map11 = Mat();
                    map12 = Mat();
                    map21 = Mat();
                    map22 = Mat();
                    break;
                case 1:
                    if (!sys3d.R1.empty())
                        initUndistortRectifyMap(sys3d.m[0], sys3d.d[0], Mat(), Mat(), webcamSize[0], CV_16SC2, map11, map12);
                    if (!sys3d.R2.empty())
                        initUndistortRectifyMap(sys3d.m[1], sys3d.d[1], Mat(), Mat(), webcamSize[1], CV_16SC2, map21, map22);
                    break;
                case 2:
                    if (!sys3d.R1.empty())
                        initUndistortRectifyMap(sys3d.m[0], Mat(), sys3d.R1, sys3d.P1, webcamSize[0], CV_16SC2, map11, map12);
                    if (!sys3d.R2.empty())
                        initUndistortRectifyMap(sys3d.m[1], Mat(), sys3d.R2, sys3d.P2, webcamSize[1], CV_16SC2, map21, map22);
                    break;
                case 3:
                    if (!sys3d.R1.empty())
                        initUndistortRectifyMap(sys3d.m[0], sys3d.d[0], sys3d.R1, sys3d.P1, webcamSize[0], CV_16SC2, map11, map12);
                    if (!sys3d.R2.empty())
                        initUndistortRectifyMap(sys3d.m[1], sys3d.d[1], sys3d.R2, sys3d.P2, webcamSize[1], CV_16SC2, map21, map22);
                    break;
                }
                break;
            case '3':
                if (sys3d.pointsCameraLeft.size() != sys3d.pointsCameraRight.size() || sys3d.pointsCameraLeft.size() == 0)
                {
                    cout << "Pas de grille coherente pour le calibrage 3D\n";
                    break;
                }
                SaveCalibrationData(&pc[0], &pc[1], &sys3d);

                sys3d.m[0] = pc[0].cameraMatrix.clone();
                sys3d.m[1] = pc[1].cameraMatrix.clone();
                sys3d.d[0] = pc[0].distCoeffs.clone();
                sys3d.d[1] = pc[1].distCoeffs.clone();
                for (int i = 0; i < sys3d.typeCalib3D.size(); i++)
                {
                    sys3d.rms = stereoCalibrate(sys3d.pointsObjets, sys3d.pointsCameraLeft, sys3d.pointsCameraRight,
                        sys3d.m[0], sys3d.d[0], sys3d.m[1], sys3d.d[1],
                        webcamSize[0], sys3d.R, sys3d.T, sys3d.E, sys3d.F,
                        sys3d.typeCalib3D[i],
                        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 5000000, 1e-8));
                }
                cout << "Erreur quadratique =" << sys3d.rms << endl;
                cout << EpipolarRightError(sys3d) << "\n";
                cout << sys3d.m[0] << "\n" << sys3d.d[0] << "\n" << sys3d.m[1] << "\n" << sys3d.d[1] << "\n";
                cout << sys3d.R << "\n" << sys3d.T << "\n" << sys3d.E << "\n" << sys3d.F << "\n";
                if (!sys3d.R.empty())
                {
                    stereoRectify(sys3d.m[0], sys3d.d[0], sys3d.m[1], sys3d.d[1],
                        webcamSize[0], sys3d.R, sys3d.T, sys3d.R1, sys3d.R2, sys3d.P1, sys3d.P2, sys3d.Q,
                        CALIB_ZERO_DISPARITY, -1, Size(), &sys3d.valid1, &sys3d.valid2);
                    SaveConfiguration("config.yml", target, pc, sys3d,pStereo);
                }
                break;
            }

        }

        if ( displayMode )
        {
            if (!algoStereo)
                x=ReadImages(&pCamera, &v, x);
            else
                x= ReadImagesSynchro(&pCamera, &v);
            if (x.size() == 2)
            {
                if (displayMode & EPIPOLAR_MODE && equEpipolar.rows == segment.size())
                {
                    for (int i = 0; i < equEpipolar.rows; i++)
                        {
                        circle(x[0], segment[i], 5, Scalar(255, 255, 0));
                        line(x[0], Point(0, segment[i].y), Point(x[0].cols - 1, segment[i].y), Scalar(255, 255, 0));
                        float a = equEpipolar.at<float>(i, 0);
                        float b = equEpipolar.at<float>(i, 1);
                        float c = equEpipolar.at<float>(i, 2);
                        if (a != 0)
                        {
                            float x0 = -c / a, x1 = (-b*x[1].rows - c) / a;
                            vector <Point2f> xOrig{Point2f(x0, 0), Point2f(x1, x[1].rows)};
                            line(x[1], xOrig[0], xOrig[1], Scalar(255, 255, 0));
                        }
                    //remap(x[1], x[1], map21b, map22b, INTER_LINEAR);

                    }
                }
                if (x.size() == 2 && typeDistorsion!=0 && !map11.empty() && !map21.empty() && !x[0].empty() && !x[1].empty())
                {
                    remap(x[0], x[0], map11, map12, INTER_LINEAR);
                    remap(x[1], x[1], map21, map22, INTER_LINEAR);
                }

                for (int i = 0; i < x.size(); i++)
                {
                    if (!x[i].empty())
                        imshow(format("Webcam %d", i), zoom(x[i], zoomDisplay,&sDistance));
                }
                if (displayMode & EPIPOLAR_MODE)
                {
                    Rect dst(0, 0, 0, 0);
                    for (int i = 0; i < x.size(); i++)
                    {
                        if (i == 0)
                            dst = Rect(Point(0, 0), webcamSize[i]);
                        else if (i % 2 == 1)
                            dst.x += webcamSize[i].width;
                        else
                            dst.y += webcamSize[i].height;
                        x[i].copyTo(frame(dst));
                    }
                    
                    imshow("Cameras", zoom(frame, zoomDisplay));
                    if (code == 'e')
                    {
                        imwrite("epipolar.png",frame);
                    }
                }

            }
            if (algoStereo && x.size() == 2 && !x[0].empty() && !x[1].empty())
            {
                Mat disp8, disp8cc, imgL, imgD;
                if (algoStereo ==2)
                    pStereo.sgbm->compute(x[0], x[1], sDistance.disparity);
                else
                {
                    cvtColor(x[0], imgL, COLOR_BGR2GRAY);
                    cvtColor(x[1], imgD, COLOR_BGR2GRAY);
                    pStereo.bm->compute(imgL, imgD, sDistance.disparity);
                }
                sDistance.disparity.convertTo(disp8, CV_8U, 1 / 16.);
                applyColorMap(disp8, disp8cc, COLORMAP_JET);
                imshow("disparity", zoom(disp8cc, zoomDisplay));
            }
        }
    }
    while (code!=27);

    for (int i = 0; i < v.size(); i++)
    {
        mtxTimeStamp[i].lock();
    }
    stopThread=1;
    for (int i = 0; i < v.size(); i++)
    {
        mtxTimeStamp[i].unlock();
    }
    std::this_thread::sleep_for (std::chrono::seconds(2));
    th.clear();
    SaveConfiguration("config.yml", target, pc, sys3d, pStereo);
    return 0;
}


void videoacquire(ParamCamera *pc)
{
    double aaButter[11], bbButter[11];
    aaButter[0] = -0.9996859;
    aaButter[1] = -0.9993719;
    aaButter[2] = -0.9968633;
    aaButter[3] = -0.9937365;
    aaButter[4] = -0.9690674;
    aaButter[5] = -0.9390625;
    aaButter[6] = -0.7265425;
    aaButter[7] = -0.5095254;
    aaButter[8] = -0.3249;
    aaButter[9] = -0.1584;
    aaButter[10] = -0.0;
    bbButter[0] = 0.0001571;
    bbButter[1] = 0.0003141;
    bbButter[2] = 0.0015683;
    bbButter[3] = 0.0031318;
    bbButter[4] = 0.0154663;
    bbButter[5] = 0.0304687;
    bbButter[6] = 0.1367287;
    bbButter[7] = 0.2452373;
    bbButter[8] = 0.3375;
    bbButter[9] = 0.4208;
    bbButter[10] = 0.5;
    int indFiltreMoyenne=0;

    int64 tpsInit = getTickCount();
    int64 tckPerSec = static_cast<int64> (getTickFrequency());
    int displayMode = 0;
    int64  tpsFrame = 0, tpsFramePre;
    int64 tpsFrameAsk;//, periodeTick = static_cast<int64> (getTickFrequency() / pc->fps);
    Mat frame,cFrame;
    Mat     frame1;
    Mat     frame2;

    *(pc->v) >> frame;
    frame.copyTo(frame1);
    frame.copyTo(frame2);
    tpsFrame = getTickCount();
    int64 offsetTime = pc->tpsGlobal + 2 * getTickFrequency();

    do
    {
        tpsFrame = getTickCount();
    } while (tpsFrame<offsetTime);
    
    tpsFrameAsk = offsetTime;
    
    int i = 0, index = pc->pc->index;

    for (int nbAcq = 0;;)
    {
        tpsFramePre = getTickCount();
        *(pc->v) >> frame;
        frame = bbButter[indFiltreMoyenne] * (frame1 + frame2) - aaButter[indFiltreMoyenne] * frame;
        frame1.copyTo(frame2);
        frame.copyTo(frame1);

        tpsFrame = getTickCount();
        nbAcq++;
        int64 dt = tpsFrame - tpsFrameAsk;
        mtxTimeStamp[pc->pc->index].lock();
        if (displayMode&CAMERASETUP_MODE )
            ManagementCmdCamera(pc);
        if (stopThread)
            break;
        if (pc->captureImage)
        {
            if (tpsFrame >= pc->tpsCapture)
            {
                pc->captureImage = 0;
                pc->imAcq.push_back(frame.clone());
                pc->debAcq.push_back(tpsFramePre);
                pc->finAcq.push_back(tpsFrame);
            }
        }
        if (pc->cmd & DISPLAY_MODE)
            displayMode = displayMode | DISPLAY_MODE;
        else
            displayMode = displayMode & ~DISPLAY_MODE;
        if (pc->cmd & CAMERASETUP_MODE )
            displayMode = displayMode | CAMERASETUP_MODE ;
        else
            displayMode = displayMode & ~CAMERASETUP_MODE ;
        mtxTimeStamp[pc->pc->index].unlock();

        if (displayMode & DISPLAY_MODE)
        {
            mtxTimeStamp[pc->pc->index].lock();
            frame.copyTo(pc->lastImage);
            mtxTimeStamp[pc->pc->index].unlock();
        }

        tpsFrame = getTickCount();
    }
    mtxTimeStamp[pc->pc->index].unlock();
}

vector<VideoCapture> SearchCamera()
{
    vector<VideoCapture> v;
    FileStorage fs("config.xml", FileStorage::READ);
    if (fs.isOpened())
    {
        FileNode n = fs["cam"];
        if (!n.empty())
        {
            FileNodeIterator it = n.begin();
            int nbCamera = 0;
            while (it != n.end() && nbCamera<NBCAMERA)
            {
                nbCamera++;
                FileNode p = *it;
                int i;
                Size s;
                (*it) >> i;
                it++;
                (*it) >> s;
                it++;
                VideoCapture vid(i);

                if (vid.isOpened() )
                {
                    vid.set(CAP_PROP_FRAME_WIDTH, s.width);
                    vid.set(CAP_PROP_FRAME_HEIGHT, s.height);
                    v.push_back(vid);
                }
                else
                    vid.release();
            }

        }
        fs.release();
    }
    else
    {

        for (size_t i = 0; i<NBCAMERA; i++)
        {
            VideoCapture video;
            video.open(static_cast<int>(i));
            if (!video.isOpened())
            {
                cout << " cannot openned camera : " << i << endl;
            }
            else
            {
                video.set(CAP_PROP_FRAME_WIDTH, 640);
                video.set(CAP_PROP_FRAME_HEIGHT, 480);
                v.push_back(video);
            }
        }
    }
    return v;
}

void ManagementCmdCamera(ParamCamera *pc)
{
    double x = pc->pc->index;
    switch (pc->cmd & 0xFF)
    {
    case 'g':
        x = pc->v->get(CAP_PROP_GAIN) - 1;
        pc->v->set(CAP_PROP_GAIN, x);
        break;
    case 'G':
        x = pc->v->get(CAP_PROP_GAIN) + 1;
        pc->v->set(CAP_PROP_GAIN, x);
        break;
    case 'b':
        x = pc->v->get(CAP_PROP_BRIGHTNESS) - 1;
        pc->v->set(CAP_PROP_BRIGHTNESS, x);
        break;
    case 'B':
        x = pc->v->get(CAP_PROP_BRIGHTNESS) + 1;
        pc->v->set(CAP_PROP_BRIGHTNESS, x);
        break;
    case 'E':
        x = pc->v->get(CAP_PROP_EXPOSURE) + 1;
        pc->v->set(CAP_PROP_EXPOSURE, x);
        break;
    case 'e':
        x = pc->v->get(CAP_PROP_EXPOSURE) - 1;
        pc->v->set(CAP_PROP_EXPOSURE, x);
        break;
    case 'c':
        x = pc->v->get(CAP_PROP_SATURATION) - 1;
        pc->v->set(CAP_PROP_SATURATION, x);
        break;
    case 'C':
        x = pc->v->get(CAP_PROP_SATURATION) + 1;
        pc->v->set(CAP_PROP_SATURATION, x);
        break;
    case 'w':
        pc->v->set(CAP_PROP_SETTINGS, x);
        break;
    }
    pc->cmd = pc->cmd & 0xFFFFFF00;
}

vector<Mat> ReadImagesSynchro(vector<ParamCamera> *pc, vector<VideoCapture> *v)
{
    vector<Mat> x;
    for (int i = 0; i<v->size(); i++)
        mtxTimeStamp[i].lock();
    int64 tps;
    for (int i = 0; i < v->size(); i++)
    {
        tps = getTickCount() + (*pc)[i].timeDelay * getTickFrequency() ;
        (*pc)[i].captureImage = 1;
        (*pc)[i].tpsCapture = tps;
        mtxTimeStamp[i].unlock();

    }
    int64 sleepTime = static_cast<int64>(25000+(tps  - getTickCount()) / getTickFrequency() * 1000000);
    std::this_thread::sleep_for(std::chrono::microseconds(sleepTime));
    for (int i = 0; i < v->size(); i++)
    {
        mtxTimeStamp[i].lock();
    }
    Rect dst(0, 0, 0, 0);
    for (int i = 0; i < v->size(); i++)
    {
        if ((*pc)[i].captureImage == 0)
        {
            x.push_back((*pc)[i].imAcq[0]);
            (*pc)[i].imAcq.pop_back();
            (*pc)[i].debAcq.pop_back();
            (*pc)[i].finAcq.pop_back();
        }
        else
            (*pc)[i].captureImage = 0;

        mtxTimeStamp[i].unlock();
    }
    if (x.size() == 1)
    {
        (*pc)[0].timeDelay += 0.01;
        (*pc)[1].timeDelay += 0.01;

    }
    if (x.size()!=v->size())
        cout<< "Image perdue!\n";
    return x;
}

bool LoadConfiguration(String fileNameConfiguration, ParamTarget &target, vector<ParamCalibration> &pc, ParamCalibration3D &sys3d, ParamAlgoStereo &pStereo)
{
    FileStorage fs(fileNameConfiguration, FileStorage::READ);

    pc[0].index = 0;
    pc[1].index = 1;
    pc[0].indexUSB = 0;
    pc[1].indexUSB = 1;
    
    if (fs.isOpened())
    {
        if (!fs["EchiquierNbL"].empty())
            fs["EchiquierNbL"] >> target.nbL;
        if (!fs["EchiquierNbC"].empty())
            fs["EchiquierNbC"]>> target.nbC;
        if (!fs["EchiquierDimCarre"].empty())
            fs["EchiquierDimCarre"] >> target.dimCarre;
        if (!fs["ArucoNbL"].empty())
            fs["ArucoNbL"] >> target.nbLAruco;
        if (!fs["ArucoNbC"].empty())
            fs["ArucoNbC"] >> target.nbCAruco;
        if (!fs["ArucoDim"].empty())
            fs["ArucoDim"] >> target.dimAruco;
        if (!fs["ArucoSep"].empty())
            fs["ArucoSep"] >> target.sepAruco;
        if (!fs["ArucoDict"].empty())
            fs["ArucoDict"] >> target.dict;
        if (!fs["Cam0index"].empty())
            fs["Cam0index"] >> pc[0].indexUSB;
        if (!fs["Cam0Size"].empty())
            fs["Cam0Size"] >> pc[0].sizeImage;
        if (!fs["Cam1Size"].empty())
            fs["Cam1Size"] >> pc[1].sizeImage;
        if (!fs["Cam1index"].empty())
            fs["Cam1index"] >> pc[1].indexUSB;
        if (!fs["typeCalib1"].empty())
            fs["typeCalib1"] >> pc[1].typeCalib2D;
        if (!fs["typeCalib0"].empty())
            fs["typeCalib0"] >> pc[0].typeCalib2D;
        if (!fs["typeCalib3d"].empty())
            fs["typeCalib3d"] >> sys3d.typeCalib3D;
        if (!fs["cameraMatrice0"].empty())
            fs["cameraMatrice0"] >> pc[0].cameraMatrix;
        if (!fs["cameraDistorsion0"].empty())
            fs["cameraDistorsion0"] >> pc[0].distCoeffs;
        if (!fs["cameraMatrice1"].empty())
            fs["cameraMatrice1"] >> pc[1].cameraMatrix;
        if (!fs["cameraDistorsion1"].empty())
            fs["cameraDistorsion1"] >> pc[1].distCoeffs;
        if (!fs["ficDonnee0"].empty())
        {
            fs["ficDonnee0"] >> pc[0].dataFile;
            if (pc[0].dataFile.length()>0)
                LoadCalibrationData(pc[0].dataFile, &pc[0], NULL);

        }
        if (!fs["ficDonnee1"].empty() )
        {
            fs["ficDonnee1"] >> pc[1].dataFile;
            if (pc[1].dataFile.length()>0)
                LoadCalibrationData(pc[1].dataFile, &pc[1], NULL);
        }
        if (!fs["ficDonnee3d"].empty())
        {
            fs["ficDonnee3d"] >> sys3d.dataFile;
            if (sys3d.dataFile.length()>0)
                LoadCalibrationData(sys3d.dataFile, NULL, &sys3d);
        }
        if (!fs["R"].empty())
        {
            fs["R"] >> sys3d.R;
            fs["T"]>> sys3d.T;
            fs["R1"]>>sys3d.R1;
            fs["R2"]>>sys3d.R2;
            fs["P1"]>>sys3d.P1;
            fs["P2"]>>sys3d.P2;
            fs["Q"]>>sys3d.Q;
            fs["F"]>>sys3d.F;
            fs["E"]>>sys3d.E;
            fs["rect1"]>>sys3d.valid1;
            fs["rect2"]>>sys3d.valid2;
            sys3d.m.resize(2);
            sys3d.d.resize(2);
            fs["M0"]>>sys3d.m[0];
            fs["D0"]>>sys3d.d[0];
            fs["M1"]>>sys3d.m[1];
            fs["D1"]>>sys3d.d[1];
        }
        if (!fs["typeAlgoSGBM"].empty())
        {
            fs["typeAlgoSGBM"]>> pStereo.typeAlgoSGBM;
            fs["preFilterType"] >> pStereo.typePreFilter;
            fs["preFilterCap"] >> pStereo.preFilterCap;
            fs["blockSize"] >> pStereo.blockSize;
            fs["numDisparities"] >> pStereo.nbDisparity;
            fs["uniquenessRatio"] >> pStereo.uniqueness;
            fs["speckleRange"] >> pStereo.extendedSpeckle;
            fs["speckleSize"] >> pStereo.sizeSpeckle;
        }
        return true;
    }
    return false;
}

void SaveConfiguration(String fileNameConfiguration,ParamTarget target,vector<ParamCalibration> pc, ParamCalibration3D &sys3d, ParamAlgoStereo &pStereo)
{
    if (fileNameConfiguration == "config.yml")
    {
        time_t rawtime;
        struct tm * timeinfo;
        char buffer[256];

        time(&rawtime);
        timeinfo = localtime(&rawtime);

        strftime(buffer, 256, "config%Y%m%d%H%M%S.yml", timeinfo);
        rename(fileNameConfiguration.c_str(), buffer);
    }
    FileStorage fs(fileNameConfiguration, FileStorage::WRITE);

    if (fs.isOpened())
    {
        time_t t1;
        time(&t1);
        struct tm *t2 = localtime(&t1);
        char tmp[1024];
        strftime(tmp, sizeof(tmp) - 1, "%c", t2);

        fs << "date" << tmp;
        fs << "EchiquierNbL"<<target.nbL;
        fs << "EchiquierNbC"<<target.nbC;
        fs << "EchiquierDimCarre"<<target.dimCarre;
        fs << "ArucoNbL"<<target.nbLAruco;
        fs << "ArucoNbC" << target.nbCAruco;
        fs << "ArucoDim" << target.dimAruco;
        fs << "ArucoSep"<< target.sepAruco;
        fs << "ArucoDict" << target.dict;
        fs << "typeCalib0" << pc[0].typeCalib2D;
        fs << "Cam0index" << pc[0].indexUSB;
        fs << "Cam0Size" << pc[0].sizeImage;
        fs << "typeCalib1" << pc[1].typeCalib2D;
        fs << "Cam1index" << pc[1].indexUSB;
        fs << "Cam1Size" << pc[1].sizeImage;
        fs << "cameraMatrice0" << pc[0].cameraMatrix;
        fs << "cameraDistorsion0" << pc[0].distCoeffs;
        fs << "cameraMatrice1" << pc[1].cameraMatrix;
        fs << "cameraDistorsion1" << pc[1].distCoeffs;
        fs << "ficDonnee0" << pc[0].dataFile;
        fs << "ficDonnee1" << pc[1].dataFile;
        fs << "ficDonnee3d" << sys3d.dataFile;
        fs << "typeCalib3d" << sys3d.typeCalib3D;
        fs << "R" << sys3d.R << "T" << sys3d.T << "R1" << sys3d.R1 << "R2" << sys3d.R2 << "P1" << sys3d.P1 << "P2" << sys3d.P2;
        fs << "Q" << sys3d.Q << "F" << sys3d.F << "E" << sys3d.E << "rect1" << sys3d.valid1 << "rect2" << sys3d.valid2;
        if (sys3d.m.size() >= 2)
        {
            fs << "M0" << sys3d.m[0] << "D0" << sys3d.d[0];
            fs << "M1" << sys3d.m[1] << "D1" << sys3d.d[1];

        }
        fs << "typeAlgoSGBM" << pStereo.typeAlgoSGBM;
        fs << "preFilterType" << pStereo.typePreFilter;
        fs << "preFilterCap" << pStereo.preFilterCap;
        fs << "blockSize" << pStereo.blockSize;
        fs << "numDisparities" << pStereo.nbDisparity;
        fs << "uniquenessRatio" << pStereo.uniqueness;
        fs << "speckleRange" << pStereo.extendedSpeckle;
        fs << "speckleSize" << pStereo.sizeSpeckle;

    }
}


vector<Mat> ReadImages(vector<ParamCamera> *pc, vector<VideoCapture> *v,vector<Mat> &x)
{
    vector<Mat> xx;
    if (x.size()!=v->size())
        xx.resize(v->size());
    else
        xx=x;
    for (int i = 0; i < v->size(); i++)
    {
        mtxTimeStamp[i].lock();
        (*pc)[i].lastImage.copyTo(xx[i]);
        mtxTimeStamp[i].unlock();
    }
    return xx;
}

#ifdef USE_VTK

void VizWorld(Mat img, Mat xyz)
{
    vector<Point3d> points;
    vector<Vec3b> couleur;
    for (int i = 0; i < xyz.rows; i++)
    {
        Vec3f *pLig = (Vec3f*)(xyz.ptr(i));
        for (int j = 0; j < xyz.cols ; j++, pLig++)
        {
            if (pLig[0][2] < 10000 )
            {
                Vec3d p1(pLig[0][0], pLig[0][1], pLig[0][2]);
                points.push_back(p1);
                couleur.push_back(img.at<Vec3b>(i, j));
            }
        }
    }
    viz::Viz3d fen3D("Monde");
    viz::WCloud nuage(points,  couleur);
    fen3D.showWidget("I3d", nuage);
    fen3D.spin();
}


void Vizdisparity(Mat img, Mat disp)
{
    vector<Point3d> points;
    vector<int> polygone;
    vector<Vec3b> couleur;
    int nbPoint = 0;
    for (int i = 1; i < img.rows - 1; i++)
    {
        float *d = disp.ptr<float>(i) + 1;
        for (int j = 1; j < img.cols - 1; j++, d++)
        {
            float disparity= *d;
            if (disparity<0)
                disparity =10000;
            Vec3d p1(j,i, *d);
            Vec3d p2(j-1,i , *d);
            Vec3d p3(j,i-1, *d);
            Vec3d p4(j-1,i-1, *d);
            points.push_back(p1);
            points.push_back(p2);
            points.push_back(p3);
            points.push_back(p4);
            if (*d >= 0)
            {
                couleur.push_back(img.at<Vec3b>(i, j));
                couleur.push_back(img.at<Vec3b>(i, j));
                couleur.push_back(img.at<Vec3b>(i, j));
                couleur.push_back(img.at<Vec3b>(i, j));
            }
            else
            {
                couleur.push_back(Vec3b(0,0,0));
                couleur.push_back(Vec3b(0, 0, 0));
                couleur.push_back(Vec3b(0, 0, 0));
                couleur.push_back(Vec3b(0, 0, 0));
            }
            polygone.push_back(4);
            polygone.push_back(nbPoint);
            polygone.push_back(nbPoint + 2);
            polygone.push_back(nbPoint + 1);
            polygone.push_back(nbPoint + 3);
            nbPoint += 4;
        }
    }
    viz::Viz3d fen3D("disparity 3D");
    viz::WMesh reseauFacette(points, polygone, couleur);
    fen3D.showWidget("I3d", reseauFacette);
    fen3D.spin();
}
#endif

void MesureDistance(int event, int x, int y, int flags, void * userData)
{
    TrackingDistance *sDistance=(TrackingDistance*)userData;
    if (sDistance->disparity.empty())
        return;
    if (event == EVENT_FLAG_LBUTTON)
    {
        Point pImg(x / sDistance->zoomDisplay, y / sDistance->zoomDisplay);
        if (sDistance->disparity.at<short>(pImg) <=-1)
            return;
        Point3d p(x / sDistance->zoomDisplay, y/ sDistance->zoomDisplay, sDistance->disparity.at<short>(pImg));
        p.z = p.z/16;
        sDistance->p.push_back(p);
        Mat ptdisparity(sDistance->p), ptXYZ;
        sDistance->m.release();
        perspectiveTransform(ptdisparity, sDistance->m, sDistance->pStereo->Q);
        cout << "\n ++++++++++\n";
        cout << ptdisparity;
        cout << "\n ";
        for (int i=0;i<sDistance->m.rows;i++)
            cout << ptdisparity.at<Vec3d>(i)<<" = "<<sDistance->m.at<Vec3d>(i)<<" --> "<<norm(sDistance->m.at<Vec3d>(i))<<"\n";
    }
    if (event == EVENT_FLAG_RBUTTON)
    {
        sDistance->p.clear();
    }
}

Mat zoom(Mat x, float w,TrackingDistance *s)
{
    if (s && !s->disparity.empty() && s->p.size() > 0)
    {

    }
    if (w!=1)
    {
        Mat y;
        resize(x,y,Size(),w,w);
        return y;
    }
    return x;
 }

void AddSlide(String nameSlide, String nameWindow, int minSlide, int maxSlide, int defaultValue, int *valSlide, void(*f)(int, void *), void *r)
{
    createTrackbar(nameSlide, nameWindow, valSlide, 1, f, r);
    setTrackbarMin(nameSlide, nameWindow, minSlide);
    setTrackbarMax(nameSlide, nameWindow, maxSlide);
    setTrackbarPos(nameSlide, nameWindow, defaultValue);
}

void UpdateParamStereo(int x, void * r)
{
    ParamAlgoStereo *pStereo= (ParamAlgoStereo *)r;

    pStereo->bm->setBlockSize(2*pStereo->blockSize+1);
    pStereo->sgbm->setBlockSize(2 * pStereo->blockSize + 1);
    pStereo->bm->setNumDisparities(16*pStereo->nbDisparity);
    pStereo->sgbm->setNumDisparities(16*pStereo->nbDisparity);
    pStereo->bm->setUniquenessRatio(pStereo->uniqueness);
    pStereo->sgbm->setUniquenessRatio(pStereo->uniqueness);
    pStereo->bm->setSpeckleWindowSize(pStereo->sizeSpeckle);
    pStereo->sgbm->setSpeckleWindowSize(pStereo->sizeSpeckle);
    pStereo->bm->setSpeckleRange(pStereo->extendedSpeckle);
    pStereo->sgbm->setSpeckleRange(pStereo->extendedSpeckle);
}

double EpipolarRightError(ParamCalibration3D sys3d)
{
    double err = 0;
    int nbPoints = 0;
    vector<Vec3f> lines[2];
    for (int i = 0; i < sys3d.pointsCameraLeft.size(); i++)
    {
        int nbPt = (int)sys3d.pointsCameraLeft[i].size();
        Mat ptImg[2];
        undistortPoints(sys3d.pointsCameraLeft[i], ptImg[0], sys3d.m[0], sys3d.d[0], Mat(), sys3d.m[0]);
        undistortPoints(sys3d.pointsCameraRight[i], ptImg[1], sys3d.m[1], sys3d.d[1], Mat(), sys3d.m[1]);
        computeCorrespondEpilines(ptImg[0], 1, sys3d.F, lines[0]);
        computeCorrespondEpilines(ptImg[1], 2, sys3d.F, lines[1]);
        for (int j = 0; j < nbPt; j++)
        {
            double errij1 = fabs(ptImg[0].at<Vec2f>(0,j)[0]*lines[1][j][0] +
                ptImg[0].at<Vec2f>(0, j)[1] *lines[1][j][1] + lines[1][j][2]);
            double errij2 = fabs(ptImg[1].at<Vec2f>(0, j)[0] *lines[0][j][0] +
                ptImg[1].at<Vec2f>(0, j)[1]*lines[0][j][1] + lines[0][j][2]);
            err += errij1 + errij2;
        }
        nbPoints += nbPt;
    }
    return err/ nbPoints/2;
}

bool AnalysisCharuco(Mat x, ParamCalibration &pc,Mat frame,ParamTarget &target)
{
    
    vector< int > ids;
    vector< vector< Point2f > > refAruco, refus;
    vector<Point2f> echiquier;
    aruco::detectMarkers(x, target.dictionary, refAruco, ids, target.detectorParams);
 
    aruco::refineDetectedMarkers(x, target.board, refAruco, ids, refus);
    if (ids.size() > 0)
    {
        Mat currentCharucoCorners, currentCharucoIds;
        aruco::interpolateCornersCharuco(refAruco, ids, x, target.gridboard, currentCharucoCorners,
            currentCharucoIds);
        aruco::drawDetectedCornersCharuco(frame, currentCharucoCorners, currentCharucoIds);
        vector<Point3f> pReel;
        for (int ir = 0; ir<refAruco.size(); ir++)
            for (int jr = 0; jr<refAruco[ir].size(); jr++)
            {
                echiquier.push_back(refAruco[ir][jr]);
                pReel.push_back(target.board.get()->objPoints[ids[ir]][jr]);
            }

        for (int ir = 0; ir < currentCharucoIds.rows; ir++)
        {
            int index= currentCharucoIds.at<int>(ir, 0);
            echiquier.push_back(currentCharucoCorners.at<Point2f>(ir,0));
            pReel.push_back(target.gridboard->chessboardCorners[index]);

        }
        pc.pointsObjets.push_back(pReel);
        pc.pointsCamera.push_back(echiquier);
    }
    else
        return false;
    return true;
}

bool AnalysisGrid(Mat x, ParamCalibration *pc, ParamCalibration3D *sys3d,int index, Mat frame, ParamTarget &target)
{
    vector<Point2f> echiquier;
    bool grille = findChessboardCorners(x, Size(target.nbC, target.nbL), echiquier, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
    if (grille)
    {
        Mat imGris;
        cvtColor(x, imGris, COLOR_BGR2GRAY);
        cornerSubPix(imGris, echiquier, Size(5, 5), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.01));
        drawChessboardCorners(frame, Size(target.nbC, target.nbL), Mat(echiquier), false);
        if (sys3d==NULL && pc)
        {
            pc->pointsCamera.push_back(echiquier);
            pc->pointsObjets.push_back(pc->gridPoints);
        }
        if (sys3d)
        {
            if (index==0)
            {
                sys3d->pointsCameraLeft.push_back(echiquier);
                sys3d->pointsObjets.push_back(pc->gridPoints);
            }
            else
                sys3d->pointsCameraRight.push_back(echiquier);
        }

    }
    else
        return false;
    return true;
}

void LoadCalibrationData(String nomFichier, ParamCalibration *pc, ParamCalibration3D *sys3d)
{
    FileStorage fs(nomFichier, FileStorage::READ);
    if (!fs.isOpened())
    {
        return;
    }
    if (pc)
    {
        if (!fs["nbGrille"].empty())
            fs["nbGrille"] >> pc->nbGridImage;
        if (!fs["Grille"].empty())
            fs["Grille"] >> pc->pointsCamera;
        if (!fs["Objet"].empty())
            fs["Objet"] >> pc->pointsObjets;
    }
    else if( sys3d)
    {
        if (!fs["GrilleG"].empty())
            fs["GrilleG"] >> sys3d->pointsCameraLeft;
        if (!fs["GrilleD"].empty())
            fs["GrilleD"] >> sys3d->pointsCameraRight;
        if (!fs["Objet"].empty())
            fs["Objet"] >> sys3d->pointsObjets;

    }
    fs.release();
}


void SaveCalibrationData(ParamCalibration *pc1, ParamCalibration *pc2,ParamCalibration3D *sys3d)
{
    if (!sys3d)
    {
        ParamCalibration *pc;

        if (pc1)
            pc =pc1;
        else
            pc =pc2;
        int nbPts=0;
        for (int i=0;i<pc->pointsCamera.size();i++)
            nbPts+= pc->pointsCamera[i].size();
        pc->dataFile= format("Echiquier%d_%d.yml", pc->indexUSB, getTickCount());
        FileStorage fEchiquier(pc->dataFile, FileStorage::WRITE);
        fEchiquier << "nbGrille" << (int)pc->pointsCamera.size();
        fEchiquier << "nbPoints" << (int)nbPts;
        fEchiquier << "Grille" << pc->pointsCamera;
        fEchiquier << "Objet" << pc->pointsObjets;
        fEchiquier.release();

    }
    else
    {
        sys3d->dataFile = format("EchiquierStereo_%d.yml", getTickCount());
        FileStorage fEchiquier(sys3d->dataFile, FileStorage::WRITE);
        int nbPts = 0;
        for (int i = 0; i<pc1->pointsCamera.size(); i++)
            nbPts += pc1->pointsCamera[i].size();
        fEchiquier << "nbGrille" << (int)pc1->pointsCamera.size();
        fEchiquier << "nbPoints" << (int)nbPts;
        fEchiquier << "GrilleG" << sys3d->pointsCameraLeft;
        fEchiquier << "GrilleD" << sys3d->pointsCameraRight;
        fEchiquier << "Objet" << sys3d->pointsObjets;
        fEchiquier.release();
    }

}

