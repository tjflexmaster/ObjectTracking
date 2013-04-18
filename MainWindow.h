#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/video/tracking.hpp>

#include <QTimer>
#include <QImage>
#include <QAction>
#include <QThread>
#include "VideoCapturer.h"
#include <QGraphicsScene>
#include <QGraphicsItem>

#include <algorithm>
#include <qmath.h>

#include "gurobi_c++.h"
#include "AdaSR.h"

namespace Ui {
class MainWindow;
}

namespace custom {
    typedef cv::Vec<int, 16> VecHC;
    typedef cv::Vec<double, 8> VecHOG;
    typedef cv::Vec<int, 48> VecHC3;
    typedef cv::Vec<double, 72> VecHOG9;
    typedef cv::Vec<double, 120> VecHOGC;
}

//struct Sample {
//    QRect rect;
//    double scale;
//    cv::Mat features;
//};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

signals:
    void startCapture();
    void stopCapture();
    void trackObject(cv::Mat);
    void setupAdaSR(QRect, cv::Mat);

private:
    Ui::MainWindow *ui;

    QAction *play_or_pause_action;
    QAction *toggle_viewer_action;
    QAction *next_frame_action;

    VideoCapturer *video_capturer;
    QThread *capture_thread;
    AdaSR *processor;
    QThread *processing_thread;

    QGraphicsScene *scene;

    //Variables for processing a single frame at a time
    int frame_number;

//    cv::VideoCapture video_capturer;
//    cv::VideoWriter video_writer;

//    cv::Mat mat_original;

    cv::Mat current_frame;

    cv::Mat mat_processed;

    QImage original_img;
    QImage processed_img;

    std::vector<cv::Vec3f> vec_circles;
    std::vector<cv::Vec3f>::iterator itr_circles;

    //Sample Window Variables
    QRect _selection;
    QRect _object;
//    int _d;
//    QRect _sample_win;
//    QPoint _sample_win_top_left;
//    int _sample_win_side;
//    cv::Mat _search_area_integral_hc;
//    cv::Mat _search_area_integral_hog;
//    QList<QRect> _samples;
//    QList<cv::Mat> _sample_feaures;

//    SBSR samples and object features
//    QList<Sample> _samples;
//    cv::Mat _obj_feature;
//    cv::Mat _sbsr;


    bool _initialized;
    bool _update_sample_window;


//    cv::Mat blue_hc;
//    cv::Mat green_hc;
//    cv::Mat red_hc;
//    cv::Mat integral_hc;
//    cv::Mat integral_hog;
//    bool hog_ready;
//    bool hc_ready;
//    void createFrameIntegralHOGC(cv::Mat *hc_mat, cv::Mat *hog_mat, cv::Mat image);
//    custom::VecHC3 getHCVal(QRect rect, cv::Mat* hc_mat);
//    custom::VecHOG9 getHOGVal(QRect rect, cv::Mat* hog_mat);
//    cv::Mat getHOGCVal(cv::Mat* hc_mat, cv::Mat* hog_mat, QRect rect);

//    GRBEnv env;
//    void performMinimization(cv::Mat *result, QList<Sample> *samples, cv::Mat *obj_feature);
//    void calculateSBSR(cv::Mat *sbsr_dst, QList<Sample> *sample_dst, QPoint point, cv::Mat *hc_mat, cv::Mat *hog_mat);
//    double calcL1Norm(cv::Mat *mat1, cv::Mat *mat2);

    //Tracking
//    cv::KalmanFilter _k_filter;
//    int _sample_count;
//    cv::Mat _adasr;
//    cv::Mat estimate;
//    cv::Mat estimate_sbsr;
//    cv::Mat estimate_pos;
//    cv::Mat prediction;
//    cv::Mat prediction_sbsr;
//    cv::Mat prediction_pos;
//    cv::Mat measurement;
    QPoint latest_point;



//    IImageProcessing *img_proc;

//    QTimer *timer;

public slots:
//    void processFrameAndUpdateGUI();
    void processNextFrame(cv::Mat next_frame);
    void toggleViewer();

private slots:
    void actionPlayOrPause();
    void actionNextFrame();
    void handleNewSelection(QRect rect);
    void handleTrackedPoint(QPoint point);
    void handleAdaSRReady();
};

//Q_DECLARE_METATYPE(cv::Mat)

#endif // MAINWINDOW_H
