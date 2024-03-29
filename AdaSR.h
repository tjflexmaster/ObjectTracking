#ifndef ADASR_H
#define ADASR_H

#include <QObject>
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

#include "gurobi_c++.h"

namespace custom {
    typedef cv::Vec<int, 16> VecHC;
    typedef cv::Vec<double, 8> VecHOG;
    typedef cv::Vec<int, 48> VecHC3;
    typedef cv::Vec<double, 72> VecHOG9;
    typedef cv::Vec<double, 120> VecHOGC;
}

struct Sample {
    QRect rect;
    double scale;
    cv::Mat features;
};

class AdaSR : public QObject
{
    Q_OBJECT
public:
    explicit AdaSR(QRect view_rect, QObject *parent = 0);

    
signals:
    void setupComplete();
    void trackedPoint(QPoint);
    void error();
    
public slots:
    void run();
    void setup(QRect rect, cv::Mat img);
    void track(cv::Mat img);

private:
    //Operating Variables
    bool _initialize;
    bool _tracking;
    bool initializeAdaSR();
    bool findObject();


    //Sample Window Variables
    QRect _view_rect;
    QRect _original_selection;
    cv::Mat _original_img;
    QRect _selection;
    QRect _object;
    int _d;
    QRect _sample_win;
    QPoint _sample_win_top_left;
    int _sample_win_side;
    bool constructSampleWindow();

    cv::Mat _search_area_integral_hc;
    cv::Mat _search_area_integral_hog;

    //SBSR samples and object features
    QList<cv::Mat> _obj_images;
    QList<cv::Mat> _obj_features;
    cv::Mat _obj_feature;
    cv::Mat _sbsr;
    cv::Mat _sparse_coef_vector;
    cv::Mat _sample_features;

    void createFrameIntegralHOGC(cv::Mat *hc_mat, cv::Mat *hog_mat, cv::Mat image);
    custom::VecHC3 getHCVal(QRect rect, cv::Mat* hc_mat);
    custom::VecHOG9 getHOGVal(QRect rect, cv::Mat* hog_mat);
    cv::Mat getHOGCVal(cv::Mat* hc_mat, cv::Mat* hog_mat, QRect rect);
    void setupKalmanFilter();
    void initializeKalmanFilter();

    GRBEnv env;
    void performMinimization(cv::Mat *result, cv::Mat *samples, cv::Mat *obj_feature);
    void calculateSBSR(cv::Mat *sbsr_dst, QList<Sample> *sample_dst, QPoint point, cv::Mat *hc_mat, cv::Mat *hog_mat);
    void getSamples(QList<Sample> *sample_dst, QPoint point, cv::Mat *hc_mat, cv::Mat *hog_mat);
    void getSampleFeatures(cv::Mat*sample_features, QPoint point, cv::Mat *hc_mat, cv::Mat *hog_mat);
    double calcL1Norm(cv::Mat *mat1, cv::Mat *mat2);

    //Tracking
    cv::Mat _tracking_img;
    cv::KalmanFilter _k_filter;
    int _sample_count;
    cv::Mat transitions;
    cv::Mat _adasr;
    cv::Mat estimate;
    cv::Mat estimate_sbsr;
    cv::Mat estimate_pos;
    cv::Mat prediction;
    cv::Mat prediction_sbsr;
    cv::Mat prediction_pos;
    cv::Mat measurement;
    int update_counter;
    QPoint latest_point;
    
};

Q_DECLARE_METATYPE(cv::Mat)

#endif // ADASR_H
