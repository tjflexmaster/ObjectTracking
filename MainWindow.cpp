#include "MainWindow.h"
#include "ui_MainWindow.h"
#include <QtCore>
#include <QThread>
#include <algorithm>
#include <qmath.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <QDebug>
#include "gurobi_c++.h"
#include <random>
#include "AdaSR.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

//    video_capturer.open(0);
//    if ( !video_capturer.isOpened() ) {
//        ui->information_display->appendPlainText("error: Unable to access the webcam");
//        return;
//    } else {
//        ui->information_display->appendPlainText("Accessed Webcam");
//    }

//    //Setup the Graphics Scene
//    scene = new QGraphicsScene();
//    ui->main_graphics_view->setScene(scene);

    toggle_viewer_action = new QAction(tr("Toggle Viewer"), this);
    ui->mainToolBar->addAction(toggle_viewer_action);
    connect(toggle_viewer_action, SIGNAL(triggered()), this, SLOT(toggleViewer()));

    play_or_pause_action = new QAction(tr("Pause"), this);
    play_or_pause_action->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
    ui->mainToolBar->addAction(play_or_pause_action);
    connect(play_or_pause_action, SIGNAL(triggered()), this, SLOT(actionPlayOrPause()));

    //Move to next frame
    next_frame_action = new QAction(tr("Next Frame"), this);
    next_frame_action->setIcon(style()->standardIcon(QStyle::SP_MediaSkipForward));
    ui->mainToolBar->addAction(next_frame_action);
    connect(next_frame_action, SIGNAL(triggered()), this, SLOT(actionNextFrame()));
    frame_number = 0;

//    timer = new QTimer(this);
//    connect(timer, SIGNAL(timeout()), this, SLOT(processFrameAndUpdateGUI()));
//    timer->start(20);

    video_capturer = new VideoCapturer();

    capture_thread = new QThread();
    video_capturer->moveToThread(capture_thread);
    connect(capture_thread, SIGNAL(started()), video_capturer, SLOT(startCapture()));
    connect(capture_thread, SIGNAL(finished()), video_capturer, SLOT(deleteLater()));

//    capture_thread->start();

    connect(video_capturer, SIGNAL(currentFrame(cv::Mat)), this, SLOT(processNextFrame(cv::Mat)));
    connect(this, SIGNAL(startCapture()), video_capturer, SLOT(startCapture()));
    connect(this, SIGNAL(stopCapture()), video_capturer, SLOT(stopCapture()));

    //Perform processing in another thread
    processor = new AdaSR(ui->original_display->rect());

    processing_thread = new QThread();
    processor->moveToThread(processing_thread);
    connect(processing_thread, SIGNAL(finished()), processor, SLOT(deleteLater()));
    connect(this, SIGNAL(trackObject(cv::Mat)), processor, SLOT(track(cv::Mat)));
    connect(this, SIGNAL(setupAdaSR(QRect,cv::Mat)), processor, SLOT(setup(QRect,cv::Mat)));
    connect(processor, SIGNAL(trackedPoint(QPoint)), this, SLOT(handleTrackedPoint(QPoint)));
    connect(processor, SIGNAL(setupComplete()), this, SLOT(handleAdaSRReady()));

    processing_thread->start();


    //Connect the SelectorLabel Selection
    _initialized = false;
    _update_sample_window = false;
    connect(ui->original_display, SIGNAL(newSelection(QRect)), this, SLOT(handleNewSelection(QRect)));

}

MainWindow::~MainWindow()
{
    delete ui;
    video_capturer->stopCapture();
    delete video_capturer;
    delete capture_thread;
}

void MainWindow::processNextFrame(cv::Mat next_frame)
{
//    ui->information_display->appendPlainText(QString("Frame Received"));

    if ( next_frame.empty() ) {
        return;
    }

    //Disconnect from the video capture until we are done processing
//    disconnect(video_capturer, SIGNAL(currentFrame(cv::Mat)), this, SLOT(processNextFrame(cv::Mat)));

    //Make a copy to show tracking on
    next_frame.copyTo(mat_processed);


    /////////////////////////////////////////////
    // PERFORM FRAME PROCESSING
    /////////////////////////////////////////////

    if ( _update_sample_window ) {
        qDebug() << "Setup AdaSR: " << QTime::currentTime().toString("hh:mm:ss.zzz");
        emit setupAdaSR(_selection, next_frame);
//        processor->setup(_selection, next_frame);
        _update_sample_window = false;
    } else if ( _initialized ) {
        qDebug() << "Track Object: " << QTime::currentTime().toString("hh:mm:ss.zzz");
        emit trackObject(next_frame);
        _initialized = false;
    }


//    //Hough Circle Tracking
//    cv::inRange(next_frame, cv::Scalar(0,0,170), cv::Scalar(100, 100, 256), mat_processed);
//    cv::GaussianBlur(mat_processed, mat_processed, cv::Size(9,9), 2, 2);
//    cv::HoughCircles(mat_processed, vec_circles, CV_HOUGH_GRADIENT, 2, mat_processed.rows / 4, 100, 50, 10, 400);

//    for( itr_circles = vec_circles.begin(); itr_circles != vec_circles.end(); itr_circles++ ) {
//        this->ui->information_display->appendPlainText(QString("ball position x =") + QString::number((*itr_circles)[0]).rightJustified(4, ' ') +
//                QString(", y =") + QString::number((*itr_circles)[1]).rightJustified(4, ' ') +
//                QString(", radius =") + QString::number((*itr_circles)[2]).rightJustified(7, ' '));
//        cv::Vec3i c = *itr_circles;
//        cv::circle(next_frame, cv::Point(c[0], c[1]), 3, cv::Scalar(0,255,0), CV_FILLED);
//        cv::circle(next_frame, cv::Point(c[0], c[1]), c[2], cv::Scalar(0,0,255), 3, CV_AA);
//    }



    //////////////////////////////////////////////
    // OUTPUT THE PROCESSED IMAGES TO THE GUI
    //////////////////////////////////////////////
//    qDebug() << "Start Display Image: " << QTime::currentTime().toString("hh:mm:ss.zzz");

    //Draw Sample Window on the image
//    cv::rectangle(next_frame, cv::Rect(_sample_win.topLeft().x(), _sample_win.topLeft().y(), _sample_win.width(), _sample_win.height()),cv::Scalar(0,0,0));

    cv::circle(mat_processed, cv::Point(latest_point.x(), latest_point.y()), 30, cv::Scalar(0,255,0));

    cv::cvtColor(next_frame, next_frame, CV_BGR2RGB);
    QImage original_img((uchar*)next_frame.data, next_frame.cols, next_frame.rows, next_frame.step, QImage::Format_RGB888);
//    QImage processed_img((uchar*)mat_processed.data, mat_processed.cols, mat_processed.rows, mat_processed.step, QImage::Format_Indexed8);
    cv::cvtColor(mat_processed, mat_processed, CV_BGR2RGB);
    QImage processed_img((uchar*)mat_processed.data, mat_processed.cols, mat_processed.rows, mat_processed.step, QImage::Format_RGB888);


    ui->original_display->setPixmap(QPixmap::fromImage(original_img));
    ui->processed_display->setPixmap(QPixmap::fromImage(processed_img));
//    qDebug() << "End Display Image: " << QTime::currentTime().toString("hh:mm:ss.zzz");


    //Reconnect to the video feed
//    connect(video_capturer, SIGNAL(currentFrame(cv::Mat)), this, SLOT(processNextFrame(cv::Mat)));

    ///////////////////////////////////////////////
    // RECORD THE VIDEO IF RECORDING
    ///////////////////////////////////////////////

    //Do this in another thread
}

void MainWindow::actionPlayOrPause()
{
    if( !video_capturer->isPaused() ) {
        std::cout << "Paused" << std::endl;
        video_capturer->stopCapture();
        play_or_pause_action->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    } else {
        std::cout << "Playing" << std::endl;
        video_capturer->startCapture();
        play_or_pause_action->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
    }
}

void MainWindow::actionNextFrame()
{
    cv::Mat image;
    QString path("C:\\Users\\TJ-ASUS\\Downloads\\redteam\\frame00000");
    QString num = QString::number(frame_number);
    QString mod_path = path.left(path.size() - num.size());
    QString final = mod_path + num + QString(".jpg");
    image = cv::imread(final.toStdString(), CV_LOAD_IMAGE_COLOR);

    if ( !image.data ) {
        this->ui->information_display->appendPlainText(QString("Unable to load image: ") + final);
        return;
    }

    this->ui->information_display->appendPlainText(QString("Loaded Image: ") + final);

    image.copyTo(current_frame);
    cv::cvtColor(image, image, CV_BGR2RGB);
    QImage original_img((uchar*)image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
    ui->original_display->setPixmap(QPixmap::fromImage(original_img));

    if ( _initialized ) {
        this->ui->information_display->appendPlainText(QString("Tracking Object ") + QTime::currentTime().toString("hh:mm:ss.zzz"));
        next_frame_action->setDisabled(true);
        emit trackObject(image);
    }

    frame_number+=5;
}

void MainWindow::toggleViewer()
{

}


void MainWindow::handleNewSelection(QRect rect)
{
    _selection = rect;
    _update_sample_window = true;

    if ( current_frame.empty() || !next_frame_action->isEnabled() )
        return;

    next_frame_action->setDisabled(true);
    this->ui->information_display->appendPlainText(QString("Initializing AdaSR ") + QTime::currentTime().toString("hh:mm:ss.zzz"));
    emit setupAdaSR(_selection, current_frame);
}

void MainWindow::handleTrackedPoint(QPoint point)
{
    this->ui->information_display->appendPlainText(QString("Tracking Complete! ") + QTime::currentTime().toString("hh:mm:ss.zzz"));
    qDebug() << "Latest Point: " << point;
    latest_point = point;
    _initialized = true;

    current_frame.copyTo(mat_processed);

    cv::circle(mat_processed, cv::Point(latest_point.x(), latest_point.y()), 30, cv::Scalar(0,255,0));

    cv::cvtColor(mat_processed, mat_processed, CV_BGR2RGB);
    QImage processed_img((uchar*)mat_processed.data, mat_processed.cols, mat_processed.rows, mat_processed.step, QImage::Format_RGB888);

    ui->processed_display->setPixmap(QPixmap::fromImage(processed_img));

    next_frame_action->setDisabled(false);
}

void MainWindow::handleAdaSRReady()
{
    this->ui->information_display->appendPlainText(QString("AdaSR Initialized ") + QTime::currentTime().toString("hh:mm:ss.zzz"));
    _initialized = true;
    next_frame_action->setDisabled(false);
}
