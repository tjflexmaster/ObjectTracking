#ifndef VIDEOCAPTURER_H
#define VIDEOCAPTURER_H

#include <QObject>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <QMutex>
#include <QTimer>

class VideoCapturer : public QObject
{
    Q_OBJECT

public:
    VideoCapturer(QObject *parent = 0);
    bool isPaused();
    bool isActive();

signals:
    void currentFrame(cv::Mat);
    void testSignal();
    
public slots:
    void run();
    void startCapture();
    void stopCapture();

private:
    cv::VideoCapture video_capturer;
    cv::VideoWriter video_writer;

    cv::Mat _current_frame;

    QMutex _paused_mutex;
    bool _paused;
    bool _active;

    QTimer *timer;
    
};

#endif // VIDEOCAPTURER_H
