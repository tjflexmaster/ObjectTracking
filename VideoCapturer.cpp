#include "VideoCapturer.h"
#include <QDebug>

VideoCapturer::VideoCapturer(QObject *parent):
    QObject(parent)
{
    video_capturer.open(0);
    if ( !video_capturer.isOpened() ) {
        std::cout << "Error" << std::endl;
        qDebug() << "Error: Unable to access the device: " << 0;
        _paused = true;
        _active = false;
    } else {
        qDebug() << "Video Capturer initialized properly";
        _paused = true;
        _active = true;
    }

    timer = NULL;

}

void VideoCapturer::run()
{
//    qDebug() << "Broadcast next frame.";
    if ( _active ) {
        if ( !_paused ) {
            video_capturer >> _current_frame;

            emit currentFrame(_current_frame);
//            std::cout << "Broadcast_next_frame" << std::endl;
//            emit testSignal();
        }
    }
}

void VideoCapturer::startCapture()
{
//    _paused_mutex.lock();
    if ( timer == NULL ) {
        timer = new QTimer();
        timer->start(67);


    }

    connect(timer, SIGNAL(timeout()), this, SLOT(run()));

    _paused = false;
//    _paused_mutex.unlock();
}

void VideoCapturer::stopCapture()
{
//    _paused_mutex.lock();
    if ( timer != NULL ) {
//        timer->stop();
        disconnect(timer, SIGNAL(timeout()), this, SLOT(run()));
        _paused =true;
    }
//    _paused_mutex.unlock();
}

bool VideoCapturer::isPaused()
{
    return _paused;
}

bool VideoCapturer::isActive()
{
    return _active;
}
