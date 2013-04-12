#include "MainWindow.h"
#include <QApplication>
#include "opencv2/core/core.hpp"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    qRegisterMetaType<cv::Mat>();
    
    return a.exec();
}
