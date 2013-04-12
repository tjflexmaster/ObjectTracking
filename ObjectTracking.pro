#-------------------------------------------------
#
# Project created by QtCreator 2013-02-20T12:35:37
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ObjectTracking
TEMPLATE = app


SOURCES += main.cpp\
        MainWindow.cpp \
    VideoCapturer.cpp \
    SelectorLabel.cpp \
    AdaSR.cpp

HEADERS  += MainWindow.h \
    VideoCapturer.h \
    SelectorLabel.h \
    AdaSR.h

FORMS    += MainWindow.ui

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../../opencv/build/x86/vc9/lib -lopencv_core243 -lopencv_highgui243 -lopencv_video243 -lopencv_legacy243 -lopencv_features2d243 -lopencv_imgproc243
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../../opencv/build/x86/vc9/lib -lopencv_core243d -lopencv_highgui243d -lopencv_video243d -lopencv_legacy243d -lopencv_features2d243d -lopencv_imgproc243

INCLUDEPATH += $$PWD/../../../../../../opencv/build/include
DEPENDPATH += $$PWD/../../../../../../opencv/build/include

#####################
# GUROBI
#####################
win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../../gurobi510/win32/lib/ -lgurobi_c++md2008 -lgurobi_c++mt2008 -lgurobi51
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../../gurobi510/win32/lib/ -lgurobi_c++mdd2008 -lgurobi_c++mtd2008 -lgurobi51

INCLUDEPATH += $$PWD/../../../../../../gurobi510/win32/include
DEPENDPATH += $$PWD/../../../../../../gurobi510/win32/include

#####################
# CGAL
#####################
#win32:CONFIG(release, debug|release): LIBS += -L$$quote(C:/Program Files (x86)/CGAL/lib/) -lCGAL_Core-vc90-mt-4.1 -lCGAL_ImageIO-vc90-mt-4.1 -lCGAL-vc90-mt-4.1 -lCGAL-vc90-mt-4.1
#else:win32:CONFIG(debug, debug|release): LIBS += -L$$quote(C:/Program Files (x86)/CGAL/lib/) -lCGAL_Core-vc90-mt-gd-4.1 -lCGAL_ImageIO-vc90-mt-gd-4.1 -lCGAL-vc90-mt-gd-4.1 -lCGAL-vc90-mt-gd-4.1

#INCLUDEPATH += $$quote(C:/Program Files (x86)/CGAL/include)
#DEPENDPATH += $$quote(C:/Program Files (x86)/CGAL/include)

#####################
# GMP
#####################
#win32: LIBS += -L$$quote(C:/Program Files (x86)/CGAL-4.1/auxiliary/gmp/lib/) -llibgmp-10 -llibmpfr-4

#INCLUDEPATH += $$quote(C:/Program Files (x86)/CGAL-4.1/auxiliary/gmp/include)
#DEPENDPATH += $$quote(C:/Program Files (x86)/CGAL-4.1/auxiliary/gmp/include)

######################
# BOOST
######################
#win32:CONFIG(release, debug|release): LIBS += -L$$quote(C:/Program Files (x86)/Boost/lib/) -llibboost_thread-vc90-mt-1_53 -llibboost_system-vc90-mt-1_53
#else:win32:CONFIG(debug, debug|release): LIBS += -L$$quote(C:/Program Files (x86)/Boost/lib/) -llibboost_thread-vc90-mt-gd-1_53 -llibboost_system-vc90-mt-gd-1_53

#INCLUDEPATH += $$quote(C:/Program Files (x86)/Boost/include/boost-1_53)
#DEPENDPATH += $$quote(C:/Program Files (x86)/Boost/include/boost-1_53)


######################
# ZLIB
######################
#win32: LIBS += -L$$quote(C:/Program Files (x86)/zlib_1.2.7/lib/) -lzdll

#INCLUDEPATH += $$quote(C:/Program Files (x86)/zlib_1.2.7/include)
#DEPENDPATH += $$quote(C:/Program Files (x86)/zlib_1.2.7/include)

