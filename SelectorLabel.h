#ifndef SELECTORLABEL_H
#define SELECTORLABEL_H

#include <QLabel>
#include <QMouseEvent>
#include <QPaintEvent>

class SelectorLabel : public QLabel
{
    Q_OBJECT
public:
    explicit SelectorLabel(QWidget *parent = 0);

    void mousePressEvent(QMouseEvent *ev);
    void mouseMoveEvent(QMouseEvent *ev);
    void mouseReleaseEvent(QMouseEvent *ev);

    void paintEvent(QPaintEvent *ev);
    
signals:
    void newSelection(QRect rect);
    
public slots:

private:
    QPoint start_point;
    QPoint end_point;
    bool dragging_mouse;
    
};

#endif // SELECTORLABEL_H
