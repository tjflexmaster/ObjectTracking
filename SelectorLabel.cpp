#include "SelectorLabel.h"
#include <QPainter>

SelectorLabel::SelectorLabel(QWidget *parent) :
    QLabel(parent)
{
    dragging_mouse = false;
}

void SelectorLabel::mousePressEvent(QMouseEvent *ev)
{
    ev->accept();
    start_point = ev->pos();
    dragging_mouse = true;
    end_point = QPoint(0,0);
}

void SelectorLabel::mouseMoveEvent(QMouseEvent *ev)
{
    if ( dragging_mouse ) {
        end_point = ev->pos();
    }
}

void SelectorLabel::mouseReleaseEvent(QMouseEvent *ev)
{
    end_point = ev->pos();
    emit newSelection(QRect(start_point, end_point));
    dragging_mouse = false;
}


void SelectorLabel::paintEvent(QPaintEvent *ev)
{
    QLabel::paintEvent(ev);

    QPainter painter(this);

    if ( !start_point.isNull() && !end_point.isNull() ) {
        QRect rectangle(start_point,end_point);
        painter.setPen(QPen(Qt::green));
        painter.drawRect(rectangle);
    }
}
