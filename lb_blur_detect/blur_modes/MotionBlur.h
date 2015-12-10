#ifndef MOTIONBLUR_H
#define MOTIONBLUR_H
#include "Blur.h"

/**
 * @brief The MotionBlur class
 *
 * Kernel creation class that creates a motion blur kernel where the kernel values will be oriented based on the angle value specified and
 * spread of it depends on the radius specified.
 */
class MotionBlur : public Blur
{
public:
    double angle;
    const QString getName() const{
        return "MotionBlur";
    }

    virtual QImage* buildKernelImage(const MotionBlur* motionBlur) {
        // motionLength plus 2*3 pixels to ensure that generated kernel will be fitted inside the image
        double motionLength = motionBlur->_radius * 2;
        double motionAngle = motionBlur->angle;
        int size = motionLength + 6;
        size += size%2;
        QImage* kernelImage = new QImage(size, size, QImage::Format_RGB32);
        kernelImage->fill(Qt::black);
        // Prepare painter to have antialiasing and sub-pixel accuracy
        QPainter kernelPainter(kernelImage);
        kernelPainter.setRenderHint(QPainter::Antialiasing);
        // Workarround to have high accuracy, otherwise drawLine method has some micro-mistakes in the rendering
        QPen pen = kernelPainter.pen();
        pen.setColor(Qt::white);
        pen.setWidthF(1.01);
        kernelPainter.setPen(pen);
        double center = 0.5 + size/2;
        double motionAngleRad = M_PI*motionAngle/180;
        QLineF line(center - motionLength*cos(motionAngleRad)/2,
                    center - motionLength*sin(motionAngleRad)/2,
                    center + motionLength*cos(motionAngleRad)/2,
                    center + motionLength*sin(motionAngleRad)/2);
        kernelPainter.drawLine(line);
        kernelPainter.end();
        return kernelImage;
    }
};
#endif // MOTIONBLUR_H
