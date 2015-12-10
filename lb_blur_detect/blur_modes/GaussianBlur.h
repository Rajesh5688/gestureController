
#ifndef GAUSSIANBLUR_H
#define GAUSSIANBLUR_H
#include "Blur.h"

/**
 * @brief The GaussianBlur class
 *
 * Gaussian Blur as the name suggests is a gaussian kernel with the variance is specified thorugh the radius parameter.
 */
class GaussianBlur : public Blur
{
public:
    const QString getName() const{
        return "GaussianBlur";
    }

    virtual QImage* buildKernelImage(const GaussianBlur* gaussianBlur) {
        // Double radius plus 2*3 pixels to ensure that generated kernel will be fitted inside the image
        int size = 3.5 * gaussianBlur->_radius + 6;
        size += size%2;
        QImage* kernelImage = new QImage(size, size, QImage::Format_RGB32);
        kernelImage->fill(Qt::red);
        // Prepare painter to have antialiasing and sub-pixel accuracy
        QPainter kernelPainter(kernelImage);
        kernelPainter.setRenderHint(QPainter::Antialiasing);
        // Workarround to have high accuracy, otherwise drawLine method has some micro-mistakes in the rendering
        QPen pen = kernelPainter.pen();
        pen.setColor(Qt::white);
        kernelPainter.setPen(pen);
        for (int y=0; y<size; y++) {
            for (int x=0; x<size; x++) {
                int value = 255*(pow((double)M_E, -(pow((double)x-size/2,2)+pow((double)y-size/2,2))/(2*pow((double)gaussianBlur->_radius,2))));
                kernelImage->setPixel(x,y,qRgb(value,value,value));
            }
        }
        kernelPainter.end();
        return kernelImage;
    }
};
#endif // GAUSSIANBLUR_H
