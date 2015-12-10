#ifndef FOCUSBLUR_H
#define FOCUSBLUR_H

#include "Blur.h"

/**
 * @brief The FocusBlur class
 *
 * Focus Blur is modelled as something that is similar to the gaussian blur with radius as the parameter that is fed into
 * this function.
 *
 * The kernel values reduces smoothly starting from the center as it goes away.
 */
class FocusBlur : public Blur
{
public:
    double _edgeFeather;
    double _correctionStrength;

    const QString getName() const{
        return "FocusBlur";
    }

    virtual QImage* buildKernelImage() {
        double radius = _radius;
        double edgeFeather = _edgeFeather;
        double correctionStrength = _correctionStrength;
        // Double radius plus 2*3 pixels to ensure that generated kernel will be fitted inside the image
        int size = 2 * radius + 6;
        size += size%2;
        QImage* kernelImage = new QImage(size, size, QImage::Format_RGB32);
        kernelImage->fill(Qt::black);
        // Prepare painter to have antialiasing and sub-pixel accuracy
        QPainter kernelPainter(kernelImage);
        kernelPainter.setRenderHint(QPainter::Antialiasing);
        kernelPainter.setBrush(QBrush(Qt::white));
        // Draw circle
        kernelPainter.drawEllipse(QPointF(0.5+kernelImage->width()/2.0, 0.5+kernelImage->height()/2.0), radius, radius);
        kernelPainter.end();
        // Draw edge correction - add ring (with radius=kernelRadiaus) blurred by Gauss to the drawed circle
        int center = size/2;
        for (int y = 0; y<size; y++) {
            for (int x = 0; x<size; x++) {
                double dist = pow((double)x-center,2) + pow((double)y-center,2);
                dist = sqrt(dist);
                if (dist <= radius) {
                    double mu = radius;
                    double sigma = radius*edgeFeather/100;
                    // Gaussian normalized by kernelStrength
                    double gaussValue = pow(M_E, -pow((dist-mu)/sigma,2)/2);
                    gaussValue *= 255*(correctionStrength)/100;
                    // Circle pixel value normalized by 1-kernelStrength
                    int curValue = qRed(kernelImage->pixel(x,y));
                    if (correctionStrength >= 0) {
                        curValue *= (100-correctionStrength)/100;
                    }
                    // Sum and check
                    curValue += gaussValue;
                    if (curValue < 0) {
                        curValue = 0;
                    }
                    if (curValue > 255) {
                        curValue = 255;
                    }
                    kernelImage->setPixel(x,y,qRgb(curValue, curValue, curValue));
                }
            }
        }
        return kernelImage;
    }
};
#endif // FOCUSBLUR_H
