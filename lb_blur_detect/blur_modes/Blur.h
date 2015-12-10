#ifndef BLUR_H
#define BLUR_H

#include <QString>
#include <QImage>

/**
 * @brief The Blur class
 *
 * Abstract base class.
 * Used to instantiate different possible blur kernel.
 */
class Blur
{
public:

    Blur(){
        _radius = 0;
        _smooth = 1;
    }
    ~Blur(){}
    double _radius;
    double _smooth;
    const virtual QString getName() const = 0;
    virtual QImage* buildKernelImage() = 0;
};
#endif // BLUR_H
